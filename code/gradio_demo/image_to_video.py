# video generation
import os
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
sys.path.append("./DiffSynth-Studio")
from diffsynth import ModelManager, WanVideoPipeline
from utils_3dscene.nvrender import perform_camera_movement_with_cam_input, load_rail
from utils_3dscene.pipeline_utils_3dscene import write_video
from PIL import Image
import imageio
import argparse
import numpy as np
import torch
import cv2
from torchvision.transforms import v2
from einops import rearrange
import torchvision
import torch.distributed as dist
from pathlib import Path
import json
import torch.multiprocessing as mp
import subprocess
import time
import traceback


MASK_RATIO = 0.
class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, vid_path, mask_path,text, max_num_frames=81, frame_interval=1, num_frames=81, height=720, width=1440, is_i2v=True):

        self.path = [vid_path]
        self.mask_video_path = [mask_path]
        self.text = [text]
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        
        # this should not be center crop
        # should be 
        self.frame_process = v2.Compose([
            #v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image
    
    def crop_and_resize_standard(self, image):
        width, height = image.size
        #scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (self.width, self.height),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = frame
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        
        first_frame = v2.functional.center_crop(first_frame, output_size=(self.height, self.width))
        first_frame = np.array(first_frame)

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames

    
    
    
    

    def load_frames_using_imageio_standard(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize_standard(frame)
            if first_frame is None:
                first_frame = frame
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        
        first_frame = v2.functional.center_crop(first_frame, output_size=(self.height, self.width))
        first_frame = np.array(first_frame)


        return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, 1, (1,))[0]
        frames = self.load_frames_using_imageio_standard(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        mask_video_path = self.mask_video_path[data_id]


        video = self.load_video(path)
        mask_video = self.load_video(mask_video_path)
        #print(video.max(), video.min(), mask_video.max(), mask_video.min())
        mask_bool = mask_video < MASK_RATIO
        masked_video = video.clone()
        masked_video[mask_bool] = -1.
        data = {"text": text, "video": video, "path": path, "masked_video": masked_video, "mask_video":mask_video}
        # HACK: save video for visualize

        return data
    

    def __len__(self):
        return len(self.path)



training_iters=3000 # optimization iterations
num_of_point_cloud=3000000 # number of point cloud unprojected from depth map
num_views_per_view=3 # inserted between adjacent camera poses
img_sample_interval=1 # images selected during training to optimize 3DGS



def optimize_mp4_faststart(input_path):
    # 创建临时输出文件
    dir_name = os.path.dirname(input_path)
    output_tmp = os.path.join(dir_name, "generated_faststart.mp4")
    cmd = [
        "ffmpeg", "-y", 
        "-i", input_path,
        "-movflags", "+faststart",
        output_tmp
    ]
    subprocess.run(cmd, check=True)
    os.replace(output_tmp, input_path)



def simple_filename(prompt):
    filename = re.sub(r'[^\w\s-]', '', prompt)  
    filename = re.sub(r'\s+', '_', filename)    
    return f"{filename[:50]}"              

class Video_Gen_Multi:
    def __init__(self, device=torch.device("cuda:0"), use_usp=False, ulysses_size=2,ring_size=1,max_gpus=2,resolution=720):
        print("多卡初始化")
        self.use_usp = use_usp
        
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'

        self.in_q_list = None
        self.out_q = None
        self.inference_pids = None
        self.ulysses_size = ulysses_size
        self.ring_size = ring_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        is_720p = resolution == 720
        self.is_720p=is_720p
        self.tgt_resolution = (1440,720) if is_720p else (960,480)

        self.dynamic_load(max_gpus)
 
    def dynamic_load(self,max_gpus=2):
        if hasattr(self, 'inference_pids') and self.inference_pids is not None:
            return
        # 例如最多使用2个GPU
        gpu_infer = min(int(os.environ.get('LOCAL_WORLD_SIZE', torch.cuda.device_count())), max_gpus)
        print(f"实际使用GPU数量: {gpu_infer}")

        pmi_rank = int(os.environ['RANK'])
        print(f"pmi_rank={pmi_rank}")
        pmi_world_size = int(os.environ['WORLD_SIZE'])
        print(f"pmi_world_size={pmi_world_size}")
        in_q_list = [
            torch.multiprocessing.Manager().Queue() for _ in range(gpu_infer)
        ]
    
        out_q = torch.multiprocessing.Manager().Queue()
        initialized_events = [
            torch.multiprocessing.Manager().Event() for _ in range(gpu_infer)
        ]
        ctx = mp.get_context("spawn")
        processes = []
        for gpu in range(gpu_infer):
            p = ctx.Process(target=self.mp_worker, args=(
                gpu, gpu_infer, pmi_rank, pmi_world_size,
                in_q_list, out_q, initialized_events,self
            ))
            p.start()
            processes.append(p)

        self.inference_pids = [p.pid for p in processes]
        all_initialized = False
        while not all_initialized:
            # print(f"all_initialized={all_initialized}")
            all_initialized = all(
                event.is_set() for event in initialized_events)
            if not all_initialized:
                time.sleep(0.1)
        print('Inference model is initialized', flush=True)
        self.in_q_list = in_q_list
        self.out_q = out_q
        # self.inference_pids = context.pids()
        self.initialized_events = initialized_events
        print("动态加载完成")


    def distributed_file_wait(self, file_path, timeout=30.0, check_interval=0.5):
        """分布式环境下等待文件出现"""
        if dist.get_rank() == 0:
            # Rank 0 负责实际检查文件
            start = time.time()
            while not os.path.exists(file_path):
                if time.time() - start > timeout:
                    raise TimeoutError(f"等待文件超时: {file_path}")
                time.sleep(check_interval)
        
        # 同步检查结果
        if dist.is_initialized():
            exists = torch.tensor([os.path.exists(file_path)], dtype=torch.int).to(device)
            dist.broadcast(exists, 0)
            if not exists.item():
                raise FileNotFoundError(f"文件不存在: {file_path}")
        return True
    def mp_worker(self, gpu, gpu_infer, pmi_rank, pmi_world_size, in_q_list,
                  out_q, initialized_events, work_env):
        try:
            world_size = pmi_world_size * gpu_infer
            rank = pmi_rank * gpu_infer + gpu
            print("world_size", world_size, "rank", rank, flush=True)
            torch.cuda.set_device(gpu)
            # 每个进程设置自己独立的环境变量
            # os.environ['RANK'] = str(rank)
            dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    rank=rank,
                    world_size=world_size)
            
            from xfuser.core.distributed import (
                    init_distributed_environment,
                    initialize_model_parallel,
                )
            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size()
            )

            world_size=dist.get_world_size()
            if world_size <= 4:
                self.ring_size = world_size  # 单组全连接
                self.ulysses_size = 1        # 禁用跨节点
            else:  # 多机场景
                self.ring_size = 4  
                self.ulysses_size = world_size // 4

            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=self.ring_size or 1,
                ulysses_degree=self.ulysses_size or 1,
            )
            print(f"sequence_parallel_degree={dist.get_world_size()},ring_degree={self.ring_size}, ulysses_degree={self.ulysses_size} ")

            debug_q = False
            print("初始化模型之前")
            # 初始化模型
            if  debug_q:
                print("skip 初始化")
            else:
                self.load_model(rank,gpu)
            torch.cuda.empty_cache()
            event = initialized_events[gpu]
            in_q = in_q_list[gpu]
            event.set()
            print("进入循环")
            while True:
                task = in_q.get()
                if task is None:
                    print(f"Rank {rank} 收到退出信号")
                    break
                print(f"收到任务: task={task}")
                seed, panorama_path, prompt,angle,movement_range, movement_mode, output_dir= task
                if debug_q:
                    result_path = "test.mp4"
                else:
                    result_path = self.video_inference(seed, panorama_path, prompt,angle,movement_range, movement_mode, output_dir)
                    # result_path="test.mp4"
                torch.cuda.empty_cache()
                if dist.is_initialized():
                    dist.barrier()
                if rank==0: 
                    out_q.put(result_path)
        except Exception as e:
            trace_info = traceback.format_exc() 
            print(trace_info, flush=True)
            print(f"Rank {rank} 初始化失败: {str(e)}", flush=True)
            
            raise
        finally:
            # 7. 关键修复2：确保资源释放
            print(f"Rank {rank} 开始清理资源...", flush=True)
            trace_info = traceback.format_exc()
            print(trace_info, flush=True)
            if dist.is_initialized():
                dist.barrier()  # 确保所有操作完成
                dist.destroy_process_group()
            torch.cuda.empty_cache()
            print(f"Rank {rank} 已安全退出", flush=True)


    def gen_video(
            self,
            seed=119223,
            panorama_path="/datasets_3d/haoyuan.li/AIGC/matrix_gradio/Matrix-3D-main/output/example1/pano_img.jpg",
            prompt="a quite small villiage with lots of trees",
            angle=90.,
            movement_range=0.3,
            movement_mode="s_curve",
            output_dir="/datasets_3d/haoyuan.li/AIGC/matrix_gradio/Matrix-3D-main/output/example1/",
            json_path="",
        ):
        input_data =(seed, panorama_path,prompt, angle,movement_range,movement_mode,output_dir)
        print("开始推理")
        for in_q in self.in_q_list:
            in_q.put(input_data)
        value_output = self.out_q.get()
        print(f"value_output={value_output}")
        return value_output 


    def load_model(self, rank,gpu):
        print(f"Rank {rank} 开始加载模型")
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        BASE_DIR = Path(__file__).parent.absolute()
        BASE_DIR=BASE_DIR.parent
        print(f"BASE_DIR={BASE_DIR}")
        BASE_DIR=BASE_DIR.parent
        print(f"final BASE_DIR={BASE_DIR}")
        model_manager.load_models(
            [str(BASE_DIR /"checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")],
            torch_dtype=torch.float32, # Image Encoder is loaded with float32
        ) 
        if self.is_720p:
            model_manager.load_models(
                [str(BASE_DIR /"checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")],
                torch_dtype=torch.float32, # Image Encoder is loaded with float32
            ) 


            model_manager.load_models([
                [str(BASE_DIR / f"checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-0000{i}-of-00007.safetensors") for i in range(1, 8)],
                str(BASE_DIR / "checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth"),
                str(BASE_DIR / "checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth")
            ])

            model_manager.load_lora(str(BASE_DIR /"checkpoints/Wan-AI/wan_lora/pano_video_gen_720p.bin"), lora_alpha=1.0)
        else:
            model_manager.load_models(
                [str(BASE_DIR /"checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")],
                torch_dtype=torch.float32, # Image Encoder is loaded with float32
            ) 

            model_manager.load_models([
                [str(BASE_DIR / f"checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-0000{i}-of-00007.safetensors") for i in range(1, 8)],
                str(BASE_DIR / "checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth"),
                str(BASE_DIR / "checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth")
            ])

            model_manager.load_lora(str(BASE_DIR /"checkpoints/Wan-AI/wan_lora/pano_video_gen_480p.ckpt"), lora_alpha=1.0)

        model = WanVideoPipeline.from_model_manager(
            model_manager,
            device=f"cuda:{rank}",
            use_usp=True if dist.get_world_size() > 1 else False
        )
        print(f"rank={rank}, gpu={gpu}")
        # shard_fn = partial(shard_model, device_id=gpu)
        torch.cuda.empty_cache()
        dist.barrier()
        self.pipe=model
        # print(f"Rank {rank}: 进入 shard_fn 前", flush=True)
        # self.pipe = shard_fn(model)
        # print("模型分片完成")
        self.pipe.enable_vram_management(num_persistent_param_in_dit=None)
        print(f"Rank {rank} 模型加载完成")

    def video_inference(
            self,
            seed=119223,
            panorama_path="/datasets_3d/haoyuan.li/AIGC/matrix_gradio/Matrix-3D-main/output/example1/pano_img.jpg",
            prompt="a quite small villiage with lots of trees",
            angle=90.,
            movement_range=0.3,
            movement_mode="s_curve",
            output_dir="/datasets_3d/haoyuan.li/AIGC/matrix_gradio/Matrix-3D-main/output",
            json_path="",
        ):
            print("生成视频")
            rank_get=dist.get_rank()
            print(f"当前rank={rank_get}")
            device = f"cuda:{rank_get}"

            case_dir = os.path.abspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            print(f"panorama_path={panorama_path}")
            panorama = cv2.resize(cv2.imread(panorama_path, cv2.IMREAD_UNCHANGED),(2048,1024),interpolation=cv2.INTER_AREA)


            input_image_path = os.path.join(case_dir, "da3.png")
            cv2.imwrite(input_image_path, panorama)
            

            if dist.get_rank() == 0:
                print("\n\nperform depth-anything-3...\n\n")
                os.system(f"cd code/DA3 && python scripts/infer_panorama_da3.py --input {os.path.abspath(input_image_path)} --output {case_dir} --device {device} --maps")

            depth_path = os.path.join(case_dir, "da3","depth.exr")
            mask_path = os.path.join(case_dir, "da3", "mask.png")
            
            
            print(f"{os.path.exists(mask_path)},{mask_path}")
            dist.barrier()  # 所有GPU等待rank0完成
            max_retries = 3
            for _ in range(max_retries):
                try:
                    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    if mask is not None:
                        mask = mask[:,:] > 127
                        break
                except Exception as e:
                    print(f"Retrying mask read: {e}")
                    time.sleep(1)
            else:
                raise RuntimeError(f"Failed to read mask after {max_retries} retries")
                print(f"[GPU {device_id}] mask_path exists: {os.path.exists(mask_path)}")
                print(f"[GPU {device_id}] mask shape: {mask.shape if mask is not None else 'None'}")


            # import pdb
            # pdb.set_trace()
            
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)[:,:] > 127
            valid_max = depth[mask].max()
            depth[~mask] = 2. * valid_max
            # generate condition
            if rank_get == 0:
                panorama_torch = (torch.from_numpy(panorama).float()/255.).to(device)
                depth_torch = torch.from_numpy(depth).float().to(device)
                mask_torch = torch.from_numpy(mask).bool().to(device)
                # rendered_rgb, rendered_mask, render_Rts, firstframe_rgb, firstframe_depth = perform_camera_movement(panorama_torch, depth_torch, angle=angle, movement_ratio=movement_range, frame_size=81)
                ########################################? 2025-6-13
                if len(json_path) > 0 and os.path.exists(json_path):
                    rail = load_rail(json_path)
                else:
                    rail = None
                rendered_rgb, rendered_mask, render_Rts, firstframe_rgb, firstframe_depth, angle = perform_camera_movement_with_cam_input(panorama_torch, depth_torch, angle=angle, movement_ratio=movement_range, frame_size=81, preset_rail=rail,mode=movement_mode)
                ###########################################
                # rendered_rgb, rendered_mask, render_Rts, firstframe_rgb, firstframe_depth = perform_camera_movement_new(panorama_torch, depth_torch, angle=angle, movement_ratio=movement_range, frame_size=81, preset_rail=None,mode=movement_mode)

            condition_dir = os.path.join(case_dir,"condition")
            os.makedirs(condition_dir, exist_ok=True)
            camera_path = os.path.join(condition_dir, "cameras.npz")
            if rank_get == 0:
                rendered_rgb_np = (rendered_rgb.cpu().numpy() * 255.).astype(np.uint8)
                rendered_mask_np = (rendered_mask.float()[:,:,:,None].repeat(1,1,1,3).cpu().numpy() * 255.).astype(np.uint8)

                write_video(rendered_rgb_np, os.path.join(condition_dir,"rendered_rgb.mp4"), 12)
                write_video(rendered_mask_np, os.path.join(condition_dir,"rendered_mask.mp4"), 12)
                # 这块存下来的深度包含一个最基础的padding;天空盒的部分用非天空盒的部分的最大深度的2倍填充。
                 ############################? 2025-6-13
                W = firstframe_rgb.shape[1]
                q = int(angle/360. * W + W//2)%W
                mask_torch = torch.cat([mask_torch[:,q:],mask_torch[:,:q]],dim=1)
                mask_torch = torch.cat([mask_torch[:,W//2:],mask_torch[:,:W//2]],dim=1)
                ###################################
                cv2.imwrite(os.path.join(condition_dir,"firstframe_rgb.png"), (firstframe_rgb.cpu().numpy()*255.).astype(np.uint8))
                cv2.imwrite(os.path.join(condition_dir,"firstframe_depth.exr"), (firstframe_depth.cpu().numpy()))
                cv2.imwrite(os.path.join(condition_dir,"firstframe_mask.png"), mask_torch.cpu().numpy().astype(np.uint8)*255)
                
            
                np.savez(camera_path, render_Rts.cpu().numpy())

                #vid_path, mask_path,text,
            dist.barrier()#?再次同步
            dset = TextVideoDataset(vid_path = os.path.join(condition_dir,"rendered_rgb.mp4"), mask_path = os.path.join(condition_dir,"rendered_mask.mp4"), text=prompt,height=self.tgt_resolution[1],width=self.tgt_resolution[0])
            cases = dset[0]
            prompt = cases["text"]
            cond_video = ((cases["masked_video"].permute(1,2,3,0) + 1.) / 2. * 255.).cpu().numpy()
            cond_mask = ((cases["mask_video"].permute(1,2,3,0) + 1.) / 2. * 255.).cpu().numpy()
            #print(prompt[i])
            video = self.pipe(
                prompt=prompt+" The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.",
                negative_prompt="The video is not of a high quality, it has a low resolution. Distortion. strange artifacts.",
                cfg_scale=5.0,
                num_frames=81,
                num_inference_steps=50,
                seed=seed, tiled=True,
                height=self.tgt_resolution[1],
                width=self.tgt_resolution[0],
                cond_video = cond_video,
                cond_mask = cond_mask
            )
            if rank_get == 0:
                generated_dir = os.path.join(case_dir,"generated")
                os.makedirs(generated_dir, exist_ok=True)
                result = []
                for j in range(81):
                    generated_image = np.array(video[j])[:,:,::-1]
                    result.append(generated_image)
                
                write_video(result, os.path.join(generated_dir,"generated.mp4"), 24)
                del video,result

                output_path = os.path.join(generated_dir,"generated.mp4")
                print(f"output_path={output_path}")
                #重写视频
                optimize_mp4_faststart(output_path)
                return output_path

    def shutdown(self):
        print("发送关闭信号给所有子进程")
        for q in self.in_q_list:
            q.put(None)
        for pid in self.inference_pids:
            os.waitpid(pid, 0)
        print("所有子进程已安全退出")

class Video_Gen_Single:
    def __init__(self,device=torch.device("cuda:0"), use_usp=False,ulysses_size=1, resolution=720):
        self.device=device if torch.cuda.is_available() else 'cpu'
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        BASE_DIR = Path(__file__).parent.absolute()
        BASE_DIR=BASE_DIR.parent
        print(f"BASE_DIR={BASE_DIR}")
        BASE_DIR=BASE_DIR.parent
        print(f"final BASE_DIR={BASE_DIR}")
        model_manager.load_models(
            [str(BASE_DIR /"checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")],
            torch_dtype=torch.float32, # Image Encoder is loaded with float32
        ) 
        # BASE_DIR = Path(__file__).parent.absolute()
        # BASE_DIR=BASE_DIR.parent

        # model_manager.load_models([
        #     [
        #             "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/panorama_scene_generation/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
        #             "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/panorama_scene_generation/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
        #             "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/panorama_scene_generation/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
        #             "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/panorama_scene_generation/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
        #             "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/panorama_scene_generation/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
        #             "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/panorama_scene_generation/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
        #             "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/panorama_scene_generation/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
        #         ],
        #     "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/panorama_scene_generation/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
        #     "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/panorama_scene_generation/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
        # ])
        # model_manager.load_lora("/ai-video-sh/zhongqi.yang/code/zhongqi.yang/panorama_scene_generation/DiffSynth-Studio/lora_weights/epoch=56-step=1824.ckpt", lora_alpha=1.0)
        # #self.pipe = WanVideoPipeline.from_model_manager(model_manager, device=f"cuda:{dist.get_rank()}",use_usp=True if dist.get_world_size() > 1 else False)
        # self.pipe = WanVideoPipeline.from_model_manager(model_manager, device=device,use_usp=False)
        # self.pipe.enable_vram_management(num_persistent_param_in_dit=None)
        is_720p = resolution == 720
        if is_720p:

            model_manager.load_models(
                [str(BASE_DIR /"checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")],
                torch_dtype=torch.float32, # Image Encoder is loaded with float32
            ) 


            model_manager.load_models([
                [str(BASE_DIR / f"checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-0000{i}-of-00007.safetensors") for i in range(1, 8)],
                str(BASE_DIR / "checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth"),
                str(BASE_DIR / "checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth")
            ])

            model_manager.load_lora(str(BASE_DIR /"checkpoints/Wan-AI/wan_lora/pano_video_gen_720p.bin"), lora_alpha=1.0)
        else:
            model_manager.load_models(
                [str(BASE_DIR /"checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")],
                torch_dtype=torch.float32, # Image Encoder is loaded with float32
            ) 

            model_manager.load_models([
                [str(BASE_DIR / f"checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-0000{i}-of-00007.safetensors") for i in range(1, 8)],
                str(BASE_DIR / "checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth"),
                str(BASE_DIR / "checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth")
            ])

            model_manager.load_lora(str(BASE_DIR /"checkpoints/Wan-AI/wan_lora/pano_video_gen_480p.ckpt"), lora_alpha=1.0)

        self.pipe = WanVideoPipeline.from_model_manager(model_manager, device=device,use_usp=False)
        self.pipe.enable_vram_management(num_persistent_param_in_dit=None)
        self.tgt_resolution = (1440,720) if is_720p else (960,480)

    def gen_video(
        self,
        seed=119223,
        panorama_path="/datasets_3d/haoyuan.li/AIGC/matrix_gradio/Matrix-3D-main/output/example1/pano_img.jpg",
        prompt="a quite small villiage with lots of trees",
        angle=90.,
        movement_range=0.3,
        movement_mode="s_curve",
        output_dir="/datasets_3d/haoyuan.li/AIGC/matrix_gradio/Matrix-3D-main/output",
        json_path=None
    ):
        print("生成视频")
        # rank_get=dist.get_rank()
        # device = f"cuda:{rank_get}"
        rank_get=0

        os.makedirs(output_dir, exist_ok=True)
        panorama_name = os.path.basename(os.path.dirname(panorama_path))
        print(f"panorama_name={panorama_name}")
        case_dir = os.path.join(output_dir, panorama_name)
        print(f"case_dir={case_dir}")
        os.makedirs(case_dir,exist_ok=True)
        panorama = cv2.resize(cv2.imread(panorama_path, cv2.IMREAD_UNCHANGED),(2048,1024),interpolation=cv2.INTER_AREA)
        input_image_path = os.path.join(case_dir, "da3.png")
        cv2.imwrite(input_image_path, panorama)


        # perform depth-anything-3 inference;
        #if dist.get_rank() == 0:
        print("\n\nperform depth-anything-3...\n\n")
        device=self.device
        os.system(f"cd code/DA3 && python scripts/infer_panorama_da3.py --input {os.path.abspath(input_image_path)} --output {case_dir} --device {device} --maps")
        depth_path = os.path.join(case_dir, "da3","depth.exr")
        mask_path = os.path.join(case_dir, "da3", "mask.png")
        print(f"{os.path.exists(mask_path)},{mask_path}")

        # import pdb
        # pdb.set_trace()
        
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)[:,:] > 127
        valid_max = depth[mask].max()
        depth[~mask] = 2. * valid_max
        # generate condition
        if rank_get == 0:
            panorama_torch = (torch.from_numpy(panorama).float()/255.).to(device)
            depth_torch = torch.from_numpy(depth).float().to(device)
            mask_torch = torch.from_numpy(mask).bool().to(device)
            # rendered_rgb, rendered_mask, render_Rts, firstframe_rgb, firstframe_depth = perform_camera_movement(panorama_torch, depth_torch, angle=angle, movement_ratio=movement_range, frame_size=81)
            
            #rendered_rgb, rendered_mask, render_Rts, firstframe_rgb, firstframe_depth = perform_camera_movement_new(panorama_torch, depth_torch, angle=angle, movement_ratio=movement_range, frame_size=81, preset_rail=None,mode=movement_mode)
            #rendered_rgb, rendered_mask, render_Rts, firstframe_rgb, firstframe_depth = perform_camera_movement_new(panorama_torch, depth_torch, angle=angle, movement_ratio=movement_range, frame_size=81, preset_rail=None,mode=movement_mode)
            ########################################? 2025-6-13
            if json_path is not None and os.path.exists(json_path) and len(json_path) > 0 :
                rail = load_rail(json_path)
            else:
                rail = None
            rendered_rgb, rendered_mask, render_Rts, firstframe_rgb, firstframe_depth, angle = perform_camera_movement_with_cam_input(panorama_torch, depth_torch, angle=angle, movement_ratio=movement_range, frame_size=81, preset_rail=rail,mode=movement_mode)
            ###########################################
        condition_dir = os.path.join(case_dir,"condition")
        os.makedirs(condition_dir, exist_ok=True)
        camera_path = os.path.join(condition_dir, "cameras.npz")
        if rank_get == 0:
            rendered_rgb_np = (rendered_rgb.cpu().numpy() * 255.).astype(np.uint8)
            rendered_mask_np = (rendered_mask.float()[:,:,:,None].repeat(1,1,1,3).cpu().numpy() * 255.).astype(np.uint8)

            write_video(rendered_rgb_np, os.path.join(condition_dir,"rendered_rgb.mp4"), 12)
            write_video(rendered_mask_np, os.path.join(condition_dir,"rendered_mask.mp4"), 12)
            # 这块存下来的深度包含一个最基础的padding;天空盒的部分用非天空盒的部分的最大深度的2倍填充。
            ############################? 2025-6-13
            W = firstframe_rgb.shape[1]
            q = int(angle/360. * W + W//2)%W
            mask_torch = torch.cat([mask_torch[:,q:],mask_torch[:,:q]],dim=1)
            mask_torch = torch.cat([mask_torch[:,W//2:],mask_torch[:,:W//2]],dim=1)
            ###################################
            cv2.imwrite(os.path.join(condition_dir,"firstframe_rgb.png"), (firstframe_rgb.cpu().numpy()*255.).astype(np.uint8))
            cv2.imwrite(os.path.join(condition_dir,"firstframe_depth.exr"), (firstframe_depth.cpu().numpy()))
            cv2.imwrite(os.path.join(condition_dir,"firstframe_mask.png"), mask_torch.cpu().numpy().astype(np.uint8)*255)
            
        
            np.savez(camera_path, render_Rts.cpu().numpy())

            #vid_path, mask_path,text,
        dset = TextVideoDataset(vid_path = os.path.join(condition_dir,"rendered_rgb.mp4"), mask_path = os.path.join(condition_dir,"rendered_mask.mp4"), text=prompt,height=self.tgt_resolution[1],width=self.tgt_resolution[0])
        cases = dset[0]
        prompt = cases["text"]
        cond_video = ((cases["masked_video"].permute(1,2,3,0) + 1.) / 2. * 255.).cpu().numpy()
        cond_mask = ((cases["mask_video"].permute(1,2,3,0) + 1.) / 2. * 255.).cpu().numpy()
        #print(prompt[i])
        video = self.pipe(
            prompt=prompt+" The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.",
            negative_prompt="The video is not of a high quality, it has a low resolution. Distortion. strange artifacts.",
            cfg_scale=5.0,
            num_frames=81,
            num_inference_steps=50,
            seed=seed, tiled=True,
            height=self.tgt_resolution[1],
            width=self.tgt_resolution[0],
            cond_video = cond_video,
            cond_mask = cond_mask
        )
        if rank_get == 0:
            generated_dir = os.path.join(case_dir,"generated")
            os.makedirs(generated_dir, exist_ok=True)
            result = []
            for j in range(81):
                generated_image = np.array(video[j])[:,:,::-1]
                result.append(generated_image)
            write_video(result, os.path.join(generated_dir,"generated.mp4"), 24)

            output_path = os.path.join(generated_dir,"generated.mp4")
            #重写视频
            optimize_mp4_faststart(output_path)
            return output_path

def main(args):
    output_dir = args.inout_dir
    panorama_path = os.path.join(output_dir, 'pano_img.jpg')
    prompt_path = os.path.join(output_dir,"prompt.txt")


    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    # Download models
    
    from xfuser.core.distributed import (initialize_model_parallel,
                                        init_distributed_environment)
    init_distributed_environment(
        rank=dist.get_rank(), world_size=dist.get_world_size())

    initialize_model_parallel(
        sequence_parallel_degree=dist.get_world_size(),
        ring_degree=1,
        ulysses_degree=dist.get_world_size(),
    )

    torch.cuda.set_device(dist.get_rank())
    
    print(f"\n\n{panorama_path}\n\n")

    
    with open(prompt_path,"r",encoding="utf-8") as f:
        prompt=f.read()
        print(f"prompt is {prompt}")
    angle = args.angle
    movement_range = args.movement_range
    movement_mode = args.movement_mode
    seed = args.seed
    resolution = args.resolution
    is_720p = resolution == 720
    # if is_720p:
    #     snapshot_download("Wan-AI/Wan2.1-I2V-14B-720P", local_dir="checkpoints/Wan-AI/Wan2.1-I2V-14B-720P")
    # else:
    #     snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir="checkpoints/Wan-AI/Wan2.1-I2V-14B-480P")

    # do other things only in the main rank;
    device = f"cuda:{dist.get_rank()}"
    case_dir = os.path.abspath(output_dir)#os.path.abspath(os.path.join(output_dir, panorama_name))
    os.makedirs(case_dir,exist_ok=True)
    print(f"panorama_path={panorama_path}")
    panorama = cv2.resize(cv2.imread(panorama_path, cv2.IMREAD_UNCHANGED),(2048,1024),interpolation=cv2.INTER_AREA)
    input_image_path = os.path.join(case_dir, "da3.png")
    cv2.imwrite(input_image_path, panorama)
    if dist.get_rank() == 0:
        print("\n\nperform depth-anything-3...\n\n")
        os.system(f"cd code/DA3 && python scripts/infer_panorama_da3.py --input {os.path.abspath(input_image_path)} --output {case_dir} --device {device} --maps")
        depth_path = os.path.join(case_dir, "da3","depth.exr")
        mask_path = os.path.join(case_dir, "da3", "mask.png")
        print(f"{os.path.exists(mask_path)},{mask_path}")
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)[:,:] > 127
        valid_max = depth[mask].max()
        depth[~mask] = 2. * valid_max

        panorama_torch = (torch.from_numpy(panorama).float()/255.).to("cuda")
        depth_torch = torch.from_numpy(depth).float().to("cuda")
        mask_torch = torch.from_numpy(mask).bool().to("cuda")
        
        if len(args.json_path) > 0 and os.path.exists(args.json_path):
            rail = load_rail(args.json_path)
        else:
            rail = None
        rendered_rgb, rendered_mask, render_Rts, firstframe_rgb, firstframe_depth, angle = perform_camera_movement_with_cam_input(panorama_torch, depth_torch, angle=angle, movement_ratio=movement_range, frame_size=81, preset_rail=rail,mode=movement_mode)

    condition_dir = os.path.join(case_dir,"condition")
    os.makedirs(condition_dir, exist_ok=True)
    camera_path = os.path.join(condition_dir, "cameras.npz")
    if dist.get_rank() == 0:
        rendered_rgb_np = (rendered_rgb.cpu().numpy() * 255.).astype(np.uint8)
        rendered_mask_np = (rendered_mask.float()[:,:,:,None].repeat(1,1,1,3).cpu().numpy() * 255.).astype(np.uint8)

        write_video(rendered_rgb_np, os.path.join(condition_dir,"rendered_rgb.mp4"), 12)
        write_video(rendered_mask_np, os.path.join(condition_dir,"rendered_mask.mp4"), 12)

        W = firstframe_rgb.shape[1]
        q = int(angle/360. * W + W//2)%W
        mask_torch = torch.cat([mask_torch[:,q:],mask_torch[:,:q]],dim=1)
        mask_torch = torch.cat([mask_torch[:,W//2:],mask_torch[:,:W//2]],dim=1)

        cv2.imwrite(os.path.join(condition_dir,"firstframe_rgb.png"), (firstframe_rgb.cpu().numpy()*255.).astype(np.uint8))
        cv2.imwrite(os.path.join(condition_dir,"firstframe_depth.exr"), (firstframe_depth.cpu().numpy()))
        cv2.imwrite(os.path.join(condition_dir,"firstframe_mask.png"), mask_torch.cpu().numpy().astype(np.uint8)*255)
        
    
        np.savez(camera_path, render_Rts.cpu().numpy())
        

    # perform panovid generation;
    dist.barrier()
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    if is_720p:
        model_manager.load_models(
            ["./checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
            torch_dtype=torch.float32, # Image Encoder is loaded with float32
        ) 
        BASE_DIR = Path(__file__).parent.absolute()
        BASE_DIR=BASE_DIR.parent

        model_manager.load_models([
            [str(BASE_DIR / f"checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-0000{i}-of-00007.safetensors") for i in range(1, 8)],
            str(BASE_DIR / "checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth"),
            str(BASE_DIR / "checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth")
        ])

        model_manager.load_lora("./checkpoints/Wan-AI/wan_lora/pano_video_gen_720p.bin", lora_alpha=1.0)
    else:
        model_manager.load_models(
            ["./checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
            torch_dtype=torch.float32, # Image Encoder is loaded with float32
        ) 
        BASE_DIR = Path(__file__).parent.absolute()
        BASE_DIR=BASE_DIR.parent

        model_manager.load_models([
            [str(BASE_DIR / f"checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-0000{i}-of-00007.safetensors") for i in range(1, 8)],
            str(BASE_DIR / "checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth"),
            str(BASE_DIR / "checkpoints/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth")
        ])

        model_manager.load_lora("./checkpoints/Wan-AI/wan_lora/pano_video_gen_480p.ckpt", lora_alpha=1.0)

    pipe = WanVideoPipeline.from_model_manager(model_manager, device=f"cuda:{dist.get_rank()}",use_usp=False)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)



    #vid_path, mask_path,text,
    tgt_resolution = (1440,720) if is_720p else (960,480)
    #dset = TextVideoDataset(vid_path = os.path.join(condition_dir,"rendered_rgb.mp4"), mask_path = os.path.join(condition_dir,"rendered_mask.mp4"), text=prompt)
    # (self, vid_path, mask_path,text, max_num_frames=81, frame_interval=1, num_frames=81, height=720, width=1440, is_i2v=True):
    dset = TextVideoDataset(vid_path = os.path.join(condition_dir,"rendered_rgb.mp4"), mask_path = os.path.join(condition_dir,"rendered_mask.mp4"), text=prompt, height=tgt_resolution[1],width=tgt_resolution[0])
    cases = dset[0]
    prompt = cases["text"]
    cond_video = ((cases["masked_video"].permute(1,2,3,0) + 1.) / 2. * 255.).cpu().numpy()
    cond_mask = ((cases["mask_video"].permute(1,2,3,0) + 1.) / 2. * 255.).cpu().numpy()
    #print(prompt[i])
    video = pipe(
        prompt=prompt+" The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.",
        negative_prompt="The video is not of a high quality, it has a low resolution. Distortion. strange artifacts.",
        cfg_scale=5.0,
        num_frames=81,
        num_inference_steps=50,
        seed=seed, tiled=True,
        height=tgt_resolution[1],
        width=tgt_resolution[0],
        cond_video = cond_video,
        cond_mask = cond_mask
    )
    if dist.get_rank() == 0:
        generated_dir = os.path.join(case_dir,"generated")
        generated_path = os.path.join(generated_dir,"generated.mp4")
        
        os.makedirs(generated_dir, exist_ok=True)
        result = []
        for j in range(81):
            generated_image = np.array(video[j])[:,:,::-1]
            result.append(generated_image)
        write_video(result, generated_path, 24)
    
        # gather output;
        gathered_video_name = f"pano_video.mp4"
        all_output_dir = case_dir
        os.makedirs(all_output_dir, exist_ok=True)
        os.system(f"cp {generated_path} {os.path.join(all_output_dir, gathered_video_name)}")
        all_cameras_list = render_Rts.cpu().numpy().tolist()

        pano_camera_path = os.path.join(all_output_dir, "pano_video_cam.json")
        with open(pano_camera_path, "w") as F_:
            F_.write(json.dumps(all_cameras_list,indent=4))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=float, default=0., help="the azimuth angle of camera movement direction. angle=0 means the camera moves towards the center of the panoramic image, angle=90 means the camera moves towards the middle-right direction of the panoramic image")
    parser.add_argument("--movement_range", type=float, default=0.6, help="relative movement range of the camera w.r.t the estimated depth of the input panorama. the value should be between 0~0.8")
    parser.add_argument("--movement_mode", type=str, default="straight", help="the shape of the rail along which the camera moves. choose between ['s_curve','l_curve','r_curve','straight']")
    parser.add_argument("--json_path", type=str, default="", help="predefined camera path. the predefined camera is stored as json file in the format defined in code/generate_example_camera.py")#######2025-6-13
    parser.add_argument("--seed", type=int, default=0, help="the generation seed")
    parser.add_argument("--resolution", type=int, default=720, help="the working resolution of the panoramic video generation model.")
    parser.add_argument("--inout_dir", type=str, default="./output/example1")
    args = parser.parse_args()
    # main(args)
    model=Video_Gen_Single()
    model.gen_video()
