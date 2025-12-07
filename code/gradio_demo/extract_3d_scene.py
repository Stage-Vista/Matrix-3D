import os
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
sys.path.append("./DiffSynth-Studio")
from PIL import Image
import imageio
import argparse
import numpy as np
import torch
from torchvision.transforms import v2
from einops import rearrange
import torchvision

# from omegaconf import OmegaConf
# from ..Pano_LRM.pano_infer import SATVideoDiffusionEngine
# import logging

# from typing import Dict, Any
# from datetime import datetime
# from ..Pano_LRM.dataset.panorama import PanoraScene

def setup_logging():
    """Setup logging configuration."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


DEFAULT_CONFIG = {
    'config_paths': ["code/Pano_LRM/lrm_gs_clear.yaml", "code/Pano_LRM/sft_paro_gs.yaml"],
    'ckpt_path': "./checkpoints/pano_lrm/pano_lrm.pt",
    'sample_idx': 0,
    'data_root': 'data',
    'device': None  # Will be auto-detected
}

MASK_RATIO = 0.
class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, vid_path, mask_path,text, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=960, is_i2v=True):

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
training_iters=6000 # optimization iterations
num_of_point_cloud=3000000 # number of point cloud unprojected from depth map
num_views_per_view=3 # 相邻两个相机位姿之间插针数目
img_sample_interval=1 # 训练时每隔多少张图片选取用于优化3DGS



def Extract_Scene(
    device=torch.device("cuda:1"),
    step1_output_dir="/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/output_step1/A_cherry_blossom_forest_with_petals_falling_gently,_a_wooden_bridge_over_a_stream,_and_a_shrine_in_the_background,_anime_style,_ultra-detailed,_soft_pastel_colors,_serene_ambiance_superres",
    prompt="The panoramic video shows a cherry blossom forest with petals falling gently, a wooden bridge over a stream, and a shrine in the background.",
    resulotion=720
    
    ):

        step1_output_dir=step1_output_dir
        generated_dir = os.path.join(step1_output_dir, "generated")
        print(f" generated_dir ={ generated_dir }")
        condition_dir = os.path.join(step1_output_dir, "condition")
        generated_video_path = os.path.join(generated_dir,"generated.mp4")

        if resulotion==720:
            width_following = 1440
            height_following = 720
        else:
            # os.system(f"cd code/VideoSR && python scripts/enhance_video_pipeline.py --version v2 --up_scale 2 --target_fps 20 --noise_aug 100 --solver_mode 'fast' --steps 15 --input_path {generated_video_path} --prompt \'{prompt}\' --save_dir {generated_dir} --suffix enhancement")
            # generated_video_path = os.path.join(generated_dir,"generated_resize_enhance.mp4")
            width_following = 1920
            height_following = 960

        camera_path = os.path.join(condition_dir,"cameras.npz")
        os.system(f"python code/utils_3dscene/panorama_video_to_perspective_depth_sequential.py \
        --device {device} \
        --camera_path {camera_path} \
        --video_path {generated_video_path} \
        --anchor_frame_depth_paths \'{os.path.join(condition_dir,'firstframe_depth.exr')}\' \
        --anchor_frame_mask_paths \'{os.path.join(condition_dir,'firstframe_mask.png')}\' \
        --anchor_frame_indices 0 \
        --output_dir {os.path.join(step1_output_dir,'geom_optim')} \
        --depth_estimation_interval 10 \
        --width {width_following} \
        --height {height_following} \
        ")
        os.system(
        f"python code/utils_3dscene/gs_optim_datagen.py \
            --optimized_depth_dir {os.path.join(step1_output_dir,'geom_optim/data/optimized_depths')} \
            --camera_path {os.path.join(step1_output_dir,'condition/cameras.npz')} \
            --output_dir {os.path.join(step1_output_dir,'geom_optim/data')} \
        "
        )

        cmd_rename = f"mv {os.path.join(step1_output_dir,'geom_optim/data/mv_rgb')} {os.path.join(step1_output_dir,'geom_optim/data/mv_rgb_ori')}"
        os.system(cmd_rename)

        cmd = f"cd code/StableSR && python scripts/sr_val_ddpm_text_T_vqganfin_old.py --init-img {os.path.join(step1_output_dir,'geom_optim/data/mv_rgb_ori')} --outdir {os.path.join(step1_output_dir,'geom_optim/data/mv_rgb')} --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt {os.path.abspath('./checkpoints/StableSR/stablesr_turbo.ckpt')} --ddpm_steps 4 --dec_w 0.5 --seed 42 --n_samples 1 --vqgan_ckpt {os.path.abspath('./checkpoints/StableSR/vqgan_cfw_00011.ckpt')} --colorfix_type wavelet"
        os.system(cmd)

        # import pdb
        # pdb.set_trace()

        gs_input_dir = os.path.join(step1_output_dir,'geom_optim/data')
        gs_output_dir = os.path.join(step1_output_dir,'geom_optim/output')
        os.system(f"cd ./code/Pano_GS_Opt && python train.py -s {gs_input_dir} -m {gs_output_dir} -r 1 --use_decoupled_appearance --save_iterations 3000 6000 9000 12000 15000 --test_iterations 3000 \
        --sh_degree 0 --densify_from_iter 500 --densify_until_iter 1501 --iterations 3000 --eval \
        --img_sample_interval 1 --num_views_per_view 3 --num_of_point_cloud 3000000 --device {device} --distortion_from_iter 6500 --depth_normal_from_iter 6500\
        ")
        # import pdb
        # pdb.set_trace()
        
        all_output_dir = step1_output_dir
        gs_output=os.path.join(gs_output_dir, "point_cloud/iteration_3000/point_cloud.ply")
        os.system(f"cp {gs_output} {os.path.join(all_output_dir, 'generated_3dgs_opt.ply')}")
        gs_output = os.path.join(all_output_dir, 'generated_3dgs_opt.ply')
        print(f"gs_output={gs_output}")
        return gs_output

        
def main(args):
    device = args.device
    step1_output_dir = args.step1_output_dir
    prompt = args.prompt
    apply_superres = args.apply_superres

    generated_dir = os.path.join(step1_output_dir, "generated")
    condition_dir = os.path.join(step1_output_dir, "condition")
    
    generated_video_path = os.path.join(generated_dir,"generated.mp4")
    width_following = 960
    height_following = 480
    if apply_superres:
        os.system(f"cd ./VideoSR && python scripts/enhance_video_pipeline.py --version v2 --up_scale 2 --target_fps 20 --noise_aug 100 --solver_mode 'fast' --steps 15 --input_path {generated_video_path} --prompt \'{prompt}\' --save_dir {generated_dir} --suffix enhancement")
        generated_video_path = os.path.join(generated_dir,"generated_resize_enhance.mp4")
        width_following = 1920
        height_following = 960
    # perform per-frame depth estimation and optimization;
    camera_path = os.path.join(condition_dir,"cameras.npz")
    os.system(f"python panorama_video_to_perspective_depth_sequential.py \
        --device {device} \
        --camera_path {camera_path} \
        --video_path {generated_video_path} \
        --anchor_frame_depth_paths \'{os.path.join(condition_dir,'firstframe_depth.exr')}\' \
        --anchor_frame_mask_paths \'{os.path.join(condition_dir,'firstframe_mask.png')}\' \
        --anchor_frame_indices 0 \
        --output_dir {os.path.join(step1_output_dir,'geom_optim')} \
        --depth_estimation_interval 10 \
        --width {width_following} \
        --height {height_following} \
    ")
    # cut everything into perspective images;


    os.system(
        f"python gs_optim_datagen.py \
            --optimized_depth_dir {os.path.join(step1_output_dir,'geom_optim/data/optimized_depths')} \
            --camera_path {os.path.join(step1_output_dir,'condition/cameras.npz')} \
            --output_dir {os.path.join(step1_output_dir,'geom_optim/data')} \
        "
    )
    # apply gs optimization;
    #for i in range(N):
    gs_input_dir = os.path.join(step1_output_dir,'geom_optim/data')
    gs_output_dir = os.path.join(step1_output_dir,'geom_optim/output')
    os.system(f"cd ./worldgen && python train.py -s {gs_input_dir} -m {gs_output_dir} -r 1 --use_decoupled_appearance --save_iterations 1000 6000 9000 12000 15000 --test_iterations 3000 \
    --sh_degree 0 --densify_from_iter 500 --densify_until_iter 5501 --iterations {training_iters} --eval \
    --img_sample_interval {img_sample_interval} --num_views_per_view {num_views_per_view} --num_of_point_cloud {num_of_point_cloud} --device {device}\
    ")
    # now what?



if __name__ == "__main__":
    '''
    device = args.device
    step1_output_dir = args.step1_output_dir
    prompt = args.prompt
    apply_superres = args.apply_superres
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The panoramic video shows a cherry blossom forest with petals falling gently, a wooden bridge over a stream, and a shrine in the background.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--apply_superres", action="store_true", help="whether to apply superres to generated results")
    parser.add_argument("--step1_output_dir", type=str, default="/ai-video-sh/haoyuan.li/AIGC/matrix3d_inference/debug/debug_video_multi/debug_video_multi")
    # parser.add_argument("--step1_output_dir", type=str, default="/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/output_step1/A_cherry_blossom_forest_with_petals_falling_gently,_a_wooden_bridge_over_a_stream,_and_a_shrine_in_the_background,_anime_style,_ultra-detailed,_soft_pastel_colors,_serene_ambiance_superres")
    args = parser.parse_args()
    

    main(args)