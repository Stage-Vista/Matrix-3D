# Based on VEnhancer
# https://github.com/Vchitect/VEnhancer/blob/main/enhance_a_video.py

from argparse import ArgumentParser, Namespace
import glob
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from easydict import EasyDict
from huggingface_hub import hf_hub_download

from inference_utils import *
from video_to_video.utils.seed import setup_seed
from video_to_video.video_to_video_model import VideoToVideo

from natsort import natsorted


logger = get_logger()
from scripts.sr_video import realesrgan_x2

class VEnhancer:
    def __init__(
        self,
        result_dir="./results/",
        version="v1",
        model_path="",
        solver_mode="fast",
        steps=15,
        guide_scale=7.5,
        s_cond=8,
    ):
        if not model_path:
            self.download_model(version=version)
        else:
            self.model_path = model_path
        assert os.path.exists(self.model_path), "Error: checkpoint Not Found!"
        logger.info(f"checkpoint_path: {self.model_path}")

        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        model_cfg = EasyDict(__name__="model_cfg")
        model_cfg.model_path = self.model_path
        self.model = VideoToVideo(model_cfg)

        steps = 15 if solver_mode == "fast" else steps
        self.solver_mode = solver_mode
        self.steps = steps
        self.guide_scale = guide_scale
        self.s_cond = s_cond

    def enhance_a_video(self, video_path, prompt, up_scale=4, target_fps=24, noise_aug=300, seed=666, suffix="enhance", sr_x2=True, total_noise_level=900):

        save_name = os.path.dirname(video_path).split("/")[-1]
        text = prompt
        logger.info(f"text: {text}")
        caption = text + self.model.positive_prompt

        input_frames, input_fps = load_video(video_path)
        in_f_num = len(input_frames)
        logger.info(f"input frames length: {in_f_num}")
        logger.info(f"input fps: {input_fps}")

        # load_video now guarantees a positive fps; compute how many
        # interpolated frames are needed to approach the desired target fps.
        interp_f_num = max(round(target_fps / input_fps) - 1, 0)
        interp_f_num = min(interp_f_num, 8)
        target_fps = input_fps * (interp_f_num + 1)
        logger.info(f"target_fps: {target_fps}")

        extend = input_frames[0].shape[-2] // 16
        logger.info(f'EXTEND: {extend}')
        video_data = preprocess(input_frames, extend=extend)
        _, _, h, w = video_data.shape
        logger.info(f"input resolution: {(h, w)}")
        target_h, target_w = adjust_resolution(h, w, up_scale)
        logger.info(f"target resolution: {(target_h, target_w)}")

        mask_cond = make_mask_cond(in_f_num, interp_f_num)
        mask_cond = torch.Tensor(mask_cond).long()

        noise_aug = min(max(noise_aug, 0), 300)
        logger.info(f"noise augmentation: {noise_aug}")
        logger.info(f"scale s is set to: {self.s_cond}")

        pre_data = {"video_data": video_data, "y": caption}
        pre_data["mask_cond"] = mask_cond
        pre_data["s_cond"] = self.s_cond
        pre_data["interp_f_num"] = interp_f_num
        pre_data["target_res"] = (target_h, target_w)
        pre_data["t_hint"] = noise_aug

        total_noise_levels = total_noise_level
        setup_seed(seed)

        with torch.no_grad():
            data_tensor = collate_fn(pre_data, "cuda:0")
            output = self.model.test(
                data_tensor,
                total_noise_levels,
                steps=self.steps,
                solver_mode=self.solver_mode,
                guide_scale=self.guide_scale,
                noise_aug=noise_aug,
            )

        # Blend repeated part
        output = blend_h(output, output, extend*2, True, True) 
        # Crop repeated part

        output = tensor2vid(output)
        if sr_x2:
            output_numpy = output.cpu().numpy()
            output_numpy = realesrgan_x2(output_numpy, device=f'cuda:0')
            output = torch.from_numpy(output_numpy)
            output = rearrange(output, "f h w c -> f c h w")
            output = blend_h(output, output, extend*4, True, True) 
            output = rearrange(output, "f c h w -> f h w c")
        f, h, w, c = output.shape
        output = output[:, :, :h*2, :]

        #dir_name = save_name.split("_")[0]
        # os.makedirs(os.path.join(self.result_dir, save_name), exist_ok=True)

        # save_video(output, os.path.join(self.result_dir, save_name), f"{save_name}_{suffix}.mp4", fps=target_fps)
        # return os.path.join(self.result_dir, save_name)

        out_folder = os.path.dirname(video_path)
        ori_fn = os.path.basename(video_path)

        save_video(output, out_folder, ori_fn.replace('.mp4', '_enhance.mp4'), fps=target_fps)
        return out_folder
        
        
        

    def download_model(self, version="v1"):
        REPO_ID = "jwhejwhe/VEnhancer"
        filename = "venhancer_paper.pt"
        if version == "v2":
            filename = "venhancer_v2.pt"
        ckpt_dir = "./checkpoints/"
        os.makedirs(ckpt_dir, exist_ok=True)
        local_file = os.path.join(ckpt_dir, filename)

        if not os.path.exists(local_file):
            logger.info(f"Downloading the VEnhancer checkpoint...")
            hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=ckpt_dir)
        self.model_path = local_file


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--input_path", required=True, type=str, help="input video path")
    parser.add_argument("--save_dir", type=str, default="results", help="save directory")
    parser.add_argument("--version", type=str, default="v1", choices=["v1", "v2"], help="model version")
    parser.add_argument("--model_path", type=str, default="", help="model path")
    parser.add_argument("--sr_x2", action="store_true", help="whether to use realesrgan to upsample")

    parser.add_argument("--prompt", type=str, default="a good video", help="prompt")
    parser.add_argument("--prompt_path", type=str, default="", help="prompt path")
    parser.add_argument("--filename_as_prompt", action="store_true")
    parser.add_argument("--suffix", type=str, default="enhance", help="suffix of video file")

    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--solver_mode", type=str, default="fast", choices=["fast", "normal"], help="fast | normal")
    parser.add_argument("--steps", type=int, default=15)

    parser.add_argument("--noise_aug", type=int, default=200, help="noise augmentation")
    parser.add_argument("--target_fps", type=int, default=24)
    parser.add_argument("--up_scale", type=float, default=2)
    parser.add_argument("--s_cond", type=float, default=8)
    parser.add_argument("--seed", type=float, default=666)
    parser.add_argument("--total_noise_level", type=int, default=900)

    return parser.parse_args()


def main():

    args = parse_args()

    input_path = args.input_path
    prompt = args.prompt
    prompt_path = args.prompt_path
    filename_as_prompt = args.filename_as_prompt
    model_path = args.model_path
    version = args.version
    save_dir = args.save_dir

    noise_aug = args.noise_aug
    up_scale = args.up_scale
    target_fps = args.target_fps
    s_cond = args.s_cond

    steps = args.steps
    solver_mode = args.solver_mode
    guide_scale = args.cfg
    seed = args.seed
    total_noise_level = args.total_noise_level

    # import pdb; pdb.set_trace()
    if os.path.isdir(input_path):
        #file_path_list = natsorted(glob.glob(os.path.join(input_path, "*refine.mp4")))
        subdir_list = natsorted(glob.glob(os.path.join(input_path, "*")))
        # file_path_list = [os.path.join(subdir, os.path.basename(subdir)+"_out.mp4") for subdir in subdir_list ]
        file_path_list = [fn for fn in subdir_list if fn.endswith('_out.mp4')]
        file_path_list = [file_path for file_path in file_path_list if os.path.isfile(file_path)]
    elif os.path.isfile(input_path):
        file_path_list = [input_path]
    else:
        raise TypeError("input must be a directory or video file!")

    #resize first, please adjust resolution here if needed
    for ind, file_path in enumerate(file_path_list):
        print('file_path: ', file_path)
        os.system('/ai-video-sh/haoxiang.guo/HoloTime/ffmpeg-7.0.2-amd64-static/ffmpeg -i {} -vf scale=960:480 {}'.format(file_path, file_path.replace('.mp4', '_resize.mp4')))
    
    venhancer = VEnhancer(
        result_dir=save_dir,
        version=version,
        model_path=model_path,
        solver_mode=solver_mode,
        steps=steps,
        guide_scale=guide_scale,
        s_cond=s_cond,
    )

    prompt_list = None
    if os.path.isfile(prompt_path):
        prompt_list = load_prompt_list(prompt_path)
        assert len(prompt_list) == len(file_path_list)

    for ind, file_path in enumerate(file_path_list):
        file_path = file_path.replace('.mp4', '_resize.mp4')
        logger.info(f"processing video {ind}, file_path: {file_path}")
        
        venhancer.enhance_a_video(file_path, prompt, up_scale, target_fps, noise_aug, seed=seed, suffix=args.suffix, sr_x2=args.sr_x2, total_noise_level = args.total_noise_level)


if __name__ == "__main__":
    main()
