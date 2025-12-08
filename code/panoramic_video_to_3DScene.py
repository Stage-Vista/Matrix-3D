import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
sys.path.append("./DiffSynth-Studio")
import argparse
import cv2


def main(args):
    device = args.device
    step1_output_dir = os.path.abspath(args.inout_dir)

    prompt_path = os.path.join(args.inout_dir, 'prompt.txt')

    with open(os.path.abspath(prompt_path), "r", encoding="utf-8") as f:
        prompt = f.read()
        print(f"prompt is {prompt}")

    generated_dir = os.path.join(step1_output_dir, "generated")
    condition_dir = os.path.join(step1_output_dir, "condition")

    generated_video_path = os.path.join(generated_dir, "generated.mp4")
    if args.resolution == 720:
        width_following = 1440
        height_following = 720
    else:
        os.system(
            f"cd code/VideoSR && python scripts/enhance_video_pipeline.py --version v2 --up_scale 2 --target_fps 20 --noise_aug 100 --solver_mode 'fast' --steps 15 --input_path {generated_video_path} --prompt \'{prompt}\' --save_dir {generated_dir} --suffix enhancement")
        generated_video_path = os.path.join(generated_dir, "generated_resize_enhance.mp4")
        width_following = 1920
        height_following = 960

    camera_path = os.path.join(condition_dir, "cameras.npz")

    # Step 1: Generate optimized depths from panorama video
    os.system(f"python code/utils_3dscene/panorama_video_to_perspective_depth_sequential.py \
        --device {device} \
        --camera_path {camera_path} \
        --video_path {generated_video_path} \
        --anchor_frame_depth_paths \'{os.path.join(condition_dir, 'firstframe_depth.exr')}\' \
        --anchor_frame_mask_paths \'{os.path.join(condition_dir, 'firstframe_mask.png')}\' \
        --anchor_frame_indices 0 \
        --output_dir {os.path.join(step1_output_dir, 'geom_optim')} \
        --depth_estimation_interval 10 \
        --width {width_following} \
        --height {height_following} \
    ")

    # Step 2: Generate GS training data (world_matrix.npz, para.json, mv_rgb/, mv_depth/)
    optimized_depth_dir = os.path.join(step1_output_dir, 'geom_optim/data/optimized_depths')
    gs_data_dir = os.path.join(step1_output_dir, 'geom_optim/data')
    os.system(f"python code/utils_3dscene/gs_optim_datagen.py \
        --optimized_depth_dir {optimized_depth_dir} \
        --camera_path {camera_path} \
        --output_dir {gs_data_dir} \
    ")

    # Optional: Apply StableSR super-resolution to mv_rgb images
    # If you want to skip StableSR, comment out the following block
    # cmd_rename = f"mv {os.path.join(step1_output_dir,'geom_optim/data/mv_rgb')} {os.path.join(step1_output_dir,'geom_optim/data/mv_rgb_ori')}"
    # os.system(cmd_rename)
    # cmd = f"cd code/StableSR && python scripts/sr_val_ddpm_text_T_vqganfin_old.py --init-img {os.path.join(step1_output_dir,'geom_optim/data/mv_rgb_ori')} --outdir {os.path.join(step1_output_dir,'geom_optim/data/mv_rgb')} --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt {os.path.abspath('./checkpoints/StableSR/stablesr_turbo.ckpt')} --ddpm_steps 4 --dec_w 0.5 --seed 42 --n_samples 1 --vqgan_ckpt {os.path.abspath('./checkpoints/StableSR/vqgan_cfw_00011.ckpt')} --colorfix_type wavelet"
    # os.system(cmd)

    # Step 3: Apply GS optimization
    gs_input_dir = os.path.join(step1_output_dir, 'geom_optim/data')
    gs_output_dir = os.path.join(step1_output_dir, 'geom_optim/output')
    os.system(f"cd ./code/Pano_GS_Opt && python train.py -s {gs_input_dir} -m {gs_output_dir} -r 1 --use_decoupled_appearance --save_iterations 3000 6000 9000 12000 15000 --test_iterations 3000 \
    --sh_degree 0 --densify_from_iter 500 --densify_until_iter 1501 --iterations 3000 --eval \
    --img_sample_interval 1 --num_views_per_view 3 --num_of_point_cloud 3000000 --device {device} --distortion_from_iter 6500 --depth_normal_from_iter 6500\
    ")

    # Step 4: Gather results
    all_output_dir = step1_output_dir
    gs_path = os.path.join(gs_output_dir, "point_cloud/iteration_3000/point_cloud.ply")
    os.system(f"cp {gs_path} {os.path.join(all_output_dir, 'generated_3dgs_opt.ply')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="the device on which the 3d scene generation runs")
    parser.add_argument("--inout_dir", type=str, default="./output/example1",
                        help="the directory storing the input and output result")
    parser.add_argument("--resolution", type=int, default=720, help="the working resolution of the 3D scene generation")
    args = parser.parse_args()

    main(args)