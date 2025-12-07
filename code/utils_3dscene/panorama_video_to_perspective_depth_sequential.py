import os
import sys
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
sys.path.append("code")
from utils_3dscene.pipeline_utils_3dscene import get_video_frames, warp_depth_to_tgt, depth_edge, optimize_depth
# from ViewCrafter.viewcrafter import ViewCrafter_Completion
from utils_3dscene.nvrender import get_mesh_from_pano_Rt, depth_edge_torch
import argparse
import numpy as np
import torch
import cv2
# from basicsr.archs.rrdbnet_arch import RRDBNet
# from basicsr.utils.download_util import load_file_from_url

# from realesrgan import RealESRGANer
# from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import shutil

import trimesh
OPTIMAL_SPLIT_FRAME_SIZE = 49


DA3_SCRIPT_CMD = "cd code/DA3 && python scripts/infer_panorama_da3.py"
def apply_warp_fix(warped_depth, warped_mask):
    warped_depth_valid = warped_depth[warped_mask]
    warped_depth[~warped_mask] = warped_depth_valid.max() * 2.
    return warped_depth
def main(args):

    device = args.device
    video_path = args.video_path
    camera_path = args.camera_path
    anchor_frame_depth_paths = args.anchor_frame_depth_paths
    anchor_frame_mask_paths = args.anchor_frame_mask_paths
    anchor_frame_indices = args.anchor_frame_indices

    output_dir = args.output_dir
    depth_estimation_interval = args.depth_estimation_interval
    # each frame is cut into 15views. which is fixed. 
    width = args.width
    height = args.height
    
    
    # ok;
    #load video frames and depths;
    video_frames = get_video_frames(video_path)
    anchor_depths = [cv2.resize(cv2.imread(i, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH),(width,height)) for i in anchor_frame_depth_paths]
    anchor_masks = []
    for i,p in enumerate(anchor_frame_mask_paths):
        if os.path.exists(p):
            anchor_masks.append(cv2.resize(cv2.imread(p, cv2.IMREAD_UNCHANGED),(width,height))>127)
        else:
            anchor_masks.append(anchor_depths[i] < 0.9 * anchor_depths[i].max())

    #load cameras;
    all_cameras = np.load(camera_path)["arr_0"]

    all_generated_frames = []
    N = len(video_frames)
    N_anchors = len(anchor_frame_indices)
    os.makedirs(output_dir, exist_ok=True)
    moge_output_dir = os.path.join(output_dir, "da3")
    data_output_dir = os.path.join(output_dir, "data")
    optimized_depth_dir = os.path.join(output_dir, "data", "optimized_depths")
    mv_rgb_dir = os.path.join(output_dir, "data", "mv_rgb")
    mv_depth_dir = os.path.join(output_dir, "data", "mv_depth")
    print(f"In Panorama ,output_dir={output_dir}")


    os.makedirs(moge_output_dir, exist_ok=True)
    print(f"In Panorama ,moge_output_dir={moge_output_dir}")
    os.makedirs(data_output_dir, exist_ok=True)
    os.makedirs(optimized_depth_dir, exist_ok=True)
    os.makedirs(mv_rgb_dir, exist_ok=True)
    os.makedirs(mv_depth_dir, exist_ok=True)

    last_optimized_depth = []
    last_optimized_mask = []
    last_optimized_Rt = []
    # import pdb
    # pdb.set_trace()
    for i in range(N):
        if i in anchor_frame_indices:
            cur_frame = video_frames[i]
            anchor_index = -1
            anchor_depth = None
            anchor_mask = None
            for j in range(N_anchors):
                if anchor_frame_indices[j] == i:
                    anchor_index = anchor_frame_indices[j]
                    anchor_depth = anchor_depths[j]
                    anchor_mask = anchor_masks[j]
                    break
            
            # predict per frame depth and calculate fg mask and seam mask
            cur_camera = all_cameras[i]

            cv2.imwrite(os.path.join(optimized_depth_dir, f"{i:04d}.exr"), cv2.resize(anchor_depth,(width, height)))
            cv2.imwrite(os.path.join(optimized_depth_dir, f"{i:04d}_mask.png"), cv2.resize(anchor_mask.astype(np.uint8)*255,(width, height)))
            cv2.imwrite(os.path.join(optimized_depth_dir, f"{i:04d}_rgb.png"),cv2.resize(cur_frame,(width,height)))

            last_optimized_depth.append(cv2.resize(anchor_depth,(width, height)))
            last_optimized_mask.append(cv2.resize(anchor_mask.astype(np.uint8)*255,(width, height))>127)
            last_optimized_Rt.append(cur_camera)
        elif i % depth_estimation_interval == 0:
            cur_frame = video_frames[i]
            if len(last_optimized_depth) > 0:
                anchor_depth = last_optimized_depth[-1]
                anchor_mask = last_optimized_mask[-1]
                anchor_camera = last_optimized_Rt[-1]
            # predict per frame depth and calculate fg mask and seam mask
            cur_camera = all_cameras[i]
            #anchor_camera = all_cameras[anchor_index]
            
            
            input_image_path = os.path.join(moge_output_dir, "input.png")
            cv2.imwrite(input_image_path, cv2.resize(cur_frame,(width,height)))
            print(width, height)
            os.system(f"{DA3_SCRIPT_CMD} --input {input_image_path} --output {moge_output_dir} --device {device} --maps")
            print(f"moge_output_dir={moge_output_dir}")
            depth_dir = os.path.join(moge_output_dir, "da3")
            cur_depth = cv2.imread(os.path.join(depth_dir, "depth.exr"), cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
            cur_fgmask = cv2.imread(os.path.join(depth_dir, "mask.png"), cv2.IMREAD_UNCHANGED)[:,:] > 127
            cur_seam_mask = ~depth_edge(cur_depth, rtol=0.05)
            
            print(f"\n\nanchor depth: {anchor_depth.max()} {anchor_depth.min()}\n\n")

            anchor_warp_apply_fg_mask = (~anchor_mask).sum() > 1000
            warped_depth, warped_depth_mask = warp_depth_to_tgt(torch.from_numpy(anchor_depth).to(device), torch.from_numpy(anchor_camera).to(device), torch.from_numpy(cur_camera).to(device)[None], apply_skybox_mask = anchor_warp_apply_fg_mask)
            
            warped_mesh = get_mesh_from_pano_Rt(warped_depth[0],warped_depth_mask[0], cur_camera)
            warped_mesh.export(os.path.join(optimized_depth_dir, f"{i:04d}_warped_mesh.obj"))
            cur_depth_fixed = apply_warp_fix(cur_depth, cur_fgmask)
            

            # optimize_depth(warped_depth, cur_depth, warped_depth_fg_mask, cur_depth_seam_mask, cur_depth_fg_mask, bg_padding_ratio = 2.):
            optimized_depth, optimized_mask = optimize_depth(warped_depth[0],cur_depth_fixed, warped_depth_mask[0], cur_seam_mask, cur_fgmask)
            
            #optimized_depth = apply_warp_fix(optimized_depth, optimized_mask)

            
            optimized_depth_edge = (~depth_edge(optimized_depth, rtol=0.05))
            mesh_optimized = get_mesh_from_pano_Rt(optimized_depth,warped_depth_mask[0] * optimized_depth_edge, cur_camera)
            mesh_optimized.export(os.path.join(optimized_depth_dir, f"{i:04d}_mesh_estim.obj"))
            # split the optimized depth to several views for optimization;

            skybox_depth = torch.from_numpy(np.ones_like(anchor_depth) * anchor_depth.max() * 2).to(device)

            skybox_warped_depth, _ = warp_depth_to_tgt(skybox_depth, torch.from_numpy(anchor_camera).to(device), torch.from_numpy(cur_camera).to(device)[None], apply_skybox_mask=False, apply_seam_mask=False)

            optimized_depth[~optimized_mask] = skybox_warped_depth[0][~optimized_mask]
            cv2.imwrite(os.path.join(optimized_depth_dir, f"{i:04d}.exr"), cv2.resize(optimized_depth,(width, height)))
            cv2.imwrite(os.path.join(optimized_depth_dir, f"{i:04d}_mask.png"), cv2.resize(optimized_mask.astype(np.uint8)*255,(width, height)))
            cv2.imwrite(os.path.join(optimized_depth_dir, f"{i:04d}_rgb.png"),cv2.resize(cur_frame,(width,height)))
            # with skybox applied;
            last_optimized_depth.append(cv2.resize(optimized_depth,(width, height)))
            last_optimized_mask.append(cv2.resize(optimized_mask.astype(np.uint8)*255,(width, height))>127)
            last_optimized_Rt.append(cur_camera)
                
if __name__ == "__main__":
    '''
        device = args.device
        video_path = args.video_path
        camera_path = args.camera_path
        anchor_frame_depth_paths = args.anchor_frame_depth_paths
        anchor_frame_indices = args.anchor_frame_indices

        output_dir = args.output_dir
        depth_estimation_interval = args.depth_estimation_interval
        # each frame is cut into 15views. which is fixed. 
        width = args.width
        height = args.height
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--device", type=str, default="cuda:1")
    #
    parser.add_argument("--camera_path", type=str, default="/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/output/A_boat_on_lake,_trees_and_rocks_near_the_lake._a_house_and_port_in_front_of_a_house,_anime_style_superres/condition/cameras.npz")
    parser.add_argument("--video_path", type=str, default="/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/output/A_boat_on_lake,_trees_and_rocks_near_the_lake._a_house_and_port_in_front_of_a_house,_anime_style_superres/generated/generated_resize_enhance.mp4")
    parser.add_argument("--anchor_frame_depth_paths", type=str, nargs="+", default=["/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/output/A_boat_on_lake,_trees_and_rocks_near_the_lake._a_house_and_port_in_front_of_a_house,_anime_style_superres/condition/firstframe_depth.exr",])
    parser.add_argument("--anchor_frame_mask_paths", type=str, nargs="+", default=["/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/output/A_boat_on_lake,_trees_and_rocks_near_the_lake._a_house_and_port_in_front_of_a_house,_anime_style_superres/condition/firstframe_mask.png",])
    parser.add_argument("--anchor_frame_indices", type=int, nargs="+", default=[0])
    parser.add_argument("--output_dir", type=str, default="/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/output/A_boat_on_lake,_trees_and_rocks_near_the_lake._a_house_and_port_in_front_of_a_house,_anime_style_superres/geom_optim_new")
    parser.add_argument("--depth_estimation_interval", type=int, default=10)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=960)

    args = parser.parse_args()
    main(args)
            

    
    