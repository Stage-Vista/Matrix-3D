import os
# os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download, snapshot_download


def download_ckpt(local_dir, repo_id, filename):
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, os.path.basename(filename))
    if not os.path.exists(local_path):
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
        )
        print(f"File has been downloaded to: {file_path}")
    else:
        print(f"File exists already: {local_path}")
os.makedirs("./checkpoints", exist_ok=True)
repo_id_list = [
    "depth-anything/DA3METRIC-LARGE",
    "depth-anything/DA3-SMALL",
    "Iceclear/StableSR",
    "Iceclear/StableSR",
    "Skywork/Matrix-3D",
    "Skywork/Matrix-3D",
    "Skywork/Matrix-3D",
    "Skywork/Matrix-3D",
    "Skywork/Matrix-3D",
]
filename_list = [
    "model.safetensors",
    "model.safetensors",
    "stablesr_turbo.ckpt",
    "vqgan_cfw_00011.ckpt",
    "checkpoints/text2panoimage_lora.safetensors",
    "checkpoints/pano_lrm_480p.pt",
    "checkpoints/pano_video_gen_480p.ckpt",
    "checkpoints/pano_video_gen_720p.bin",
    "checkpoints/pano_video_gen_720p_5b.safetensors",
]
local_dir_list = [
    "./checkpoints/DA3METRIC-LARGE",
    "./checkpoints/DA3-SMALL",
    "./checkpoints/StableSR",
    "./checkpoints/StableSR",
    "./checkpoints/flux_lora",
    "./checkpoints/pano_lrm",
    "./checkpoints/Wan-AI/wan_lora",
    "./checkpoints/Wan-AI/wan_lora",
    "./checkpoints/Wan-AI/wan_lora",
]

N = len(repo_id_list)
for i in range(N):
    repo_id = repo_id_list[i]
    filename = filename_list[i]
    local_dir = local_dir_list[i]
    print(f"\nDownloading {filename} from {repo_id} to local folder {local_dir}...\n")
    download_ckpt(local_dir, repo_id, filename)
