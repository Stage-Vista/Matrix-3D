output_dir=output/example1

# Step1: text to panorama image
python code/panoramic_image_generation.py \
    --mode=t2p \
    --seed=1024 \
    --prompt="a modern fully furnished open plan living room (with a full kitchen containing a fridge, stovetop, microwave, conventional oven, dishwasher, sink, cabinetry and countertops) of an apartment in a high-rise-building, detailed textures, high resolution, functional and aesthetic layout" \
    --output_path=$output_dir

# Or you can choose image to panorama image generation
# python code/panoramic_image_generation.py \
#     --mode=i2p \
#     --input_image_path="./data/image2.jpg" \
#     --output_path=$output_dir

# Step2: panorama image to video generation
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir=$output_dir  \
  --resolution=720 \
  --use_5b_model \
  --enable_vram_management

# Step3: 3d scene extraction
python code/panoramic_video_to_3DScene.py \
    --inout_dir=$output_dir \
    --resolution=720
