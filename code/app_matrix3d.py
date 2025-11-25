import os
import sys
os.environ["HF_HOME"] = "/datasets_3d/common/huggingface/huggingface/"
# os.environ["GRADIO_FRPC_PATH"] = "/datasets_3d/haoyuan.li/AIGC/gradio_frpc/frpc_linux_amd64_v0.3"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
sys.path.append("./DiffSynth-Studio")
import argparse
import gradio as gr
# from gradio_litmodel3d import LitModel3D
import numpy as np
import torch
import shutil
import time
import re
import signal

from PIL import Image 
from safetensors.torch import load_file

from pano_init.utils.pipeline_flux import FluxPipeline
from pano_init.i2p_model import i2pano
from gradio_demo.image_to_video import Video_Gen_Single,Video_Gen_Multi
from gradio_demo.extract_3d_scene import Extract_Scene
video_mode="multi"
# video_mode="single"
max_gpus=2
css = """
.light-button { background-color: #e0e0e0 !important; color: #888 !important; }
.dark-button { background-color: #4f46e5 !important; color: white !important; }
"""
lora_path="./checkpoints/flux_lora/pano_image_lora.safetensors"
i2p_Pipeline=None
t2p_Pipeline=None
video_Pipeline=None
current_resolution="720p"
pano_prompt=None
pano_path="/ai-video-sh/zhongqi.yang/code/zhongqi.yang/panorama_scene_generation/panorama_scene_generation/output_step1/save/save.png"
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
output_dir = os.path.join(project_root, "output")
TMP_DIR = os.path.join(output_dir, 'tmp')
print(f"TMP_DIR={TMP_DIR}")
os.makedirs(TMP_DIR, exist_ok=True)

MAX_SEED = np.iinfo(np.int32).max
# last_trigger = gr.State("")
mode=1 #1,test pano_init
debug_i2p_pipe=True
img_select=True
session_cleanup_flags = {}
def mark_session_done(session_hash):
    session_cleanup_flags[session_hash] = True
def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the input image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The preprocessed image.
    """
    # processed_image = pipeline.preprocess_image(image)
    return image



def start_session(req: gr.Request):
    print("Create temporary files")
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    session_hash = req.session_hash
    if not os.path.exists(user_dir):
        print(f"Directory no longer exists: {user_dir}")
        return
    if not session_cleanup_flags.get(session_hash, False):
        print(f"Video is not finished, skip cleanup: {user_dir}")
        return
    try:
        shutil.rmtree(user_dir, ignore_errors=True, onerror=handle_rmtree_error)    
    except Exception as e:
        print(f"Cleanup failed: {user_dir}\nError: {str(e)}")
    print("Clear temporary files")

def handle_exit_signal(signum, frame):
    print(f"Received exit signal {signum}, started cleaning...")
    unload_all_models()
    sys.exit(0)



def unload_all_models():
    global video_Pipeline, t2p_Pipeline, i2p_Pipeline

    print("Unloading model...")
    try:
        if video_Pipeline is not None:
            video_Pipeline.shutdown()
            video_Pipeline = None
    except Exception as e:
        print(f"Uninstalling video_Pipeline failed: {e}")

    try:
        if t2p_Pipeline is not None:
            t2p_Pipeline.to("cpu")
            del t2p_Pipeline
            t2p_Pipeline = None
    except Exception as e:
        print(f"Uninstalling t2p_Pipeline failed: {e}")

    try:
        if i2p_Pipeline is not None:
            i2p_Pipeline.to("cpu")
            del i2p_Pipeline
            i2p_Pipeline = None
    except Exception as e:
        print(f"Uninstalling i2p_Pipeline failed: {e}")

    try:
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Failed to release CUDA cache: {e}")

    print("Model unloading and resource cleanup completed")

def handle_rmtree_error(func, path, exc_info):
    """Custom error handling"""
    if not os.path.exists(path):
        return
    os.chmod(path, 0o700)  # 强制修改权限
    func(path)  # 重试删除

def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def reset_parameters():
    """Reset all control parameters to default values"""
    return [
        "720p",
        30,    # angle_control Defalut
        0.3,   # movement_range Defalut
        "straight"  # movement_mode Defalut
    ]

def extract_scene(video_resolution,output_buf,req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    # gs_path = os.path.join(user_dir, str(req.session_hash))
    gs_path=user_dir
    status = "Waiting for 3D model to load...",None
    yield status
    for i in range(0, 101, 20):
        yield f"Loading... {i}%",None
        time.sleep(0.2)
    def _extract_task(gs_path=None):
        if gs_path is None:
            step1_output_dir="/ai-video-sh/haoyuan.li/AIGC/matrix3d_inference/debug/debug_video_multi/debug_video_multi"
        else:
            step1_output_dir=gs_path
            print(f"gs_path={gs_path}")
        # Here is actual extraction logic
        if video_resolution=="720p":
            output_ply = Extract_Scene(
                step1_output_dir=step1_output_dir
            )
        else:
            output_ply = Extract_Scene(
                step1_output_dir=step1_output_dir,
                resulotion=480
            )
        return output_ply
    import threading
    result = []
    # gs_path="/datasets_3d/haoyuan.li/AIGC/matrix_gradio/Matrix-3D-main/output/example1"
    print(f"gs_path={gs_path}")
    thread = threading.Thread(target=lambda: result.append(_extract_task(gs_path)))
    thread.start()
    while thread.is_alive():
        yield "⏳ Extracting 3D scene... (Running) This could take a while — feel free to take a break or do something else!", None
        time.sleep(0.5)

    if not result or result[0] is None:
        yield "❌ Extraction failed (no output)", None
    else:
        yield "✅ Finished! You can now click the button below to download.", result[0]

def load_video(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    print(f"video_output={user_dir}")
    
    fixed_video_path = os.path.join(user_dir,"generated","generated.mp4")
    return fixed_video_path  

def pano_video_generation(video_resolution, angle_control, movement_range, movement_mode, req: gr.Request):
    global video_Pipeline, pano_path, pano_prompt

    if video_Pipeline is None or current_resolution!=video_resolution:
            for _ in init_video(device = torch.device("cuda:0"),video_resolution=video_resolution):
                pass

    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    pano_path = os.path.join(user_dir, 'pano_img.jpg')
    prompt_path = os.path.join(user_dir,'prompt.txt')
    print(f"prompt_path={prompt_path}")

    with open(prompt_path,"r",encoding="utf-8") as f:
        pano_prompt=f.read()
    seed = get_seed(False, 119223)
    print(f"pano_path={pano_path}")
    print(f"pano_prompt={pano_prompt}")
    print(f"angle_control={angle_control}")
    print(f"movement_range={movement_range}")
    print(f"movement_mode={movement_mode}")
    #pano_prompt="a wooden cabin on stilts, serene lake, majestic mountains, misty clouds, lush greenery, rustic charm, tranquil atmosphere, high-resolution, vibrant colors, detailed textures"
    output_path=video_Pipeline.gen_video(
                                seed=seed, 
                                panorama_path=pano_path, 
                                prompt=pano_prompt, 
                                angle=angle_control,
                                movement_range=movement_range,
                                movement_mode=movement_mode,
                                output_dir=user_dir)
    mark_session_done(req.session_hash)
    
    # output_path="/ai-video-sh/zhongqi.yang/code/zhongqi.yang/panorama_scene_generation/output_step1/save/generated/generated.mp4"
    print(f"video_output_path={output_path}")
    time.sleep(0.5)
    return output_path
def init_video(device=torch.device("cuda"),video_resolution="720p"):
    global video_Pipeline,pano_prompt,pano_path,current_resolution
    print(f"new_resolution={video_resolution},old_resolution={current_resolution}")
    # try:
    if video_Pipeline is not None and current_resolution == video_resolution:
        print(f"{video_resolution} model has been loaded, no need to reinitialize")
        return

    if video_Pipeline is not None:
        print(f"Unload the current {current_resolution} model...")
        del video_Pipeline
        video_Pipeline = None
        torch.cuda.empty_cache()  # free gpu

    if video_Pipeline is None:
        print(f"Loading {video_resolution} vdieo Model...")
        log = ""
        if video_mode=="single":
            if video_resolution=="720p":
                video_Pipeline=Video_Gen_Single(device, resolution=720)
            else:
                video_Pipeline=Video_Gen_Single(device, resolution=480)
        else:
            global max_gpus
            if video_resolution=="720p":
                video_Pipeline=Video_Gen_Multi(device,max_gpus=max_gpus,resolution=720)
            else:
                video_Pipeline=Video_Gen_Multi(device,max_gpus=max_gpus,resolution=480)
        current_resolution = video_resolution
        log += "✅ Basic model loading completed\n"
        yield log + "🎉  video Model is ready!！"

def init_t2p(device=torch.device("cuda")):
    global i2p_Pipeline, t2p_Pipeline
    print(f"t2p_Pipeline target transfer device: {device}")
    try:
        # if i2p_Pipeline is not None:
        #     del i2p_Pipeline
        #     i2p_Pipeline=None
        #     torch.cuda.empty_cache()
        if t2p_Pipeline is None:
            print("Loading the t2p model")
            t2p_Pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            ).to(device) 
            
            print(f"lora path={os.path.abspath(lora_path)}")
            t2p_Pipeline.load_lora_weights(lora_path)
            match = re.search(r"cuda:(\d+)", str(device))
            gpu_id = int(match.group(1)) if match else 0 
            t2p_Pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
            t2p_Pipeline.enable_vae_tiling()
            device = t2p_Pipeline.vae.device
            if t2p_Pipeline is not None:
                print(f"t2p_Pipeline Device: {device}", flush=True)
                log = ""
                yield log + "🎉 t2p Model is ready!"
    except Exception as e:
        yield f"❌ t2p Initialization failed: {str(e)}"
        print(f"❌ t2p Initialization failed: {str(e)}")
        t2p_Pipeline = None
    
def init_i2p(device=torch.device("cuda")):
    global i2p_Pipeline, t2p_Pipeline
    try:
        # if i2p_Pipeline is not None:
        #     del i2p_Pipeline
        #     i2p_Pipeline=None
        #     torch.cuda.empty_cache()
        if i2p_Pipeline is None:
            print("Loading the i2p model")
            log = ""
            yield log + "⏳ Load base model..."
            i2p_Pipeline = i2pano(device)
            if i2p_Pipeline is not None:
                print(f"i2p_Pipeline Device: {device}", flush=True)
                yield log + "🎉 i2p Model is ready!"
    except Exception as e:
        yield f"❌ i2p Initialization failed: {str(e)}"
        print(f"❌ i2p Initialization failed: {str(e)}")
        i2p_Pipeline = None


def pano_image_generation_img(image_prompt, req: gr.Request):
    global i2p_Pipeline
    prompt=None
    if image_prompt:
        if mode==1:  
                    os.makedirs("uploads", exist_ok=True)
                    from datetime import datetime
                    
                    os.makedirs("panorama_scene_generation/output_step1/save/", exist_ok=True)
                    save_upload_path = f"panorama_scene_generation/output_step1/save/image_upload.png"
                    image_prompt.save(save_upload_path)
                    seed = get_seed(False, 119223)
                    if debug_i2p_pipe:
                        torch.cuda.empty_cache()
                        pano_image, prompt =i2p_Pipeline.inpaint_img(save_upload_path,seed)
                    print(f"i2p prompt is {prompt}")
                
        else: 
            print("Execute i2p")
            pano_image = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
            pano_image = Image.fromarray(pano_image)
    else:
        print("No prompt input")
        pano_image = None


    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    pano_path = os.path.join(user_dir, 'pano_img.jpg')
    pano_image.save(pano_path)

    prompt_path = os.path.join(user_dir,"prompt.txt")
    if prompt is not None:
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
    torch.cuda.empty_cache()
    print(f"pano_prompt is {prompt}")
    return pano_image

def pano_image_generation(text_prompt, req: gr.Request):
    """
    Generate a panoramic image based on the text or image prompt.
    """
    global t2p_Pipeline
    print(text_prompt)
    prompt=None
    # Placeholder for actual image generation logic
    if text_prompt:
        # Simulate image generation from text
            
        # init_t2p()
        # for _ in init_t2p(): 
        #     pass
        if mode==1:  
            if t2p_Pipeline is None:
                raise gr.Error("Model Initialization Failed")
            seed = get_seed(False, 119223)
            torch.cuda.empty_cache()
            pano_image = t2p_Pipeline(text_prompt, 
                height=512,
                width=1024,
                generator=torch.Generator("cpu").manual_seed(seed),
                num_inference_steps=50, 
                blend_extend=0,
                guidance_scale=7).images[0]   
        else:
            print("Execute t2p")
            pano_image = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
            pano_image = Image.fromarray(pano_image)
        prompt=text_prompt
    else:
        pano_image = None

    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    pano_path = os.path.join(user_dir, 'pano_img.jpg')
    pano_image.save(pano_path)

    prompt_path = os.path.join(user_dir,"prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    torch.cuda.empty_cache()
    print(f"pano_prompt is {prompt}")
    return pano_image






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Video Generation Pipeline')
    parser.add_argument('--max_gpus', type=int, default=2, 
                       help='最大GPU数量')
    args = parser.parse_args()
    max_gpus = args.max_gpus
    if max_gpus>1:
        video_mode="multi"
    else:
        video_mode="single"

    print("\n=== Starting Initialize Pipeline ===")
    print("Initialize t2p Pipeline")
    if t2p_Pipeline is None:
        for _ in init_t2p(device = torch.device("cuda:0")):  # 执行所有初始化步骤
            pass
    if video_Pipeline is None:
            for _ in init_video(device = torch.device("cuda:0")):
                pass
    if args.max_gpus>1:
        print("Initialize i2p Pipeline")
        if i2p_Pipeline is None:
            for _ in init_i2p(device = torch.device("cuda:1")):  # 执行所有初始化步骤
                pass


    with gr.Blocks(title="Matrix-3D App", delete_cache=(600, 600)) as demo:
        gr.Markdown("""
        ## Text/image to 3D Scene with Matrix3D
        """)
        
        with gr.Row():
            with gr.Column():
                tab_selector = gr.Radio(["text", "image"], value="text", visible=False) 
                with gr.Tabs() as input_tabs:
                    with gr.Tab(label="Text Prompt", id=0) as text_input_tab:
                        text_prompt = gr.Textbox(label="Text Prompt", info="Text describing a 3D scene, e.g., A small anime village with thatched-roof houses.", lines=3, max_lines=10)
                        # text_prompt.change(lambda _: gr.update(value="text"), None, tab_selector)
                        text_prompt.change(lambda _: "text", inputs=text_prompt, outputs=tab_selector)
                        generate_image_btn = gr.Button("STEP1: Generate Pano Image")
                    if max_gpus > 1:
                        print(f"有{max_gpus}用来初始化")
                        if img_select:
                            with gr.Tab(label="Image Prompt", id=1) as image_input_tab:
                                image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGB", type="pil", height=300)
                                # image_prompt.change(lambda _: gr.update(value="image"), None, tab_selector)
                                image_prompt.change(lambda _: "image", inputs=image_prompt, outputs=tab_selector)
                                # Example images at the bottom of the page
                                with gr.Row():
                                    with gr.Column():
                                        loading_text = gr.Markdown("""
                                    **Image Input Options**   You can either:  1️⃣ **Upload** your own image   2️⃣ **Select** from the examples below  """,elem_classes="requirement-text")
                                        with gr.Row() as single_image_example:
                                            print(f"show_datapath={os.path.abspath('./data')}")
                                            examples = gr.Examples(
                                                examples=[
                                                    f'data/{image}'
                                                    for image in os.listdir("./data")
                                                    if os.path.isfile(os.path.join("./data", image))
                                                ],
                                                inputs=[image_prompt],
                                                fn=preprocess_image,
                                                outputs=[image_prompt],
                                                run_on_click=True,
                                                examples_per_page=5,
                                                label="Sample Images"

                                            )
                                generate_image_btn_img = gr.Button("STEP1: Generate Pano Image")
                    else:
                        image_prompt = gr.State("none")

                pano_image = gr.Image(label="Generated Pano Image", format="png", image_mode="RGB", type="pil", height=300)
                
                generate_video_btn = gr.Button("STEP2: Generate Pano Video",interactive=False)
                    # Record trigger source status
                state = gr.State("none")
                print(text_prompt)
        

                with gr.Accordion("⚙️ Panorama Video Generation Setting", open=False): 
                    with gr.Row():
                        # Resolution selection (drop-down menu)
                        video_resolution = gr.Dropdown(
                            choices=["720p", "480p"],
                            value="720p",
                            label="Video Resolution",
                            interactive=True
                        )
                    with gr.Row():
                        # Angle control
                        angle_control = gr.Slider(
                            minimum=0, maximum=360, value=30, step=1,
                            label="Angle Control", interactive=True
                        )
                    
                        # range of motion control
                        movement_range = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.3, step=0.05,
                            label="Movement Range", interactive=True
                        )
                    with gr.Row():
                        # movement select
                        movement_mode = gr.Dropdown(
                            choices=["straight", "s_curve","r_curve", "l_curve"],
                            value="straight",
                            label="Movement Mode"
                        )
                            
                # Reset
                reset_btn = gr.Button("Reset",interactive=False)
            
            with gr.Column():

                video_output = gr.Video(label="Generated Pano Video", format="mp4",autoplay=True, loop=True, include_audio=False,height=300)
                reload_btn = gr.Button("Reload Video",interactive=False)
                reload_btn.click(load_video, outputs=video_output)

                generate_scene_btn = gr.Button("STEP3: Extract Scene",interactive=False)
                loading_text = gr.Markdown("")

                # scene_output = LitModel3D(label="extracted 3D Scene", exposure=10.0, height=300)
                output_buf = gr.State()
                download_gs = gr.DownloadButton(label="STEP4: Download 3D Scene", interactive=False, elem_classes="light-button")
                gr.Markdown("""
                ### When the download is done, just head over to the [viewer](https://sparkjs.dev/viewer/), upload your result, and you’ll see the 3D reconstruction.
                """)
        demo.load(start_session)
        demo.unload(end_session)

        def clear_inputs():
            return "", None  # clear text and image
        generate_image_btn.click(
            fn=lambda: [gr.Button(interactive=False),gr.Button(interactive=False),gr.Button(interactive=False)],
            outputs=[generate_video_btn,  generate_scene_btn, download_gs],
            queue=False
        ).then(
            fn=pano_image_generation,
            inputs=[text_prompt],
            outputs=pano_image,
            concurrency_limit=1,
            queue=True 
        ).then(
            fn=lambda: gr.update(interactive=True),
            outputs=[generate_video_btn],
            queue=False
        )


        if max_gpus>1:
            generate_image_btn_img.click(
                fn=lambda: [gr.Button(interactive=False),gr.Button(interactive=False),gr.Button(interactive=False)],
                outputs=[generate_video_btn,  generate_scene_btn, download_gs],
                queue=False
            ).then(
                fn=pano_image_generation_img,
                inputs=[image_prompt],
                outputs=pano_image,
                concurrency_limit=1,
                queue=True 
            ).then(
                fn=lambda: gr.update(interactive=True),
                outputs=[generate_video_btn],
                queue=False
            )

        generate_video_btn.click(
            fn=pano_video_generation,
            inputs=[video_resolution,angle_control, movement_range, movement_mode],
            outputs=video_output,
            concurrency_limit=1,
            queue=True 
        ).then(
            fn=lambda: gr.update(interactive=True),
            outputs=[generate_scene_btn],
            queue=False
        ).then(
            fn=lambda: gr.update(interactive=True),
            outputs=[reset_btn],
            queue=False
        ).then(
            fn=lambda: gr.update(interactive=True),
            outputs=[reload_btn],
            queue=False
        )

        reset_btn.click(
            fn=reset_parameters,
            outputs=[video_resolution, angle_control, movement_range, movement_mode],
            concurrency_limit=1
        )

        generate_scene_btn.click(
            fn=lambda: gr.Button(interactive=False),
            outputs=[download_gs],
            queue=False
        ).then(
            fn=extract_scene,
            inputs=[video_resolution, output_buf],
            outputs=[loading_text, download_gs],
            concurrency_limit=1
        ).then(
            fn=lambda: gr.Button(interactive=True, elem_classes="dark-button"),
            outputs=[download_gs],
            queue=False
        )


    print("\n=== Starting Gradio Service ===")
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)
    demo.queue(max_size=4)
    demo.launch(share=True)
    # if mode==1:
    #     print("初始化 t2p")
    #     if t2p_Pipeline is None:
    #         for _ in init_t2p(device = torch.device("cuda:0")):  # 执行所有初始化步骤
    #             pass
    #     # print("初始化 i2p")
    #     # if i2p_Pipeline is None:
    #     #     for _ in init_i2p(device = torch.device("cuda:1")):  # 执行所有初始化步骤
    #     #         pass
    #     if video_Pipeline is None:
    #         for _ in init_video(device = torch.device("cuda:0")):
    #             pass
    
    # # #?用来测试
    # # # i2p_Pipeline = i2pano(device = torch.device("cuda:1"))
    # # # # Extract_Scene()
    # # # # video_Pipeline.gen_video()
    # # # model=Video_Gen(device = torch.device("cuda:3"))
    # # # model.gen_video()
    # demo.queue(max_size=4)
    # demo.launch(share=True)
