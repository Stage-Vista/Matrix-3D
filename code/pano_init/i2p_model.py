import math
import sys
from pathlib import Path

import cv2
import torch
from pano_init.prompt.prompt import Lamma_Video
from pano_init.src.worldgen import WorldGen
from torch import device

DA3_SRC_DIR = Path(__file__).absolute().parents[1] / "DA3" / "src"
if str(DA3_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DA3_SRC_DIR))

from depth_anything_3.api import DepthAnything3


class i2pano:
    def __init__(self, device):

        self.device = device
        self.da3_small = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
        self.da3_small = self.da3_small.to(self.device)
        self.worldgen = WorldGen(mode="i2s", device=self.device)
        self.Lamma_Video = Lamma_Video(self.device)

    def inpaint_img(self, img_path, seed=42, prompt=None, fov=None):
        if fov is None:
            hFov, wFov = self.calculate_FOV(img_path)
        else:
            hFov = fov
            wFov = fov
        try:
            if prompt is not None:
                prompt = prompt
            else:
                prompt = self.Lamma_Video.extract_prompt(img_path, debug=True)
        except:
            print("Lamma Video prompt failed")
            promt = "a lot of trees"
        print(f"Lamma Video prompt {prompt}")
        prompt_copy = prompt
        pano_img = self.worldgen.generate(img_path, prompt,
                                          wFov, hFov, seed)
        return pano_img, prompt_copy

    def calculate_FOV(self, img_path, debug=True):
        input_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if input_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)

        prediction = self.da3_small.inference([input_rgb])
        K = prediction.intrinsics[0]

        def calculateFov(focal, size):
            return 2 * math.degrees(math.atan(size / (2 * focal)))

        fx, fy, cx, cy = K[0, 0].item(), K[1, 1].item(), K[0, 2].item(), K[1, 2].item()
        img_h = 2 * cy
        img_w = 2 * cx
        hFov = calculateFov(fy, img_h)
        wFov = calculateFov(fx, img_w)
        if debug:
            print(f"hFov={hFov}, wFov={wFov}")
        return hFov, wFov


if __name__ == "__main__":
    device = torch.device("cuda")
    model = i2pano(device)
    img_path = "/ai-video-sh/haoyuan.li/AIGC/Panodiff/datasets/ours/split_mp4/pers/Rail_00_22_Take_051_rgb.jpg"
    pano_img = model.inpaint_img(img_path)
    pano_img.save("/ai-video-sh/haoyuan.li/AIGC/WorldGen/debug_img/test_pano.jpg")
