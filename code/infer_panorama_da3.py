import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

# Ensure DA3 src is on the path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from depth_anything_3.api import DepthAnything3


def load_da3_model(model_name: str, device: str) -> DepthAnything3:
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(torch.device(device))
    return model


def run_da3_on_image(
        model: DepthAnything3,
        image_path: Path,
) -> np.ndarray:
    """Run DA3METRIC (or other DA3 model) on a single RGB image path and return depth.

    DepthAnything3.inference expects a list of images; we pass a single numpy RGB image.
    """
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    prediction = model.inference([rgb])
    # prediction.depth: (N, H, W). We take the first element.
    depth = prediction.depth[0]
    return depth.astype(np.float32)


def save_outputs(
        image_path: Path,
        output_root: Path,
        depth: np.ndarray,
        save_maps: bool = True,
        subdir_name: str = "da3",
) -> None:
    """Save depth and a simple mask under a DA3-specific subdirectory.

    Given:
      --input <.../foo.png>
      --output <OUT>
    we write into: <OUT>/<subdir_name>/{image,depth,mask}.*
    """
    save_dir = output_root / subdir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if save_maps:
        # Save original image (for consistency with previous convention)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is not None:
            cv2.imwrite(str(save_dir / "image.jpg"), bgr)

        # Depth EXR
        cv2.imwrite(
            str(save_dir / "depth.exr"),
            depth,
            [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT],
        )

        # Simple visualization
        depth_vis = depth.copy()
        finite_mask = np.isfinite(depth_vis)
        if finite_mask.any():
            dmin = depth_vis[finite_mask].min()
            dmax = depth_vis[finite_mask].max()
            if dmax > dmin:
                depth_norm = (depth_vis - dmin) / (dmax - dmin)
            else:
                depth_norm = np.zeros_like(depth_vis)
        else:
            depth_norm = np.zeros_like(depth_vis)
        depth_color = (cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
        cv2.imwrite(str(save_dir / "depth_vis.png"), depth_color)

        # Binary mask: valid where depth is finite and > 0
        mask = (np.isfinite(depth) & (depth > 0)).astype(np.uint8) * 255
        cv2.imwrite(str(save_dir / "mask.png"), mask)


def collect_image_paths(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
        paths: List[Path] = []
        for ext in exts:
            paths.extend(sorted(input_path.rglob(f"*{ext}")))
        return paths
    else:
        return [input_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DA3-based panorama depth inference (MoGe-compatible I/O)",
    )
    parser.add_argument("--input", dest="input_path", type=str, required=True)
    parser.add_argument("--output", dest="output_path", type=str, required=True)
    parser.add_argument("--device", dest="device_name", type=str, default="cuda")
    parser.add_argument("--pretrained", dest="pretrained_name_or_path", type=str,
                        default="depth-anything/DA3METRIC-LARGE")

    # Compatibility flags (accepted but currently unused)
    parser.add_argument("--resize", dest="resize_to", type=int, default=None)
    parser.add_argument("--resolution_level", type=int, default=9)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--splitted", dest="save_splitted", action="store_true")
    parser.add_argument("--maps", dest="save_maps", action="store_true")
    parser.add_argument("--glb", dest="save_glb", action="store_true")
    parser.add_argument("--ply", dest="save_ply", action="store_true")
    parser.add_argument("--show", dest="show", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device_name
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No input images found in {input_path}")

    # Load DA3 metric model
    model = load_da3_model(args.pretrained_name_or_path, device)

    for img_path in image_paths:
        depth = run_da3_on_image(model, img_path)

        # If resize_to is specified, resize depth accordingly
        if args.resize_to is not None:
            h, w = depth.shape[:2]
            new_h = min(args.resize_to, int(args.resize_to * h / w))
            new_w = min(args.resize_to, int(args.resize_to * w / h))
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        save_outputs(img_path, output_path, depth, save_maps=args.save_maps or True)


if __name__ == "__main__":
    main()
