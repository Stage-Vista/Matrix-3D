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
) -> tuple[np.ndarray, np.ndarray]:
    """Run DA3 on a single RGB image path and return (depth, confidence).

    DepthAnything3.inference expects a list of images; we pass a single numpy RGB image.
    """
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    prediction = model.inference([rgb])
    # prediction.depth, prediction.conf : (N, H, W). We take the first element of each.
    depth = prediction.depth[0].astype(np.float32)
    conf = prediction.conf[0].astype(np.float32)

    # Sanitize depth: remove NaNs/Infs and ensure we have some positive values
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    finite = np.isfinite(depth)
    if not np.any(finite):
        raise RuntimeError(f"DA3 produced no finite depth values for image: {image_path}")

    # Ensure depth is positive somewhere; if not, shift into a positive range.
    positive = finite & (depth > 0)
    if not np.any(positive):
        finite_vals = depth[finite]
        min_val = float(finite_vals.min())
        depth = depth - min_val + 1e-3

    return depth, conf


def save_outputs(
        image_path: Path,
        output_root: Path,
        depth: np.ndarray,
        conf: np.ndarray | None = None,
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

        # Binary mask: derive from DA3 confidence if available, otherwise from depth.
        if conf is not None:
            # Normalize confidence to [0,1] and threshold.
            conf_valid = np.isfinite(conf)
            if np.any(conf_valid):
                c = conf.copy()
                c[~conf_valid] = 0.0
                c_min = float(c[conf_valid].min())
                c_max = float(c[conf_valid].max())
                if c_max > c_min:
                    c_norm = (c - c_min) / (c_max - c_min)
                else:
                    c_norm = np.zeros_like(c)
                # Primary threshold on confidence.
                valid_mask = c_norm > 0.2
                if not np.any(valid_mask):
                    # If too strict, accept any positive confidence.
                    valid_mask = c_norm > 0.0
            else:
                valid_mask = np.ones_like(depth, dtype=bool)
        else:
            valid_mask = np.isfinite(depth) & (depth > 0)
            if not np.any(valid_mask):
                valid_mask = np.isfinite(depth)

        if not np.any(valid_mask):
            raise RuntimeError(f"DA3 produced empty validity mask for image: {image_path}")

        mask = valid_mask.astype(np.uint8) * 255
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

    # Load DA3 model (can be metric or any-view; configured via args)
    model = load_da3_model(args.pretrained_name_or_path, device)

    for img_path in image_paths:
        depth, conf = run_da3_on_image(model, img_path)

        # If resize_to is specified, resize depth accordingly
        if args.resize_to is not None:
            h, w = depth.shape[:2]
            new_h = min(args.resize_to, int(args.resize_to * h / w))
            new_w = min(args.resize_to, int(args.resize_to * w / h))
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        save_outputs(img_path, output_path, depth, conf, save_maps=args.save_maps or True)


if __name__ == "__main__":
    main()
