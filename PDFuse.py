import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

from utils.degration import build_degradation_pipeline, strip_exposure_operators
from utils.utils import rgb01_to_ycbcr01, ycbcr01_to_rgb01

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "IVF"
DEFAULT_DPS_SKIP_STEP = 2
DEFAULT_LIGHT_ITERATIONS = 10
DEFAULT_LIGHT_STRATEGY = 2
DEFAULT_CG_DAMPING = 1e-4


def transfer_chroma_if_grayscale(img1, img2):
    img1_is_gray = (
        torch.equal(img1[:, 0, :, :], img1[:, 1, :, :])
        and torch.equal(img1[:, 1, :, :], img1[:, 2, :, :])
    )
    img2_is_gray = (
        torch.equal(img2[:, 0, :, :], img2[:, 1, :, :])
        and torch.equal(img2[:, 1, :, :], img2[:, 2, :, :])
    )

    if img1_is_gray and not img2_is_gray:
        img1_ycbcr = rgb01_to_ycbcr01(img1)
        img2_ycbcr = rgb01_to_ycbcr01(img2)
        img1_ycbcr[:, 1:3, :, :] = img2_ycbcr[:, 1:3, :, :]
        img1 = ycbcr01_to_rgb01(img1_ycbcr)
    elif img2_is_gray and not img1_is_gray:
        img1_ycbcr = rgb01_to_ycbcr01(img1)
        img2_ycbcr = rgb01_to_ycbcr01(img2)
        img2_ycbcr[:, 1:3, :, :] = img1_ycbcr[:, 1:3, :, :]
        img2 = ycbcr01_to_rgb01(img2_ycbcr)

    return img1, img2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image_as_tensor(path, device, pad_size=16):
    image = Image.open(path).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    if pad_size > 0:
        tensor = F.pad(tensor, pad=(pad_size, pad_size, pad_size, pad_size), mode="reflect")

    return tensor.to(device)


def pad_tensor(tensor, pad_size):
    if pad_size <= 0:
        return tensor
    return F.pad(tensor, pad=(pad_size, pad_size, pad_size, pad_size), mode="reflect")


def build_observed_image(image, degradation_spec):
    pipeline = strip_exposure_operators(build_degradation_pipeline(degradation_spec))
    return pipeline(image).clamp(0, 1)


def iter_paired_images(input1_dir, input2_dir, extensions):
    input1_dir = Path(input1_dir)
    input2_dir = Path(input2_dir)

    for path1 in sorted(input1_dir.iterdir()):
        if not path1.is_file() or path1.suffix.lower() not in extensions:
            continue

        path2 = input2_dir / path1.name
        if not path2.exists():
            print(f"[skip] Missing paired image: {path2}")
            continue

        yield path1, path2


def has_images(directory, extensions):
    return any(path.is_file() and path.suffix.lower() in extensions for path in directory.iterdir())


def resolve_input_dirs(data_dir, extensions):
    data_dir = Path(data_dir).expanduser()
    if not data_dir.exists():
        fallback_root = SCRIPT_DIR / "data"
        available = []
        if fallback_root.is_dir():
            available = [str(path) for path in sorted(fallback_root.iterdir()) if path.is_dir()]
        hint = f" Available datasets: {', '.join(available)}." if available else ""
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}.{hint}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_dir}")

    image_dirs = [
        path for path in sorted(data_dir.iterdir())
        if path.is_dir() and has_images(path, extensions)
    ]
    if len(image_dirs) != 2:
        names = ", ".join(path.name for path in image_dirs) or "none"
        raise ValueError(
            f"Expected exactly two image subdirectories under {data_dir}, found: {names}. "
            "Pass --image1 and --image2 for one-pair inference, or point --data_dir to a directory "
            "containing exactly two image subdirectories."
        )

    return image_dirs[0], image_dirs[1]


def build_solver(args, device):
    from latent_diffusion import get_solver

    return get_solver(
        name="PDFuse",
        device=device,
        model_key=args.model_id,
        solver_config={"num_sampling": args.nfe},
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run PDFuse image fusion.")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("DATA_DIR", str(DEFAULT_DATA_DIR)))
    parser.add_argument("--image1", type=str, default=None, help="First source image for one-pair inference.")
    parser.add_argument("--image2", type=str, default=None, help="Second source image for one-pair inference.")
    parser.add_argument("--output", type=str, default=os.environ.get("RESULT", "./result/"))
    parser.add_argument("--model_id", type=str, default="./pretrained/stable-diffusion-v1-5/")
    parser.add_argument("--null_prompt", type=str, default="noisy, low quality")
    parser.add_argument("--prompt", type=str, default="noise-free, high quality")
    parser.add_argument("--nfe", type=int, default=50, help="Number of function evaluations / sampling steps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--pad_size", type=int, default=16)
    parser.add_argument(
        "--degradation1",
        type=str,
        default="none",
        help="Degradation pipeline for source A, e.g. none, lowlight, overexposure, gaussian, motion.",
    )
    parser.add_argument(
        "--degradation2",
        type=str,
        default="none",
        help="Degradation pipeline for source B, e.g. none, gaussian, motion, thermal_diffusion.",
    )
    parser.add_argument(
        "--fusion_task",
        type=str,
        default="IVF",
        choices=["IVF", "MMIF", "MFIF", "MEIF"],
        help="Pixel target task setting for infrared-visible, multi-focus, or multi-exposure fusion.",
    )
    parser.add_argument("--extensions", type=str, default=",".join(ext.strip(".") for ext in IMAGE_EXTENSIONS))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cpu" if args.cpu else "cuda:0"
    output_dir = Path(args.output)
    if output_dir.suffix.lower() in IMAGE_EXTENSIONS:
        raise ValueError(
            f"--output must be a directory, not an image file: {output_dir}. "
            "Use --image2 for the second source image."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    observed1_dir = output_dir / "observed1"
    observed2_dir = output_dir / "observed2"
    observed1_dir.mkdir(parents=True, exist_ok=True)
    observed2_dir.mkdir(parents=True, exist_ok=True)

    extensions = tuple(f".{ext.strip().lower().lstrip('.')}" for ext in args.extensions.split(",") if ext.strip())
    if args.image1 and args.image2:
        image_pairs = [(Path(args.image1), Path(args.image2))]
        print(f"Input A: {args.image1}")
        print(f"Input B: {args.image2}")
    else:
        input1_dir, input2_dir = resolve_input_dirs(args.data_dir, extensions)
        print(f"Input A: {input1_dir}")
        print(f"Input B: {input2_dir}")
        image_pairs = list(iter_paired_images(input1_dir, input2_dir, extensions))

    degradation1 = args.degradation1
    degradation2 = args.degradation2
    print(f"Degradation A: {degradation1}")
    print(f"Degradation B: {degradation2}")

    solver = build_solver(args, device)

    num_processed = 0
    for path1, path2 in image_pairs:
        img1 = load_image_as_tensor(path1, device=device, pad_size=args.pad_size)
        img2 = load_image_as_tensor(path2, device=device, pad_size=args.pad_size)

        observed1 = build_observed_image(img1, degradation1)
        observed2 = build_observed_image(img2, degradation2)
        save_image(observed1[:,:,args.pad_size:-args.pad_size,args.pad_size:-args.pad_size].float().cpu().clamp(0, 1), observed1_dir / path1.name)
        save_image(observed2[:,:,args.pad_size:-args.pad_size,args.pad_size:-args.pad_size].float().cpu().clamp(0, 1), observed2_dir / path1.name)

        img1, img2 = transfer_chroma_if_grayscale(observed1, observed2)

        sample_output = solver.sample(
            measurement=img1,
            measurement1=img2,
            prompt=[args.null_prompt, [args.prompt]],
            fusion_task=args.fusion_task,
            degradation1=degradation1,
            degradation2=degradation2,
            light_iterations=DEFAULT_LIGHT_ITERATIONS,
            light_strategy=DEFAULT_LIGHT_STRATEGY,
            cg_damping=DEFAULT_CG_DAMPING,
            inputs_are_observed=True,
            return_observations=True,
            no_dps=lambda step: step % DEFAULT_DPS_SKIP_STEP == 0 and step <= args.nfe,
        )
        recon = sample_output["recon"]

        if args.pad_size > 0:
            recon = recon[
                :,
                :,
                args.pad_size : recon.shape[2] - args.pad_size,
                args.pad_size : recon.shape[3] - args.pad_size,
            ]

        save_image(recon.float().clamp(0, 1), output_dir / path1.name)
        num_processed += 1
        print(f"[done] {path1.name}")

    print(f"Processed {num_processed} image pair(s).")


if __name__ == "__main__":
    main()
