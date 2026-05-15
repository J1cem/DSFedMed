"""Server-side synthetic image-mask production from uploaded ControlNet branches."""

from __future__ import annotations

import argparse
import json
import os
import random
from glob import glob
from typing import Dict, List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.controlnet import ControlNetConfig, ControlNetGenerator, average_control_states, normalize_mask


def parse_client_names(value: str) -> List[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def load_mask(path: str, image_size: int) -> np.ndarray:
    mask = np.load(path)
    if mask.ndim == 3:
        mask = mask.mean(axis=-1)
    if mask.shape[0] != image_size or mask.shape[1] != image_size:
        mask = cv2.resize(mask.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    max_value = max(float(mask.max()), 1.0)
    return (mask / max_value).astype(np.float32)[None]


def random_structural_mask(image_size: int) -> np.ndarray:
    mask = np.zeros((image_size, image_size), dtype=np.float32)
    for _ in range(random.randint(1, 5)):
        center = (random.randint(0, image_size - 1), random.randint(0, image_size - 1))
        axes = (random.randint(image_size // 16, image_size // 4), random.randint(image_size // 16, image_size // 4))
        angle = random.randint(0, 180)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask[None]


def discover_label_paths(mask_root: str, client_names: List[str]) -> Dict[str, List[str]]:
    return {name: sorted(glob(os.path.join(mask_root, name, "label_npy", "*.npy"))) for name in client_names}


def main():
    parser = argparse.ArgumentParser(description="Generate DSFedMed synthetic data with uploaded ControlNet branches")
    parser.add_argument("--controlnet_dir", required=True, help="Directory produced by train_controlnet_clients.py")
    parser.add_argument("--output_dir", required=True, help="Synthetic dataset root to write")
    parser.add_argument("--client_names", required=True, help="Comma separated output client names")
    parser.add_argument("--mask_root", default=None, help="Optional private/seed dataset root used only to sample structural label_npy masks")
    parser.add_argument("--samples_per_client", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--aggregate", action="store_true", help="Use the averaged ControlNet branch for every client instead of each branch separately")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.controlnet_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    config = ControlNetConfig(**manifest["config"])
    states = []
    for client in manifest["clients"]:
        ckpt = torch.load(os.path.join(args.controlnet_dir, client["checkpoint"]), map_location="cpu")
        states.append(ckpt["control_state_dict"])

    client_names = parse_client_names(args.client_names)
    label_paths = discover_label_paths(args.mask_root, client_names) if args.mask_root else {name: [] for name in client_names}
    averaged_state = average_control_states(states) if args.aggregate else None

    for client_idx, client_name in enumerate(client_names):
        model = ControlNetGenerator(config).to(device).eval()
        state = averaged_state if averaged_state is not None else states[min(client_idx, len(states) - 1)]
        model.load_control_state_dict(state)
        data_dir = os.path.join(args.output_dir, client_name, "data_npy")
        label_dir = os.path.join(args.output_dir, client_name, "label_npy")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        paths = label_paths.get(client_name, [])
        for sample_idx in tqdm(range(args.samples_per_client), desc=f"generate={client_name}"):
            if paths:
                mask_np = load_mask(random.choice(paths), args.image_size)
            else:
                mask_np = random_structural_mask(args.image_size)
            mask = normalize_mask(torch.from_numpy(mask_np).unsqueeze(0).to(device))
            with torch.no_grad():
                image = model(mask).squeeze(0).permute(1, 2, 0).cpu().numpy()
            image_u8 = (image.clip(0, 1) * 255).astype(np.uint8)
            label = (mask_np.transpose(1, 2, 0) > 0.5).astype(np.float32)
            filename = f"syn_{sample_idx:05d}.npy"
            np.save(os.path.join(data_dir, filename), image_u8)
            np.save(os.path.join(label_dir, filename), label)


if __name__ == "__main__":
    main()
