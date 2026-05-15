"""Client-side DSFedMed ControlNet personalization training.

Each client trains only `ControlNetGenerator.control` on private image-mask pairs
and exports the lightweight branch parameters. Raw images never leave the client.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.federated_dataloader import Dataset, ProstateDataset, ToTensor
from models.controlnet import ControlNetConfig, ControlNetGenerator, normalize_mask, reconstruction_loss


def parse_client_names(value: str) -> List[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def build_dataset(args, client_idx: int, client_names: List[str]):
    dataset_cls = ProstateDataset if "prostate" in args.data.lower() else Dataset
    return dataset_cls(
        client_idx=client_idx,
        data_path=args.data_path,
        freq_site_idx=client_idx,
        split="train",
        transform=transforms.Compose([ToTensor()]),
        client_name=client_names,
    )


def main():
    parser = argparse.ArgumentParser(description="Train per-client ControlNet branches for DSFedMed")
    parser.add_argument("--data_path", required=True, help="Root dataset path containing Client/data_npy and label_npy folders")
    parser.add_argument("--client_names", required=True, help="Comma separated client folder names")
    parser.add_argument("--data", default="generic", help="Dataset name; use a value containing 'prostate' for ProstateDataset")
    parser.add_argument("--output_dir", required=True, help="Directory where uploaded ControlNet branches are saved")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--noise_channels", type=int, default=8)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument(
        "--controlnet_tuning_mode",
        choices=["lora", "full"],
        default="lora",
        help="Use memory-efficient LoRA adapters or full ControlNet branch fine-tuning",
    )
    parser.add_argument("--controlnet_lora_rank", type=int, default=4, help="LoRA rank used when --controlnet_tuning_mode=lora")
    parser.add_argument("--controlnet_lora_alpha", type=float, default=1.0, help="LoRA scale alpha used when --controlnet_tuning_mode=lora")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    client_names = parse_client_names(args.client_names)
    config = ControlNetConfig(
        hidden_channels=args.hidden_channels,
        noise_channels=args.noise_channels,
        num_blocks=args.num_blocks,
        tuning_mode=args.controlnet_tuning_mode,
        lora_rank=args.controlnet_lora_rank,
        lora_alpha=args.controlnet_lora_alpha,
    )

    manifest = {"config": config.__dict__, "clients": []}
    for client_idx, client_name in enumerate(client_names):
        dataset = build_dataset(args, client_idx, client_names)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        # Keep the frozen ControlNet backbone identical across clients; in LoRA
        # mode only the low-rank adapters become client-specific.
        torch.manual_seed(args.seed)
        model = ControlNetGenerator(config).to(device)
        model.base.eval()
        model.control.train()
        total_params, trainable_params = model.control_parameter_counts()
        print(
            f"client={client_name} ControlNet tuning={config.tuning_mode} "
            f"trainable={trainable_params:,}/{total_params:,} params"
        )
        optimizer = torch.optim.AdamW(model.control_trainable_parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            running = 0.0
            for batch in tqdm(loader, desc=f"client={client_name} epoch={epoch}", leave=False):
                image = batch["image"][:, :3].to(device).float().clamp(0, 1)
                mask = normalize_mask(batch["label"].to(device))
                pred = model(mask)
                loss, parts = reconstruction_loss(pred, image, mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item()
            print(f"client={client_name} epoch={epoch} loss={running / max(len(loader), 1):.6f}")

        ckpt_path = os.path.join(args.output_dir, f"controlnet_client_{client_idx}_{client_name}.pth")
        torch.save({"client_idx": client_idx, "client_name": client_name, "config": config.__dict__, "control_state_dict": model.control_state_dict()}, ckpt_path)
        manifest["clients"].append({"client_idx": client_idx, "client_name": client_name, "checkpoint": os.path.basename(ckpt_path), "num_samples": len(dataset)})

    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
