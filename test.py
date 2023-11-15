import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from opr.testing import test
from opr.utils import parse_device


def main() -> None:
    """Test a model on a dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint file")
    parser.add_argument("--model_config", type=Path, required=True, help="Path to model configuration file")
    parser.add_argument(
        "--dataset_config", type=Path, required=True, help="Path to dataset configuration file"
    )
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint)
    model_cfg = OmegaConf.load(args.model_config)
    dataset_cfg = OmegaConf.load(args.dataset_config)
    dataset_cfg.dataset_root = args.dataset_dir

    model = instantiate(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(parse_device(args.device))
    model.eval()

    dataset = instantiate(dataset_cfg, subset="test")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

    mean_recall_at_n, mean_recall_at_one_percent, mean_top1_distance = test(model, dataloader)

    print(f"Mean AverageRecall@N:\n{mean_recall_at_n}")
    print(f"Mean AverageRecall@1%: {mean_recall_at_one_percent}")
    print(f"Mean Top-1 Distance: {mean_top1_distance}")


if __name__ == "__main__":
    main()
