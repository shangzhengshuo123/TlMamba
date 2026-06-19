from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.TSKD_train import DATASET_DIRS, create_model, dataset_name, model_name, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a released TlMamba checkpoint.")
    parser.add_argument(
        "--dataset",
        type=dataset_name,
        default="hldlc",
        help="Dataset preset. Uses the same names as models/TSKD_train.py.",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--test-dir", default="test")
    parser.add_argument(
        "--model",
        type=model_name,
        default="TlMamba",
        help="Model name: tlmamba, mambavision, vision_mamba, pure_mamba, fastvit, resnet18, or crt.",
    )
    parser.add_argument("--method", default="tlmamba_full", help="Method name used to build released checkpoints.")
    parser.add_argument("--timm-name", default=None)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", default="outputs/evaluation_metrics.json")
    return parser.parse_args()


def dataset_root(args: argparse.Namespace) -> Path:
    if args.dataset_root:
        root = resolve_path(args.dataset_root)
        assert root is not None
        return root
    data_root = resolve_path(args.data_root)
    assert data_root is not None
    if args.dataset == "custom":
        return data_root
    return data_root / DATASET_DIRS[args.dataset]


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.90062904, 0.90062904, 0.90062904],
                std=[0.26650605, 0.26650605, 0.26650605],
            ),
        ]
    )


def load_state(path: Path, model: torch.nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    model.load_state_dict(state_dict)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    test_root = dataset_root(args) / args.test_dir
    if not test_root.exists():
        raise FileNotFoundError(f"Test directory not found: {test_root}")

    dataset = datasets.ImageFolder(test_root, transform=eval_transform(args.image_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        args.model,
        len(dataset.classes),
        args.timm_name,
        method=args.method,
        image_size=args.image_size,
    ).to(device)
    weight_path = resolve_path(args.weights)
    assert weight_path is not None
    load_state(weight_path, model, device)
    model.eval()

    targets: list[int] = []
    preds: list[int] = []
    for images, target in tqdm(loader):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1).cpu().tolist()
        preds.extend(pred)
        targets.extend(target.tolist())

    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "precision_macro": precision_score(targets, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(targets, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
        "classes": dataset.classes,
        "confusion_matrix": confusion_matrix(targets, preds).tolist(),
    }
    output = resolve_path(args.output)
    assert output is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
