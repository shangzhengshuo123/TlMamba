from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from tqdm import tqdm

try:
    import timm
    from timm.data.mixup import Mixup
    from timm.loss import SoftTargetCrossEntropy
except ImportError:  # pragma: no cover - handled at runtime for non-TlMamba models.
    timm = None
    Mixup = None
    SoftTargetCrossEntropy = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))


DATASET_DIRS = {
    "hldlc": "HLDLC1.0",
    "amadi_lontarset": "AMADI_LontarSet",
    "sleukrith": "SleukRith",
    "kuzushiji_ogihan": "Kuzushiji-Ogihan",
}

DATASET_NAME_ALIASES = {
    "hldlc": "hldlc",
    "hldlc1.0": "hldlc",
    "amadi_lontarset": "amadi_lontarset",
    "amadi-lontarset": "amadi_lontarset",
    "amadilontarset": "amadi_lontarset",
    "sleukrith": "sleukrith",
    "kuzushiji_ogihan": "kuzushiji_ogihan",
    "kuzushiji-ogihan": "kuzushiji_ogihan",
    "kuzushijiogihan": "kuzushiji_ogihan",
    "custom": "custom",
}

MODEL_ALIASES = {
    "MambaVision": "mambavision_tiny_1k",
    "VisionMamba": "vim_tiny_patch16_224",
    "PureMamba": "vim_tiny_patch16_224",
    "FastViT": "fastvit_t8",
    "FasterViT": "faster_vit_0_224",
    "ResNet18": "resnet18",
    "DenseNet121": "densenet121",
    "EfficientNet": "efficientnet_b0",
    "SwinTransformer": "swin_tiny_patch4_window7_224",
}

MODEL_NAME_ALIASES = {
    "tlmamba": "TlMamba",
    "mamba_vision": "MambaVision",
    "mambavision": "MambaVision",
    "vision_mamba": "VisionMamba",
    "visionmamba": "VisionMamba",
    "pure_mamba": "PureMamba",
    "puremamba": "PureMamba",
    "fastvit": "FastViT",
    "fast_vit": "FastViT",
    "fastervit": "FasterViT",
    "faster_vit": "FasterViT",
    "crt": "CRT",
    "resnet18": "ResNet18",
    "resnet_18": "ResNet18",
    "densenet121": "DenseNet121",
    "densenet_121": "DenseNet121",
    "efficientnet": "EfficientNet",
    "efficientnet_b0": "EfficientNet",
    "swintransformer": "SwinTransformer",
    "swin_transformer": "SwinTransformer",
}

METHOD_SPECS = {
    "tlmamba_full": {"model": "TlMamba", "loss": "ce"},
    "tlmamba": {"model": "TlMamba", "loss": "ce"},
    "mambavision": {"model": "MambaVision", "loss": "ce"},
    "vision_mamba": {"model": "VisionMamba", "loss": "ce"},
    "pure_mamba": {"model": "PureMamba", "loss": "ce"},
    "fastvit": {"model": "FastViT", "loss": "ce"},
    "fastervit": {"model": "FasterViT", "loss": "ce"},
    "resnet18": {"model": "ResNet18", "loss": "ce"},
    "densenet121": {"model": "DenseNet121", "loss": "ce"},
    "efficientnet": {"model": "EfficientNet", "loss": "ce"},
    "swintransformer": {"model": "SwinTransformer", "loss": "ce"},
    "swin_transformer": {"model": "SwinTransformer", "loss": "ce"},
    "focal_loss": {"model": "ResNet18", "loss": "focal"},
    "class_balanced_loss": {"model": "ResNet18", "loss": "class_balanced"},
    "balanced_softmax": {"model": "ResNet18", "loss": "balanced_softmax"},
    "ldam": {"model": "ResNet18", "loss": "ldam"},
    "crt": {"model": "ResNet18", "loss": "ce", "sampler": "weighted"},
}


@dataclass
class TrainState:
    best_acc: float = 0.0
    start_epoch: int = 1


def model_name(value: str) -> str:
    normalized = value.strip().replace("-", "_").replace(" ", "_").lower()
    if normalized not in MODEL_NAME_ALIASES:
        valid = ", ".join(sorted(MODEL_NAME_ALIASES))
        raise argparse.ArgumentTypeError(f"Unsupported model '{value}'. Valid names: {valid}")
    return MODEL_NAME_ALIASES[normalized]


def dataset_name(value: str) -> str:
    normalized = value.strip().replace(" ", "_").lower()
    if normalized not in DATASET_NAME_ALIASES:
        valid = ", ".join(sorted(DATASET_NAME_ALIASES))
        raise argparse.ArgumentTypeError(f"Unsupported dataset '{value}'. Valid names: {valid}")
    return DATASET_NAME_ALIASES[normalized]


def method_name(value: str) -> str:
    normalized = value.strip().replace("-", "_").replace(" ", "_").lower()
    if normalized not in METHOD_SPECS:
        valid = ", ".join(sorted(METHOD_SPECS))
        raise argparse.ArgumentTypeError(f"Unsupported method '{value}'. Valid names: {valid}")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TlMamba and comparison models in a self-contained release directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=dataset_name,
        default="hldlc",
        help="Dataset preset. Each preset is resolved under --data-root.",
    )
    parser.add_argument("--data-root", default="data", help="Root directory containing released datasets.")
    parser.add_argument("--dataset-root", default=None, help="Override the preset dataset directory.")
    parser.add_argument("--train-dir", default="train")
    parser.add_argument("--val-dir", default="val")
    parser.add_argument("--test-dir", default="test")
    parser.add_argument(
        "--model",
        type=model_name,
        default="TlMamba",
        help="Model name: tlmamba, mambavision, vision_mamba, pure_mamba, fastvit, resnet18, or crt.",
    )
    parser.add_argument("--method", type=method_name, default=None, help="Experiment method name used in released tables.")
    parser.add_argument("--timm-name", default=None, help="Optional timm model name for non-TlMamba baselines.")
    parser.add_argument(
        "--release-setting",
        dest="release_setting",
        action="store_true",
        default=False,
        help="Use the released training defaults.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--auxiliary-epochs", type=int, default=300, help=argparse.SUPPRESS)
    parser.add_argument("--auxiliary-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--auxiliary-image-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--scheduler-t-max", type=int, default=300, help="CosineAnnealingLR T_max.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience", type=int, default=20, help="Stop after N epochs without improvement; 0 disables it.")
    parser.add_argument("--early-stopping-delta", type=float, default=0.0, help="Minimum validation improvement for early stopping.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--run-name", default=None, help="Optional output folder name under outputs/<dataset>/.")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--amp", dest="amp", action="store_true", default=True, help="Enable CUDA automatic mixed precision.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable CUDA automatic mixed precision.")
    parser.add_argument("--ema", action="store_true", help="Maintain an exponential moving average checkpoint.")
    parser.add_argument("--train-drop-last", dest="train_drop_last", action="store_true", default=True)
    parser.add_argument("--no-train-drop-last", dest="train_drop_last", action="store_false")
    parser.add_argument("--mixup", dest="mixup", action="store_true", default=True)
    parser.add_argument("--no-mixup", dest="mixup", action="store_false")
    parser.add_argument("--mixup-alpha", type=float, default=0.8)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-prob", type=float, default=0.1)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
    parser.add_argument("--mixup-label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--teacher-checkpoint",
        default=None,
        help="PiT-B teacher checkpoint for tlmamba_full. Defaults to weights/pit_teacher_<dataset>_seed<seed>_best.pth when available.",
    )
    parser.add_argument("--distill-temperature", type=float, default=4.0)
    parser.add_argument("--distill-classification-weight", type=float, default=0.7)
    parser.add_argument(
        "--distillation",
        default="auto",
        choices=["auto", "on", "off"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--temperature", type=float, default=4.0, help=argparse.SUPPRESS)
    parser.add_argument("--distill-alpha", type=float, default=0.3, help=argparse.SUPPRESS)
    parser.add_argument(
        "--sampler",
        default="none",
        choices=["none", "weighted"],
        help="Optional class-balanced sampler, useful for CRT-style re-training.",
    )
    args = parser.parse_args()
    args.loss = "ce"
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation-steps must be at least 1.")
    if args.min_epochs < 1:
        raise ValueError("--min-epochs must be at least 1.")
    if args.method:
        apply_method(args)
    if args.release_setting:
        apply_release_setting(args)
    args.temperature = args.distill_temperature
    args.distill_alpha = 1.0 - args.distill_classification_weight
    return args


def apply_method(args: argparse.Namespace) -> None:
    spec = METHOD_SPECS[args.method]
    args.model = spec["model"]
    args.loss = spec["loss"]
    if "sampler" in spec:
        args.sampler = spec["sampler"]


def apply_release_setting(args: argparse.Namespace) -> None:
    args.image_size = 224
    args.lr = args.lr or 1e-4
    args.weight_decay = args.weight_decay or 0.05
    if args.teacher_checkpoint is None and args.dataset == "hldlc":
        args.teacher_checkpoint = str(Path(args.weights_dir) / f"pit_teacher_{args.dataset}_seed{args.seed}_best.pth")
    if args.model == "CRT":
        args.sampler = "weighted"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


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


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    mean = [0.90062904, 0.90062904, 0.90062904]
    std = [0.26650605, 0.26650605, 0.26650605]
    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def load_datasets(args: argparse.Namespace) -> tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    root = dataset_root(args)
    train_dir = root / args.train_dir
    val_dir = root / args.val_dir
    test_dir = root / args.test_dir
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    if not val_dir.exists():
        val_dir = test_dir

    train_transform, eval_transform = build_transforms(args.image_size)
    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    val_set = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_set = datasets.ImageFolder(test_dir, transform=eval_transform)
    if train_set.class_to_idx != val_set.class_to_idx:
        raise ValueError("Train and validation class folders do not match.")
    if train_set.class_to_idx != test_set.class_to_idx:
        raise ValueError("Train and test class folders do not match.")
    return train_set, val_set, test_set


def class_counts(dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = torch.zeros(len(dataset.classes), dtype=torch.float32)
    for _, target in dataset.samples:
        counts[target] += 1.0
    return counts.clamp_min(1.0)


def build_sampler(dataset: datasets.ImageFolder, counts: torch.Tensor, enabled: bool) -> WeightedRandomSampler | None:
    if not enabled:
        return None
    weights = [1.0 / counts[target].item() for _, target in dataset.samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, counts: torch.Tensor):
        super().__init__()
        self.register_buffer("log_counts", counts.float().log())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.log_counts.to(logits.device), target)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class LDAMLoss(nn.Module):
    def __init__(self, counts: torch.Tensor, max_m: float = 0.5, scale: float = 30.0):
        super().__init__()
        margins = 1.0 / torch.sqrt(torch.sqrt(counts.float()))
        margins = margins * (max_m / margins.max())
        self.register_buffer("margins", margins)
        self.scale = scale

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        margins = self.margins.to(logits.device)[target]
        adjusted = logits.clone()
        adjusted[torch.arange(logits.size(0), device=logits.device), target] -= margins
        return F.cross_entropy(self.scale * adjusted, target)


class ClassBalancedLoss(nn.Module):
    def __init__(self, counts: torch.Tensor, beta: float = 0.9999):
        super().__init__()
        effective_num = 1.0 - torch.pow(torch.full_like(counts, beta), counts.float())
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(counts)
        self.register_buffer("weights", weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, weight=self.weights.to(logits.device))


def build_loss(name: str, counts: torch.Tensor) -> nn.Module:
    if name == "balanced_softmax":
        return BalancedSoftmaxLoss(counts)
    if name == "ldam":
        return LDAMLoss(counts)
    if name == "class_balanced":
        return ClassBalancedLoss(counts)
    if name == "focal":
        return FocalLoss()
    return nn.CrossEntropyLoss()


def build_mixup(args: argparse.Namespace, loss_name: str, num_classes: int):
    if (
        Mixup is None
        or not args.mixup
        or loss_name != "ce"
        or args.mixup_prob <= 0
        or (args.mixup_alpha <= 0 and args.cutmix_alpha <= 0)
    ):
        return None
    return Mixup(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_minmax=None,
        prob=args.mixup_prob,
        switch_prob=args.mixup_switch_prob,
        mode="batch",
        label_smoothing=args.mixup_label_smoothing,
        num_classes=num_classes,
    )


def create_model(
    name: str,
    num_classes: int,
    timm_name: str | None = None,
    method: str | None = None,
    image_size: int = 224,
) -> nn.Module:
    if name in {"TlMamba", "CRT", "VisionMamba", "PureMamba"}:
        if method == "tlmamba_full":
            from models.revision_ablation_models import build_tlmamba_variant

            return build_tlmamba_variant("full", num_classes)
        if method == "vision_mamba":
            from models.revision_ablation_models import build_vision_mamba

            return build_vision_mamba(num_classes=num_classes, image_size=image_size)
        if method == "pure_mamba":
            from models.revision_ablation_models import build_pure_mamba

            return build_pure_mamba(num_classes=num_classes, image_size=image_size)
        from models.TlMamba import VSSM as TlMamba

        model = TlMamba(num_classes=num_classes)
        if hasattr(model, "head") and hasattr(model.head, "in_features"):
            model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    if timm is None:
        raise ImportError("timm is required for non-TlMamba comparison models.")
    model_name = timm_name or MODEL_ALIASES[name]
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def create_auxiliary_model(num_classes: int, image_size: int, arch: str = "pit_b_224") -> nn.Module:
    if timm is None:
        raise ImportError("timm is required for the internal auxiliary model.")
    return timm.create_model(arch, pretrained=False, num_classes=num_classes, img_size=image_size)


def build_teacher_model(args: argparse.Namespace, num_classes: int, device: torch.device) -> nn.Module:
    teacher = create_auxiliary_model(num_classes, args.image_size, arch="pit_b_224")
    checkpoint_path = resolve_path(args.teacher_checkpoint) if args.teacher_checkpoint else None
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(
            "PiT-B teacher checkpoint is required for tlmamba_full to match the second-review training pipeline. "
            f"Expected: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    teacher.load_state_dict(checkpoint_state_dict(checkpoint))
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    print(f"[teacher] loaded from {checkpoint_path}")
    return teacher


def accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item() * 100.0


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    clip_grad: float = 5.0,
    auxiliary_model: nn.Module | None = None,
    auxiliary_image_size: int | None = None,
    temperature: float = 2.0,
    distill_alpha: float = 0.3,
    distill_classification_weight: float | None = None,
    accumulation_steps: int = 50,
    mixup_fn=None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    iterator = tqdm(loader, leave=False)
    if is_train:
        optimizer.zero_grad(set_to_none=True)
    for step, (images, target) in enumerate(iterator, start=1):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        metric_target = target
        loss_target = target
        if is_train and mixup_fn is not None:
            images, loss_target = mixup_fn(images, target)
        batch_size = target.size(0)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, loss_target)
                if auxiliary_model is not None:
                    classification_loss = loss
                    with torch.no_grad():
                        auxiliary_images = images
                        if auxiliary_image_size and auxiliary_image_size != images.shape[-1]:
                            auxiliary_images = F.interpolate(
                                images,
                                size=(auxiliary_image_size, auxiliary_image_size),
                                mode="bilinear",
                                align_corners=False,
                            )
                        auxiliary_logits = auxiliary_model(auxiliary_images)
                    distill = F.kl_div(
                        F.log_softmax(logits / temperature, dim=1),
                        F.softmax(auxiliary_logits / temperature, dim=1),
                        reduction="batchmean",
                    ) * (temperature**2)
                    if distill_classification_weight is not None:
                        loss = (
                            distill_classification_weight * classification_loss
                            + (1.0 - distill_classification_weight) * distill
                        )
                    else:
                        loss = (1.0 - distill_alpha) * classification_loss + distill_alpha * distill

        if is_train:
            loss_for_backward = loss / accumulation_steps
            if scaler is not None:
                scaler.scale(loss_for_backward).backward()
                if step % accumulation_steps == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss_for_backward.backward()
                if step % accumulation_steps == 0 or step == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * batch_size
        total_acc += accuracy_top1(logits.detach(), metric_target) * batch_size
        total_samples += batch_size
        iterator.set_description(f"{'train' if is_train else 'eval'} loss={loss.item():.4f}")

    return total_loss / total_samples, total_acc / total_samples


def save_class_index(dataset: datasets.ImageFolder, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "class_indices.json").open("w", encoding="utf-8") as file:
        json.dump(dataset.class_to_idx, file, ensure_ascii=False, indent=2)


def save_checkpoint(path: Path, model: nn.Module, epoch: int, best_acc: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({"epoch": epoch, "state_dict": state_dict, "best_acc": best_acc}, path)


def checkpoint_state_dict(checkpoint: object) -> object:
    if isinstance(checkpoint, dict):
        return checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    return checkpoint


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> TrainState:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint_state_dict(checkpoint)
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(state_dict)
    if isinstance(checkpoint, dict):
        return TrainState(best_acc=float(checkpoint.get("best_acc", 0.0)), start_epoch=int(checkpoint.get("epoch", 0)) + 1)
    return TrainState()


def prepare_auxiliary_model(
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    criterion: nn.Module,
    device: torch.device,
    auxiliary_path: Path,
) -> nn.Module:
    auxiliary_image_size = args.auxiliary_image_size or args.image_size
    auxiliary_model = create_auxiliary_model(num_classes, auxiliary_image_size).to(device)
    checkpoint_path = resolve_path(args.auxiliary_checkpoint) if args.auxiliary_checkpoint else auxiliary_path
    assert checkpoint_path is not None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        auxiliary_model.load_state_dict(checkpoint_state_dict(checkpoint))
        auxiliary_model.eval()
        for parameter in auxiliary_model.parameters():
            parameter.requires_grad = False
        return auxiliary_model

    train_mixup = build_mixup(args, "ce", num_classes)
    if train_mixup is not None and SoftTargetCrossEntropy is not None:
        train_criterion = SoftTargetCrossEntropy().to(device)
    else:
        train_criterion = nn.CrossEntropyLoss().to(device)
    eval_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(auxiliary_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.scheduler_t_max),
        eta_min=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None
    best_acc = 0.0
    stale_epochs = 0
    for epoch in range(1, args.auxiliary_epochs + 1):
        run_epoch(
            auxiliary_model,
            train_loader,
            train_criterion,
            optimizer,
            device,
            scaler,
            args.clip_grad,
            accumulation_steps=args.accumulation_steps,
            mixup_fn=train_mixup,
        )
        _, val_acc = run_epoch(auxiliary_model, val_loader, eval_criterion, None, device)
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            stale_epochs = 0
            auxiliary_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(auxiliary_model.state_dict(), auxiliary_path)
        else:
            stale_epochs += 1
        print(f"auxiliary stage {epoch:03d}/{args.auxiliary_epochs:03d} val_acc={val_acc:.2f}")
        if epoch >= args.min_epochs and stale_epochs >= args.early_stopping_patience:
            break

    auxiliary_model.load_state_dict(torch.load(auxiliary_path, map_location=device))
    auxiliary_model.eval()
    for parameter in auxiliary_model.parameters():
        parameter.requires_grad = False
    return auxiliary_model


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = resolve_path(args.output_dir)
    weights_dir = resolve_path(args.weights_dir)
    assert output_dir is not None and weights_dir is not None
    method_key = args.method or args.model
    run_name = args.run_name or f"{method_key}_bs{args.batch_size}_acc{args.accumulation_steps}_seed{args.seed}"
    run_dir = output_dir / args.dataset / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set, test_set = load_datasets(args)
    counts = class_counts(train_set)
    save_class_index(train_set, run_dir)

    sampler = build_sampler(train_set, counts, args.sampler == "weighted" or args.model == "CRT")
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common_loader_args = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if args.num_workers > 0:
        common_loader_args["persistent_workers"] = True

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=args.train_drop_last,
        **common_loader_args,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_args,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        args.model,
        len(train_set.classes),
        args.timm_name,
        method=args.method,
        image_size=args.image_size,
    ).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    mixup_fn = build_mixup(args, args.loss, len(train_set.classes))
    if mixup_fn is not None and SoftTargetCrossEntropy is not None:
        criterion = SoftTargetCrossEntropy().to(device)
    else:
        criterion = build_loss(args.loss, counts).to(device)
    eval_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.scheduler_t_max),
        eta_min=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None

    state = TrainState()
    if args.resume:
        resume_path = resolve_path(args.resume)
        assert resume_path is not None
        state = load_checkpoint(resume_path, model, device)

    use_distillation = args.distillation == "on" or (args.distillation == "auto" and method_key == "tlmamba_full")
    auxiliary_model = None
    if use_distillation:
        auxiliary_model = build_teacher_model(args, len(train_set.classes), device)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_acc": None}
    best_path = run_dir / "best.pth"
    stale_epochs = 0
    for epoch in range(state.start_epoch, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            args.clip_grad,
            auxiliary_model,
            args.image_size,
            args.temperature,
            args.distill_alpha,
            args.distill_classification_weight,
            args.accumulation_steps,
            mixup_fn,
        )
        val_loss, val_acc = run_epoch(model, val_loader, eval_criterion, None, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_acc > state.best_acc + args.early_stopping_delta
        if improved:
            state.best_acc = val_acc
            save_checkpoint(best_path, model, epoch, state.best_acc)
            stale_epochs = 0
        else:
            stale_epochs += 1
        save_checkpoint(run_dir / "last.pth", model, epoch, state.best_acc)
        with (run_dir / "history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
        print(
            f"epoch={epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f} best={state.best_acc:.2f}"
        )
        if (
            epoch >= args.min_epochs
            and args.early_stopping_patience > 0
            and stale_epochs >= args.early_stopping_patience
        ):
            print(f"early stopping at epoch {epoch} after {stale_epochs} stale epochs")
            break

    if best_path.exists():
        load_checkpoint(best_path, model, device)
    _, test_acc = run_epoch(model, test_loader, eval_criterion, None, device)
    history["test_acc"] = test_acc
    with (run_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    print(f"test_acc={test_acc:.2f}")


if __name__ == "__main__":
    main()
