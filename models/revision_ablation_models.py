from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )


class DirectionAgnosticSPE(nn.Module):
    """SPE ablation retaining multi-branch fusion but removing oriented kernels."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 4):
        super().__init__()
        if out_channels % 4:
            raise ValueError("SPE output channels must be divisible by four.")
        branch_channels = out_channels // 4
        self.branches = nn.ModuleList(
            [
                ConvBNAct(
                    in_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
                for _ in range(4)
            ]
        )
        self.fusion = ConvBNAct(out_channels, out_channels, kernel_size=2, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([branch(x) for branch in self.branches], dim=1))


class SPEWithoutFusionMapping(nn.Module):
    """SPE ablation retaining directional perception without learnable fusion."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 4):
        super().__init__()
        from PConv import Conv

        if out_channels % 4:
            raise ValueError("SPE output channels must be divisible by four.")
        branch_channels = out_channels // 4
        pads = [(kernel_size, 0, 1, 0), (0, kernel_size, 0, 1), (0, 1, kernel_size, 0), (1, 0, 0, kernel_size)]
        self.pads = nn.ModuleList([nn.ZeroPad2d(pad) for pad in pads])
        self.horizontal = Conv(in_channels, branch_channels, (1, kernel_size), s=stride, p=0)
        self.vertical = Conv(in_channels, branch_channels, (kernel_size, 1), s=stride, p=0)
        self.spatial_reduce = nn.AvgPool2d(kernel_size=2, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [
            self.horizontal(self.pads[0](x)),
            self.horizontal(self.pads[1](x)),
            self.vertical(self.pads[2](x)),
            self.vertical(self.pads[3](x)),
        ]
        return self.spatial_reduce(torch.cat(features, dim=1))


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    batch, height, width, channels = x.shape
    channels_per_group = channels // groups
    x = x.view(batch, height, width, groups, channels_per_group)
    return x.transpose(3, 4).contiguous().view(batch, height, width, channels)


class SSDComponentAblation(nn.Module):
    def __init__(
        self,
        source: nn.Module,
        use_structural: bool = True,
        use_semantic: bool = True,
        use_shuffle: bool = True,
    ):
        super().__init__()
        self.use_structural = use_structural
        self.use_semantic = use_semantic
        self.use_shuffle = use_shuffle
        if use_structural:
            self.cf_block = source.cf_block
        if use_semantic:
            self.ln_1 = source.ln_1
            self.self_attention = source.self_attention
            self.drop_path = source.drop_path

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        left, right = inputs.chunk(2, dim=-1)
        if self.use_structural:
            left = left.permute(0, 3, 1, 2).contiguous()
            left = self.cf_block(left)
            left = left.permute(0, 2, 3, 1).contiguous()
        if self.use_semantic:
            right = self.drop_path(self.self_attention(self.ln_1(right)))
        output = torch.cat((left, right), dim=-1)
        if self.use_shuffle:
            output = channel_shuffle(output, groups=2)
        return output + inputs


def _replace_ssd_blocks(
    model: nn.Module,
    *,
    use_structural: bool,
    use_semantic: bool,
    use_shuffle: bool,
) -> None:
    for layer in model.layers:
        for index, block in enumerate(layer.blocks):
            layer.blocks[index] = SSDComponentAblation(
                block,
                use_structural=use_structural,
                use_semantic=use_semantic,
                use_shuffle=use_shuffle,
            )


def _replace_module_attr(model: nn.Module, attr_name: str, replacement_factory) -> int:
    replaced = 0
    for module in model.modules():
        if hasattr(module, attr_name):
            setattr(module, attr_name, replacement_factory())
            replaced += 1
    return replaced


def build_tlmamba_variant(variant: str, num_classes: int) -> nn.Module:
    if variant in {"base", "tskd"}:
        from MedMamba import VSSM

        return VSSM(num_classes=num_classes)
    if variant == "spe":
        from MedMamba_PConv import VSSM

        return VSSM(num_classes=num_classes)
    if variant in {"ssd", "ssd_no_structural", "ssd_no_semantic", "ssd_no_shuffle"}:
        from MedMamba_CFBlock import VSSM

        model = VSSM(num_classes=num_classes)
        if variant == "ssd_no_structural":
            _replace_ssd_blocks(model, use_structural=False, use_semantic=True, use_shuffle=True)
        elif variant == "ssd_no_semantic":
            _replace_ssd_blocks(model, use_structural=True, use_semantic=False, use_shuffle=True)
        elif variant == "ssd_no_shuffle":
            _replace_ssd_blocks(model, use_structural=True, use_semantic=True, use_shuffle=False)
        return model
    if variant in {"spe_no_direction", "spe_no_fusion"}:
        from MedMamba_PConv import VSSM

        model = VSSM(num_classes=num_classes)
        if variant == "spe_no_direction":
            model.patch_embed.proj = DirectionAgnosticSPE(3, model.embed_dim)
        else:
            model.patch_embed.proj = SPEWithoutFusionMapping(3, model.embed_dim)
        return model
    if variant == "spe_ssd_no_cfblock":
        from MedMamba_PConv_CFBlock import VSSM

        model = VSSM(num_classes=num_classes)
        replaced = _replace_module_attr(model, "cf_block", nn.Identity)
        if replaced == 0:
            raise RuntimeError("No cf_block modules were found for spe_ssd_no_cfblock.")
        return model
    if variant in {"spe_ssd", "full"}:
        from MedMamba_PConv_CFBlock import VSSM

        return VSSM(num_classes=num_classes)
    raise ValueError(f"Unknown TlMamba variant: {variant}")


class MambaResidualBlock(nn.Module):
    def __init__(self, dim: int, bidirectional: bool = False, d_state: int = 16):
        super().__init__()
        from mamba_ssm import Mamba

        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(dim)
        self.forward_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=2)
        self.backward_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=2) if bidirectional else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        output = self.forward_mamba(normalized)
        if self.backward_mamba is not None:
            backward = torch.flip(self.backward_mamba(torch.flip(normalized, dims=[1])), dims=[1])
            output = 0.5 * (output + backward)
        return x + output


class PatchMambaClassifier(nn.Module):
    """Isolated-character adaptation used for Pure Mamba and Vision Mamba rows."""

    def __init__(
        self,
        num_classes: int,
        image_size: int = 224,
        patch_size: int = 16,
        dim: int = 192,
        depth: int = 8,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.position = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.blocks = nn.Sequential(
            *[MambaResidualBlock(dim=dim, bidirectional=bidirectional) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.position[:, : x.shape[1]]
        x = self.blocks(x)
        return self.head(self.norm(x).mean(dim=1))


def build_pure_mamba(num_classes: int, image_size: int = 224) -> nn.Module:
    return PatchMambaClassifier(num_classes=num_classes, image_size=image_size, bidirectional=False)


def build_vision_mamba(num_classes: int, image_size: int = 224) -> nn.Module:
    return PatchMambaClassifier(num_classes=num_classes, image_size=image_size, bidirectional=True)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def classifier_parameters(model: nn.Module) -> Sequence[nn.Parameter]:
    return list(classifier_module(model).parameters())


def classifier_module(model: nn.Module) -> nn.Module:
    for name in ("head", "fc", "classifier"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            return module
    if hasattr(model, "get_classifier"):
        module = model.get_classifier()
        if isinstance(module, nn.Module):
            return module
    raise RuntimeError(f"Unable to locate classifier parameters for {type(model).__name__}.")
