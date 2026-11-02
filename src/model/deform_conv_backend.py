"""Lazy, explicit DCN backend; torchvision inference needs no local extension."""

import torch
from torchvision.ops import deform_conv2d


def modulated_deform_conv2d(input, offset, mask, weight, bias, stride=1,
                          padding=0, dilation=1, groups=1, deformable_groups=1,
                          im2col_step=64, backend="torchvision"):
    if backend == "torchvision":
        # Propagation accumulates in FP32, including under native AMP.
        with torch.autocast(device_type=input.device.type, enabled=False):
            return deform_conv2d(input.float(), offset.float(), weight.float(),
                                 None if bias is None else bias.float(), stride=stride,
                                 padding=padding, dilation=dilation, mask=mask.float())
    if backend == "legacy_dcn":
        from .modulated_deform_conv_func import ModulatedDeformConvFunction
        return ModulatedDeformConvFunction.apply(
            input, offset, mask, weight, bias, stride, padding, dilation,
            groups, deformable_groups, im2col_step)
    raise ValueError(f"Unsupported DCN backend: {backend}")
