"""
ROCm 兼容性修复 — 必须在 transformers 模型代码导入之前调用 patch_grouped_mm()。

问题: torch._grouped_mm 使用 CK (Composable Kernel) grouped GEMM 后端,
在 ROCm + MI325X 上 workspace buffer 未正确分配, 导致运行时崩溃:
  RuntimeError: The gemm workspace buffer is not allocated!

修复: 用逐专家顺序矩阵乘法替换 _grouped_mm, 功能等价, 训练/推理均兼容。

用法:
  import rocm_compat
  rocm_compat.patch_grouped_mm()   # 必须在 from_pretrained 之前
"""

import torch

_patched = False


def _sequential_grouped_mm(input, weight, offs=None, bias=None, **kwargs):
    """torch._grouped_mm 的顺序回退实现, 兼容所有后端和 PyTorch 版本。"""
    if offs is None:
        out = input @ weight
        if bias is not None:
            out = out + bias
        return out
    output = torch.empty(
        input.shape[0], weight.shape[-1],
        device=input.device, dtype=input.dtype,
    )
    start = 0
    for i in range(len(offs)):
        end = int(offs[i])
        if end > start:
            output[start:end] = input[start:end] @ weight[i]
            if bias is not None:
                output[start:end] += bias[i]
        start = end
    return output


def patch_grouped_mm():
    """替换 torch._grouped_mm 为顺序计算实现。"""
    global _patched
    if _patched:
        return
    if hasattr(torch, "_grouped_mm"):
        torch._grouped_mm = _sequential_grouped_mm
        _patched = True
