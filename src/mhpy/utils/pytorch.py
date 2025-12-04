from loguru import logger
import torch
import torch.nn as nn


def get_dtype(use_gpu: bool = True, use_bf16: bool = True) -> torch.dtype:
    if use_gpu and torch.cuda.is_available():
        support_native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        use_bf16 = use_bf16 and support_native_bf16

        if use_bf16:
            return torch.bfloat16

        if not support_native_bf16:
            logger.warning("bfloat16 is not natively supported. Falling back to float16.")
        logger.warning("Make sure you use GradScaler for mixed precision!")

        return torch.float16

    return torch.float32


def log_model_size(model: nn.Module) -> None:
    param_count, size_all_mb = get_model_size(model)
    logger.info(f"Model {param_count} parameters and size of {size_all_mb:.2f} MB")


def get_model_size(model: nn.Module) -> tuple[int, float]:
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return param_count, size_all_mb
