from loguru import logger
import torch.nn as nn


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
