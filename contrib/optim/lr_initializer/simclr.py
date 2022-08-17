import numpy as np

def calculate_scaled_lr(
    base_lr: float, batch_size: int, lr_schedule: str = "linear"
) -> float:
    """Mini-batch size dependent initial learning rate proposed by Chen et al.

    Note: SimCLR paper says squared learning rate is recommended
        when the size of mini-batches is small.

    Args:
        base_lr (float): Base learning rate.
        batch_size (int): The number of batches in a mini-batch.
        lr_schedule (str): Type of lr calculation. Should be either
            "linear" or "square". Defaults to "linear".

    Returns:
        float: Initial learning rate.

    References:
        Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton, A Simple Framework for Contrastive Learning of Visual Representations. In ICML, 119:1597-1607, 2020.
    """

    assert base_lr > 0.0
    assert batch_size >= 1
    assert lr_schedule in {"linear", "square"}

    if lr_schedule == "linear":
        scaled_lr = base_lr * batch_size / 256.0
    else:
        scaled_lr = base_lr * np.sqrt(batch_size)

    return scaled_lr
