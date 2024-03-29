import numpy as np
import torch


def get_linear_warmup_cosine_annealing_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    cosine_epochs: int,
    num_batch_per_epoch: int,
    is_lr_update_per_iteration: bool = False,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup and cosine annealing without restart learning rate scheduler.

    The calculated value is multiply by base_lr in optimizer by `LambdaLR`.
    Original implementation comes from https://github.com/facebookresearch/swav/blob/master/main_swav.py#L178-L182

    Assume total number of epochs is `warmup_epochs + num_cosine_epoch`.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer instance whose initial
            learning rate should be set correctly.
        warmup_epochs (int): The number of epochs of linear warmup.
        num_cosine_epoch (int): The number of epochs of cosine-annealing
            without restart.
        num_batch_per_epoch (int): The number of batches (: updates) per epoch.
            Usually, this value should be `len(train_dataloader)`. When `is_lr_update_per_iteration`
            is `False`, this value is ignored.
        is_lr_update_per_iteration (bool): If `True`,
            learning rate changes at every iteration.
            Otherwise, learning rate changes at the end of every epoch.
            Defaults to `False`.

    Returns:
        `torch.optim.lr_scheduler.LambdaLR` representing the scheduler.

    NOTE:
        In SimCLR and SWaV style: `is_lr_update_per_iteration=True`.
        In SimSiam style: `is_lr_update_per_iteration=False`.

    Note:
        The first learning rate value is `0` when `warmup_epochs > 0`.

    References:
        Ilya Loshchilov & Frank Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts. In ICLR, 2017.
    """

    if cosine_epochs <= 0:
        raise ValueError(f"`cosine_epochs` should be positive, not {cosine_epochs}.")

    if is_lr_update_per_iteration:
        if num_batch_per_epoch <= 0:
            raise ValueError(
                f"`num_batch_per_epoch` should be positive when `is_lr_update_per_iteration=True`, not {num_batch_per_epoch}."
            )
        num_lr_updates_per_epoch = num_batch_per_epoch
    else:
        num_lr_updates_per_epoch = 1

    num_warmup_lr_updates = warmup_epochs * num_lr_updates_per_epoch
    num_cosine_lr_updates = cosine_epochs * num_lr_updates_per_epoch

    def _linear_warmup_cosine_annealing_without_restart(step: int) -> float:
        _step = step
        if not is_lr_update_per_iteration:
            _step = step // num_batch_per_epoch

        # Depending on the step, lr calculation uses either linear warmup or cosine annealing without restart.
        if _step < num_warmup_lr_updates:
            # Linear warmup part.
            # Since returned value is multipied by base_lr in optimizer,
            # the maximum should be 1.
            return np.linspace(0, 1.0, num_warmup_lr_updates)[_step]

        else:
            # Cosine decay part.
            # Since returned value is multipied by base_lr in optimizer,
            # we drop init_lr from the calculation.
            t = _step - num_warmup_lr_updates
            return 0.5 * (1.0 + np.cos(np.pi * t / num_cosine_lr_updates))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer, lr_lambda=_linear_warmup_cosine_annealing_without_restart
    )
