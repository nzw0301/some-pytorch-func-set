import numpy as np
import torch

from contrib.optim.lr_scheduler import get_linear_warmup_cosine_annealing_lr_scheduler


def extract_lrs(optimizer, lr_scheduler, epochs, num_batch_per_epoch) -> np.ndarray:
    lrs = []
    for _ in range(epochs):
        for _ in range(num_batch_per_epoch):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
    return np.array(lrs)


def test_calculate_lr_list_simsiam_small_minibatch() -> None:
    # Case 1: simsiam way's small mini-batch: update lr by epoch + no-warmup
    # Ref: https://arxiv.org/abs/2011.10566

    lr = 0.1
    warmup_epochs = 0
    cosine_epochs = 10
    epochs = warmup_epochs + cosine_epochs
    num_batch_per_epoch = 5
    optimizer = torch.optim.SGD(lr=lr, params=list(torch.tensor([0.1])))

    scheduler = get_linear_warmup_cosine_annealing_lr_scheduler(
        optimizer,
        warmup_epochs,
        cosine_epochs,
        num_batch_per_epoch=num_batch_per_epoch,
        is_lr_update_per_iteration=False,
    )

    lrs = extract_lrs(optimizer, scheduler, epochs, num_batch_per_epoch)
    lrs = lrs.reshape(epochs, num_batch_per_epoch)

    # lr monotonically decreases. at every epoch
    lrs_at_first_iteration_in_epoch = lrs[:, 0]
    assert all(np.diff(lrs_at_first_iteration_in_epoch) < 0.0)

    # first lr should be base lr.
    assert lr == lrs[0, 0]

    # in each epoch, lr should be the same.
    for epoch in range(epochs):
        assert all(lrs[epoch, :] == lrs[epoch, 0])


def test_calculate_lr_list_simsiam_large_minibatch() -> None:
    # Case 2: simsiam way's large mini-batch: update lr by epoch + warmup
    # Ref: https://arxiv.org/abs/2011.10566

    lr = 0.1
    warmup_epochs = 3
    cosine_epochs = 7
    epochs = warmup_epochs + cosine_epochs
    num_batch_per_epoch = 5
    optimizer = torch.optim.SGD(lr=lr, params=list(torch.tensor([0.1])))

    scheduler = get_linear_warmup_cosine_annealing_lr_scheduler(
        optimizer,
        warmup_epochs,
        cosine_epochs,
        num_batch_per_epoch=num_batch_per_epoch,
        is_lr_update_per_iteration=False,
    )
    lrs = extract_lrs(optimizer, scheduler, epochs, num_batch_per_epoch).reshape(
        epochs, num_batch_per_epoch
    )

    # linear warmup period: lr monotonically increases
    expected_linear_warmup_lrs = [0, lr / 2.0, lr]
    for epoch, expected_linear_lr in enumerate(expected_linear_warmup_lrs):
        assert all(lrs[epoch, :] == expected_linear_lr)

    # cosine period: lr monotonically decreases
    lrs_at_first_iteration_in_epoch = lrs[warmup_epochs:, 0]
    assert all(np.diff(lrs_at_first_iteration_in_epoch) < 0.0)

    # in each epoch, lr should be the same.
    for epoch in range(epochs):
        assert all(lrs[epoch, :] == lrs[epoch, 0])


def test_calculate_lr_list_simclr() -> None:
    # case 3: SimCLR/SWaV style: update lr by iteration
    # Ref:
    # SimCLR: https://arxiv.org/abs/2002.05709
    # SWaV: https://arxiv.org/abs/2006.09882

    lr = 0.1
    warmup_epochs = 3
    cosine_epochs = 7
    epochs = warmup_epochs + cosine_epochs
    num_batch_per_epoch = 5

    optimizer = torch.optim.SGD(lr=lr, params=list(torch.tensor([0.1])))

    scheduler = get_linear_warmup_cosine_annealing_lr_scheduler(
        optimizer,
        warmup_epochs,
        cosine_epochs,
        num_batch_per_epoch=num_batch_per_epoch,
        is_lr_update_per_iteration=True,
    )

    lrs = extract_lrs(optimizer, scheduler, epochs, num_batch_per_epoch)

    # linear warmup period: lr monotonically increases
    num_warmup_updates = warmup_epochs * num_batch_per_epoch
    assert all(np.diff(lrs[:num_warmup_updates]) > 0.0)
    # cosine period: lr monotonically decreases
    assert all(np.diff(lrs[num_warmup_updates:]) < 0.0)
