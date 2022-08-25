import pytest
from contrib.optim.lr_initializer.simclr import calculate_scaled_lr


@pytest.mark.parametrize("lr_scale", ("linear", "square"))
def test_calculate_scaled_lr(lr_scale: str) -> None:
    lr = 0.1
    batch_size = 32
    scaled_lr = calculate_scaled_lr(
        base_lr=lr,
        batch_size=batch_size,
        lr_scale=lr_scale,
    )
    assert scaled_lr > 0.0
