import pytest
from contrib.optim.lr_initializer.simclr import calculate_scaled_lr


@pytest.mark.parametrize("lr_schedule_name", ("linear", "square"))
def test_calculate_scaled_lr(lr_schedule_name: str) -> None:
    lr = 0.1
    batch_size = 32
    scaled_lr = calculate_scaled_lr(
        base_lr=lr,
        batch_size=batch_size,
        lr_schedule=lr_schedule_name,
    )
    assert scaled_lr > 0.0
