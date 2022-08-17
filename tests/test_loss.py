import torch
from contrib.loss import SNN

num_samples = 4096
dim = 512
num_classes = 10
device = torch.device("cpu")


def test_non_negative() -> None:
    features = torch.rand(num_samples, dim, device=device)
    targets = torch.randint(0, num_classes, (num_samples,), device=device)

    snn = SNN(device)
    for _ in range(5):
        loss = snn(features, targets)
        assert loss.item() > 0.0


def test_shape() -> None:
    features = torch.rand(num_samples, dim, device=device)
    targets = torch.randint(0, num_classes, (num_samples,), device=device)

    snn = SNN(device)
    loss = snn(features, targets)
    assert loss.numpy().shape == ()

    snn = SNN(device, reduce="none")
    loss = snn(features, targets)
    assert loss.numpy().shape == (num_samples,)
