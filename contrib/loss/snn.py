import torch


class SNN(torch.nn.Module):
    """
    Implementation of Soft nearest neighbour loss [1].

    [1] Nicholas Frosst, Nicolas Papernot, Geoffrey Hinton.
        Analyzing and Improving Representations with the Soft Nearest Neighbor Loss.
        pages: 2012–2020,
        In ICML,
        2019.
        https://arxiv.org/abs/1902.01889
    """

    def __init__(
        self, device: torch.device, t: float = 1.0, reduce: str = "mean", normalize=True
    ) -> None:

        assert t > 0.0
        assert reduce in {"mean", "sum", "none"}

        self._device = device
        self._t = t
        self._reduce = reduce
        self._normalize = normalize

        super().__init__()

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        N = len(features)

        if self._normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        sim = torch.matmul(features, features.t())

        sim = torch.exp(
            sim.flatten()[1:].view(N - 1, N + 1)[:, :-1].reshape(N, N - 1) / self._t
        )
        denom = torch.sum(sim, dim=1)

        numerator_mask = (
            (targets == targets.view(N, 1))
            .flatten()[1:]
            .view(N - 1, N + 1)[:, :-1]
            .reshape(N, N - 1)
        )

        # TODO(nzw0301): can the following part faster?
        scores = sim * numerator_mask.float()
        loss = torch.log(torch.sum(scores, dim=1) / denom)

        if self._reduce == "sum":
            loss = torch.sum(loss)

        if self._reduce == "mean":
            loss = torch.mean(loss)

        return -loss


if __name__ == "__main__":
    torch.manual_seed(7)
    num_samples = 4096
    dim = 512
    num_classes = 10
    device = torch.device("cpu")

    features = torch.rand(num_samples, dim, device=device)
    targets = torch.randint(0, num_classes, (num_samples,), device=device)

    snn = SNN(device)
    for _ in range(5):
        loss = snn(features, targets)
    print(loss)
