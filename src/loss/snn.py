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
        self, device: torch.device, t: float = 1.0, reduce: str = "mean"
    ) -> None:

        assert t > 0.0
        assert reduce in {"mean", "sum", "none"}

        self._device = device
        self._t = t
        self._reduce = reduce

        super().__init__()

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        N = len(features)
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
        loss_per_sample = []

        # TODO(nzw0301): can the following part faster?
        for s, n, d in zip(sim, numerator_mask, denom):
            scores = s[n]
            if len(scores):
                loss_per_sample.append(torch.log(torch.sum(scores) / d))

        loss = torch.stack(loss_per_sample)

        if self._reduce != "none":
            loss = torch.sum(loss)

        if self._reduce == "mean":
            loss /= N

        return -loss


if __name__ == "__main__":
    device = torch.device("cpu")
    features = torch.rand(4, 8, device=device)
    targets = torch.tensor([0, 0, 0, 2], device=device)

    snn = SNN(device)

    print(snn(features, targets))
