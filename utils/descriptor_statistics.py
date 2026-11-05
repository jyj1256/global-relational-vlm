import json
from pathlib import Path

import torch


class OnlineDescriptorStatistics:
    """
    Online mean and variance calculation using
    the numerically stable Welford algorithm.
    """

    def __init__(
        self,
        descriptor_names,
        eps=1e-6,
    ):
        self.descriptor_names = tuple(
            descriptor_names
        )

        self.dimension = len(
            self.descriptor_names
        )

        self.eps = eps
        self.count = 0

        self.mean = torch.zeros(
            self.dimension,
            dtype=torch.float64,
        )

        self.m2 = torch.zeros(
            self.dimension,
            dtype=torch.float64,
        )

        self.minimum = torch.full(
            (self.dimension,),
            float("inf"),
            dtype=torch.float64,
        )

        self.maximum = torch.full(
            (self.dimension,),
            float("-inf"),
            dtype=torch.float64,
        )

    def update(
        self,
        descriptor_batch,
    ):
        descriptor_batch = torch.as_tensor(
            descriptor_batch,
            dtype=torch.float64,
        )

        if descriptor_batch.ndim == 1:
            descriptor_batch = (
                descriptor_batch.unsqueeze(0)
            )

        if descriptor_batch.ndim != 2:
            raise ValueError(
                "descriptor_batch must have "
                "shape (B, D)."
            )

        if descriptor_batch.shape[1] != (
            self.dimension
        ):
            raise ValueError(
                f"Expected descriptor dimension "
                f"{self.dimension}, but received "
                f"{descriptor_batch.shape[1]}."
            )

        if not torch.isfinite(
            descriptor_batch
        ).all():
            raise ValueError(
                "Descriptor batch contains NaN or Inf."
            )

        batch_count = descriptor_batch.shape[0]

        if batch_count == 0:
            return

        batch_mean = descriptor_batch.mean(dim=0)

        batch_difference = (
            descriptor_batch - batch_mean
        )

        batch_m2 = (
            batch_difference.square().sum(dim=0)
        )

        batch_minimum = descriptor_batch.min(
            dim=0
        ).values

        batch_maximum = descriptor_batch.max(
            dim=0
        ).values

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean.clone()
            self.m2 = batch_m2.clone()
            self.minimum = batch_minimum.clone()
            self.maximum = batch_maximum.clone()
            return

        total_count = self.count + batch_count

        delta = batch_mean - self.mean

        self.mean = (
            self.mean
            + delta
            * (batch_count / total_count)
        )

        self.m2 = (
            self.m2
            + batch_m2
            + delta.square()
            * (
                self.count
                * batch_count
                / total_count
            )
        )

        self.minimum = torch.minimum(
            self.minimum,
            batch_minimum,
        )

        self.maximum = torch.maximum(
            self.maximum,
            batch_maximum,
        )

        self.count = total_count

    def state_dict(self):
        return {
            "descriptor_names": list(
                self.descriptor_names
            ),
            "eps": self.eps,
            "count": self.count,
            "mean": self.mean.clone(),
            "m2": self.m2.clone(),
            "minimum": self.minimum.clone(),
            "maximum": self.maximum.clone(),
        }

    def load_state_dict(self, state):
        state_names = tuple(
            state["descriptor_names"]
        )

        if state_names != self.descriptor_names:
            raise ValueError(
                "Checkpoint descriptor names do not match."
            )

        count = int(state["count"])

        if count < 0:
            raise ValueError(
                "Checkpoint count cannot be negative."
            )

        tensors = {
            "mean": torch.as_tensor(
                state["mean"],
                dtype=torch.float64,
            ),
            "m2": torch.as_tensor(
                state["m2"],
                dtype=torch.float64,
            ),
            "minimum": torch.as_tensor(
                state["minimum"],
                dtype=torch.float64,
            ),
            "maximum": torch.as_tensor(
                state["maximum"],
                dtype=torch.float64,
            ),
        }

        expected_shape = (self.dimension,)

        for name, tensor in tensors.items():
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"Unexpected checkpoint {name} shape: "
                    f"{tuple(tensor.shape)}"
                )

        self.count = count
        self.mean = tensors["mean"].clone()
        self.m2 = tensors["m2"].clone()
        self.minimum = tensors["minimum"].clone()
        self.maximum = tensors["maximum"].clone()

        return self

    def finalize(self):
        if self.count == 0:
            raise RuntimeError(
                "No descriptors were accumulated."
            )

        variance = self.m2 / float(self.count)

        variance = variance.clamp_min(0.0)

        raw_std = torch.sqrt(variance)

        safe_std = raw_std.clamp_min(
            self.eps
        )

        return {
            "count": self.count,
            "descriptor_names": list(
                self.descriptor_names
            ),
            "mean": self.mean.float(),
            "std": safe_std.float(),
            "raw_std": raw_std.float(),
            "variance": variance.float(),
            "minimum": self.minimum.float(),
            "maximum": self.maximum.float(),
            "eps": self.eps,
        }


def save_descriptor_statistics(
    output_path,
    statistics,
    metadata=None,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "count": int(statistics["count"]),
        "descriptor_names": list(
            statistics["descriptor_names"]
        ),
        "mean": (
            statistics["mean"].tolist()
        ),
        "std": statistics["std"].tolist(),
        "raw_std": (
            statistics["raw_std"].tolist()
        ),
        "variance": (
            statistics["variance"].tolist()
        ),
        "minimum": (
            statistics["minimum"].tolist()
        ),
        "maximum": (
            statistics["maximum"].tolist()
        ),
        "eps": float(statistics["eps"]),
        "metadata": metadata or {},
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_descriptor_statistics(
    input_path,
    expected_descriptor_names=None,
):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    descriptor_names = tuple(
        payload["descriptor_names"]
    )

    if expected_descriptor_names is not None:
        expected_descriptor_names = tuple(
            expected_descriptor_names
        )

        if descriptor_names != (
            expected_descriptor_names
        ):
            raise ValueError(
                "Descriptor name/order mismatch."
            )

    mean = torch.tensor(
        payload["mean"],
        dtype=torch.float32,
    )

    std = torch.tensor(
        payload["std"],
        dtype=torch.float32,
    )

    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError(
            "Mean and std must be one-dimensional."
        )

    if mean.shape != std.shape:
        raise ValueError(
            "Mean and std shapes do not match."
        )

    if torch.any(std <= 0.0):
        raise ValueError(
            "All standard deviations must be positive."
        )

    return {
        "count": int(payload["count"]),
        "descriptor_names": descriptor_names,
        "mean": mean,
        "std": std,
        "raw_std": torch.tensor(
            payload["raw_std"],
            dtype=torch.float32,
        ),
        "variance": torch.tensor(
            payload["variance"],
            dtype=torch.float32,
        ),
        "minimum": torch.tensor(
            payload["minimum"],
            dtype=torch.float32,
        ),
        "maximum": torch.tensor(
            payload["maximum"],
            dtype=torch.float32,
        ),
        "eps": float(payload["eps"]),
        "metadata": payload.get(
            "metadata",
            {},
        ),
    }