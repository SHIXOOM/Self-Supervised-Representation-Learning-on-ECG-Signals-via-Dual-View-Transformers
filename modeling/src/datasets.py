from typing import Tuple

import numpy as np
import torch
from src import DualAugmenter

def _compute_channel_stats(data: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dataset-level per-channel mean and std with numerical stability."""
    if data.shape[0] == 0:
        raise ValueError("Cannot compute normalization statistics on an empty dataset.")
    means = np.mean(data, axis=(0, 1), dtype=np.float64)
    stds = np.std(data, axis=(0, 1), dtype=np.float64)
    stds = np.maximum(stds, eps)
    return means.astype(np.float32), stds.astype(np.float32)


class ECGContrastiveTrainDataset(torch.utils.data.Dataset):
    """
    ECG dataset for contrastive learning using DualAugmenter.

    Applies dataset-level per-channel Z-score normalization before augmentation.
    Ensures that the two augmented views are guaranteed to be different.

    Args:
        X: Input signals of shape (num_samples, time_steps, num_channels)
        y: Labels (used for filtering valid samples)
        dual_augmenter: DualAugmenter instance
        channel_means: Optional per-channel means to reuse instead of recomputing
        channel_stds: Optional per-channel stds to reuse instead of recomputing
        normalization_eps: Small constant added to stds to avoid divide-by-zero
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dual_augmenter: DualAugmenter,
        channel_means: np.ndarray = None,
        channel_stds: np.ndarray = None,
        normalization_eps: float = 1e-8,
    ) -> None:
        self.X = X
        self.y = y
        self.dual_augmenter = dual_augmenter
        self._num_channels = self.X.shape[2]
        self.normalization_eps = normalization_eps

        if (channel_means is None) != (channel_stds is None):
            raise ValueError("channel_means and channel_stds must both be provided or both be None.")

        if channel_means is None:
            # Compute dataset-level statistics once per dataset
            computed_means, computed_stds = _compute_channel_stats(self.X, self.normalization_eps)
            self.channel_means = computed_means
            self.channel_stds = computed_stds
        else:
            self.channel_means = np.asarray(channel_means, dtype=np.float32)
            self.channel_stds = np.asarray(channel_stds, dtype=np.float32)

        if self.channel_means.shape != (self._num_channels,):
            raise ValueError(
                f"channel_means must have shape ({self._num_channels},), got {self.channel_means.shape}"
            )
        if self.channel_stds.shape != (self._num_channels,):
            raise ValueError(
                f"channel_stds must have shape ({self._num_channels},), got {self.channel_stds.shape}"
            )
        self.channel_stds = np.maximum(self.channel_stds, self.normalization_eps)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get signal and normalize
        signal = self.X[idx, :, :].astype(np.float32)  # (time_steps, num_channels)
        signal = (signal - self.channel_means) / self.channel_stds

        # Use dual augmenter to get two diverse augmented views
        aug1, aug2 = self.dual_augmenter(ecg=signal)

        aug1_tensor = torch.tensor(aug1, dtype=torch.float32)
        aug2_tensor = torch.tensor(aug2, dtype=torch.float32)

        return aug1_tensor, aug2_tensor


class ECGDataset(torch.utils.data.Dataset):
    """
    ECG dataset for downstream tasks (classification, clustering, etc).

    Applies dataset-level per-channel Z-score normalization.
    Returns single signals with their labels for supervised or semi-supervised tasks.

    Assumes labels are already mapped to integers.

    Args:
        X: Input signals of shape (num_samples, time_steps, num_channels)
        y: Integer labels for each signal
        channel_means: Optional per-channel means to reuse instead of recomputing
        channel_stds: Optional per-channel stds to reuse instead of recomputing
        normalization_eps: Small constant added to stds to avoid divide-by-zero
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        channel_means: np.ndarray = None,
        channel_stds: np.ndarray = None,
        normalization_eps: float = 1e-8,
    ) -> None:
        self.X = X
        self.y = y.values if hasattr(y, "values") else np.array(y)

        # Ensure labels are integers
        if not np.issubdtype(self.y.dtype, np.integer):
            raise TypeError(f"Labels must be integers, got {self.y.dtype}")

        # Calculate number of classes
        self.num_classes = int(np.max(self.y)) + 1

        self._num_channels = self.X.shape[2]
        self.normalization_eps = normalization_eps

        if (channel_means is None) != (channel_stds is None):
            raise ValueError("channel_means and channel_stds must both be provided or both be None.")

        if channel_means is None:
            # Compute dataset-level statistics once per dataset
            computed_means, computed_stds = _compute_channel_stats(self.X, self.normalization_eps)
            self.channel_means = computed_means
            self.channel_stds = computed_stds
        else:
            self.channel_means = np.asarray(channel_means, dtype=np.float32)
            self.channel_stds = np.asarray(channel_stds, dtype=np.float32)

        if self.channel_means.shape != (self._num_channels,):
            raise ValueError(
                f"channel_means must have shape ({self._num_channels},), got {self.channel_means.shape}"
            )
        if self.channel_stds.shape != (self._num_channels,):
            raise ValueError(
                f"channel_stds must have shape ({self._num_channels},), got {self.channel_stds.shape}"
            )
        self.channel_stds = np.maximum(self.channel_stds, self.normalization_eps)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get signal and normalize
        signal = self.X[idx, :, :].astype(np.float32)  # (time_steps, num_channels)
        signal = (signal - self.channel_means) / self.channel_stds
        label = int(self.y[idx])

        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        return signal_tensor, label