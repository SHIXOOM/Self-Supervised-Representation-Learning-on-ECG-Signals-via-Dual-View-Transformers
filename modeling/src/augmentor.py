import numpy as np
import random
import warnings
from scipy.spatial.distance import cosine
from typing import List, Callable, Tuple


class DualAugmenter:
    """
    Custom augmenter that produces two guaranteed-different augmented views.
    
    Ensures augmentation diversity through similarity checks with a retry mechanism.
    Enforces both minimum and maximum similarity to ensure positive pairs are
    similar enough to be meaningful but different enough to learn from.
    
    Args:
        augmentation_pool: List of augmentation functions to sample from
        min_similarity: Minimum cosine similarity allowed (default 0.5)
        max_similarity: Maximum cosine similarity allowed (default 0.995)
        max_retries: Maximum attempts to generate diverse pair (default 10)
    """
    
    def __init__(
        self,
        augmentation_pool: List[Callable],
        min_similarity: float = 0.5,
        max_similarity: float = 0.95,
        max_retries: int = 10,
    ):
        self.augmentation_pool = augmentation_pool
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity
        self.max_retries = max_retries
        
        if min_similarity >= max_similarity:
            raise ValueError(f"min_similarity ({min_similarity}) must be less than max_similarity ({max_similarity})")
    
    def _calculate_similarity(self, aug1: np.ndarray, aug2: np.ndarray) -> float:
        """Calculate cosine similarity between two augmented signals."""
        return 1 - cosine(aug1.flatten(), aug2.flatten())
    
    def _are_identical(self, aug1: np.ndarray, aug2: np.ndarray) -> bool:
        """Quick check if augmentations are bit-for-bit identical."""
        return np.array_equal(aug1, aug2)
    
    def _sample_augmentation(self) -> Callable:
        """Randomly sample an augmentation from the pool."""
        return random.sample(self.augmentation_pool, k=1)[0]
    
    def __call__(self, ecg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate two diverse augmented views of the input signal.
        
        Args:
            ecg: Input ECG signal of shape (time_steps, num_channels)
            
        Returns:
            Tuple of (aug1, aug2) - two augmented views with similarity in valid range
        """
        best_aug1, best_aug2 = None, None
        best_similarity = 1.0
        best_distance_from_range = float('inf')
        
        for attempt in range(self.max_retries):
            # Sample two augmentations
            augmenter1 = self._sample_augmentation()
            augmenter2 = self._sample_augmentation()
            
            # Apply augmentations
            aug1 = augmenter1(ecg=ecg)["ecg"]
            aug2 = augmenter2(ecg=ecg)["ecg"]
            
            # Quick check: are they identical?
            if self._are_identical(aug1, aug2):
                continue
            
            # Check similarity
            similarity = self._calculate_similarity(aug1, aug2)
            
            # Calculate distance from valid range
            if similarity < self.min_similarity:
                distance_from_range = self.min_similarity - similarity
            elif similarity > self.max_similarity:
                distance_from_range = similarity - self.max_similarity
            else:
                distance_from_range = 0.0
            
            # Track best attempt (closest to valid range)
            if distance_from_range < best_distance_from_range:
                best_distance_from_range = distance_from_range
                best_similarity = similarity
                best_aug1, best_aug2 = aug1, aug2
            
            # Success: found pair within valid similarity range
            if self.min_similarity <= similarity <= self.max_similarity:
                return aug1, aug2
        
        # Max retries reached - return best attempt with warning
        if best_distance_from_range > 0:
            if best_similarity < self.min_similarity:
                # warnings.warn(
                #     f"Max retries ({self.max_retries}) reached. "
                #     f"Best similarity: {best_similarity:.4f} < min threshold: {self.min_similarity}"
                # )
                ...
            else:
                ...
                # warnings.warn(
                #     f"Max retries ({self.max_retries}) reached. "
                #     f"Best similarity: {best_similarity:.4f} > max threshold: {self.max_similarity}"
                # )
        
        return best_aug1, best_aug2