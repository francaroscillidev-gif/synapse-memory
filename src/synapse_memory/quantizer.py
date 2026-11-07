from typing import Tuple
import numpy as np


class VectorQuantizer:
    """Provides scalar Int8 quantization for dense embedding vectors (75% memory footprint reduction)."""

    @staticmethod
    def quantize_to_int8(vector: np.ndarray) -> Tuple[bytes, float, float]:
        """Quantize a float32 vector to int8 bytes, returning (int8_bytes, scale, min_val)."""
        vec = vector.astype(np.float32)
        min_val = float(np.min(vec))
        max_val = float(np.max(vec))
        diff = max_val - min_val if max_val != min_val else 1.0

        scale = diff / 255.0
        # Map to 0..255 then offset to -128..127
        quantized = np.clip(np.round((vec - min_val) / scale) - 128, -128, 127).astype(np.int8)
        return quantized.tobytes(), scale, min_val

    @staticmethod
    def dequantize_from_int8(raw_bytes: bytes, scale: float, min_val: float) -> np.ndarray:
        """Dequantize int8 bytes back to normalized float32 vector."""
        int8_arr = np.frombuffer(raw_bytes, dtype=np.int8).astype(np.float32)
        dequantized = (int8_arr + 128.0) * scale + min_val
        # Normalize
        norm = np.linalg.norm(dequantized)
        if norm > 0:
            dequantized = dequantized / norm
        return dequantized.astype(np.float32)

    @staticmethod
    def fast_cosine_similarity(query_norm: np.ndarray, doc_norm: np.ndarray) -> float:
        """Compute cosine similarity between two unit vectors."""
        return float(np.dot(query_norm, doc_norm))
