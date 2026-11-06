import numpy as np
from typing import List
from fastembed import TextEmbedding

# Default lightweight model, executes on CPU/Metal locally with ONNX
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class LocalEmbedder:
    _instance = None

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            # Lazy initialization
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string into a normalized numpy vector."""
        model = self._get_model()
        generator = model.embed([text])
        vector = next(generator)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return np.array(vector, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple strings efficiently."""
        if not texts:
            return []
        model = self._get_model()
        vectors = []
        for vec in model.embed(texts):
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(np.array(vec, dtype=np.float32))
        return vectors
