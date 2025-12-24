# app/embeddings/service.py
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import math
import os

_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # "cpu" safe en docker
_model: Optional[SentenceTransformer] = None

def get_model_name() -> str:
    return _MODEL_NAME

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # IMPORTANT: charge directement sur le device, évite .to(device) ensuite
        _model = SentenceTransformer(_MODEL_NAME, device=_DEVICE)
    return _model

def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return vec if norm == 0 else [x / norm for x in vec]

def embed_text(text: str, normalize: bool = True) -> List[float]:
    if not text or not text.strip():
        return []
    model = _get_model()
    vec = model.encode(text).tolist()
    return _l2_normalize(vec) if normalize else vec
