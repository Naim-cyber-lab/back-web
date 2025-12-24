from typing import List
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from sentence_transformers import SentenceTransformer
import math
from app.embeddings.service import embed_text, get_model_name


router = APIRouter()

# 🔥 Modèle plus puissant que all-MiniLM-L6-v2
# - Meilleure qualité sémantique
# - Vecteurs de dimension 768
_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Charge le modèle globalement (lazy loading).
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _l2_normalize(vec: List[float]) -> List[float]:
    """
    Optionnel : normalisation L2 de l'embedding.
    Utile si tu utilises beaucoup la cosine similarity.
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def get_embedding(text: str, normalize: bool = True) -> List[float]:
    """
    Prend un texte en entrée et renvoie son embedding
    sous forme de liste de floats.
    - Utilise un modèle plus puissant (all-mpnet-base-v2)
    - Normalise L2 par défaut (pratique pour similarity)
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    model = _get_model()
    vector = model.encode(text)
    vec_list = vector.tolist()

    if normalize:
        vec_list = _l2_normalize(vec_list)

    return vec_list


class EmbeddingRequest(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int
    model_name: str


@router.post("/embeddings", response_model=EmbeddingResponse, tags=["embeddings"])
def create_embedding(payload: EmbeddingRequest):
    vec = embed_text(payload.text, normalize=True)

    if not vec:
        raise HTTPException(status_code=400, detail="Text is empty or embedding is empty.")

    return EmbeddingResponse(
        embedding=vec,
        dimension=len(vec),
        model_name=get_model_name(),
    )
