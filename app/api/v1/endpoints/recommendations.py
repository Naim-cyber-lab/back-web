from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
import requests
from elasticsearch import Elasticsearch
import os
from app.core.db import *
from app.mappings import *
from app.schemas import *
from app.api.v1.sql.fetch_events_with_relations_by_ids import *
from app.embeddings.service import embed_text
from app.api.v1.sql.fetch_winkers_by_ids import *
from app.api.utils import *
from datetime import datetime, timezone, date
import hashlib

router = APIRouter()

# ---- Config ----
ES_INDEX = "nisu_events"
ES_HOST = os.getenv("ES_HOST", "http://192.168.1.213:9200")
ES_USER = os.getenv("ES_USER")
ES_PASS = os.getenv("ES_PASS")

EMBEDDINGS_URL = "https://recommendation.nisu.fr/api/v1/recommendations/embeddings"
EMBEDDINGS_TIMEOUT = 60

es = Elasticsearch(
    [ES_HOST],
    basic_auth=(ES_USER, ES_PASS) if ES_USER and ES_PASS else None,
)

# ---- Helpers ----


def get_embedding(text: str) -> List[float]:
    return embed_text(text, normalize=True)

def build_winker_profile_text(w: Dict[str, Any]) -> str:
    """
    Construit un texte stable qui représente le profil du winker.
    Tu peux l’ajuster selon tes champs vraiment utiles.
    """
    parts: List[str] = []

    bio = (w.get("bio") or "").strip()
    if bio:
        parts.append(bio)

    # localisation
    city = (w.get("city") or "").strip()
    region = (w.get("region") or "").strip()
    subregion = (w.get("subregion") or "").strip()
    if city or region or subregion:
        parts.append(" ".join([p for p in [city, subregion, region] if p]))

    # dernière recherche event (souvent super signal)
    last_search = (w.get("derniereRechercheEvent") or "").strip()
    if last_search and last_search not in ("{}", "[]", "null"):
        parts.append(last_search)

    # si tu stockes des tags/prefer dans d'autres champs, ajoute-les ici

    return " | ".join(parts).strip()


def parse_geo(w: Dict[str, Any]) -> Optional[Dict[str, float]]:
    lat = w.get("lat")
    lon = w.get("lon")
    try:
        if lat is None or lon is None:
            return None
        return {"lat": float(lat), "lon": float(lon)}
    except Exception:
        return None


# ---- Ta route existante (inchangée) ----
@router.get("/get_rencontre_from_winker/{user_id}")
def get_profil_winker_raw(user_id: int):

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM profil_winker WHERE id = %s", (user_id,))
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Profil introuvable")
            columns = [desc[0] for desc in cur.description]

    return dict(zip(columns, row))


# ---- Nouvelle route : top 4 events ----

@router.get("/get_events_for_winker/{user_id}", response_model=List[EventOut])
def get_events_for_winker(user_id: int) -> List[EventOut]:
    winker = get_profil_winker_raw(user_id)

    profile_text = build_winker_profile_text(winker)
    if not profile_text:
        raise HTTPException(status_code=400, detail="Profil trop vide pour recommander des events.")

    qvec = get_embedding(profile_text)
    if not qvec:
        raise HTTPException(status_code=500, detail="Impossible de générer l'embedding du profil.")

    user_geo = parse_geo(winker)

    knn_query: Dict[str, Any] = {
        "field": "embedding_vector",
        "query_vector": qvec,
        "k": 50,
        "num_candidates": 200
    }

    base_query: Dict[str, Any] = {"match_all": {}}
    if user_geo:
        base_query = {
            "bool": {
                "filter": [
                    {"geo_distance": {"distance": "200km", "localisation": user_geo}}
                ]
            }
        }

    # seed stable "par jour" et par user (même résultats dans la journée, change le lendemain)
    seed_str = f"{user_id}-{date.today().isoformat()}"          # ex: "123-2025-12-17"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)  # int 32-bit

    body: Dict[str, Any] = {
        "size": 4,
        "query": base_query,
        "knn": knn_query,
        "rescore": {
            "window_size": 50,
            "query": {
                "rescore_query": {
                    "function_score": {
                        "query": {"match_all": {}},
                        "functions": [
                            # ✅ 1) Boost proximité géographique
                            *([
                                {
                                    "gauss": {
                                        "localisation": {
                                            "origin": user_geo,     # {"lat":..., "lon":...}
                                            "scale": "25km",
                                            "offset": "2km",
                                            "decay": 0.5
                                        }
                                    },
                                    "weight": 6.0
                                }
                            ] if user_geo else []),

                            # ✅ 2) Ton boost existant
                            {
                                "field_value_factor": {
                                    "field": "boost",
                                    "factor": 0.05,
                                    "missing": 0
                                },
                                "weight": 1.0
                            },

                            # ✅ 3) Variation stable par jour (petit bruit aléatoire)
                            {
                                "random_score": {
                                    "seed": seed,
                                    "field": "_seq_no"   # ou "_id" si tu préfères
                                },
                                "weight": 0.15         # augmente si tu veux plus de variété
                            }
                        ],
                        "score_mode": "sum",
                        "boost_mode": "sum"
                    }
                },
                "query_weight": 0.7,
                "rescore_query_weight": 1.6
            }
        },
        "_source": False,
    }
    resp = es.search(index=ES_INDEX, body=body)
    hits = resp.get("hits", {}).get("hits", [])

    event_ids: List[int] = []
    for h in hits:
        try:
            event_ids.append(int(h.get("_id")))
        except Exception:
            continue

    if not event_ids:
        return []

    # ====== ICI: tu remplaces par TON accès DB (SQLAlchemy / raw SQL / repo) ======
    # Objectif: récupérer Event + creatorWinker + participants + participants.participeWinker
    events_orm = fetch_events_with_relations_by_ids(event_ids)
    # ============================================================================

    # préserver l'ordre ES
    by_id = {e.get("id"): e for e in events_orm}
    ordered = [by_id[eid] for eid in event_ids if eid in by_id]

    return [to_event_out(e) for e in ordered]




def age_from_birth_year(birth_year: Any) -> int:
    """
    Calcule l'âge à partir de birthYear (année), avec garde-fous.
    """
    try:
        by = int(birth_year)
        now_year = datetime.now(timezone.utc).year
        age = now_year - by
        return max(0, min(age, 120))
    except Exception:
        return 0


@router.get("/get_winkers_for_winker/{user_id}")
def get_winkers_for_winker(
    user_id: int,
    limit: int = Query(12, ge=1, le=50),
    radius_km: int = Query(30, ge=1, le=300),
) -> List[Dict[str, Any]]:
    """
    Reco: winkers proches + similarité profil (KNN embeddings).
    Puis re-ranking (rescore) avec:
      - activité récente (gauss sur lastConnection)
      - proximité géographique (gauss sur localisation)
      - proximité d'âge (gauss sur age, origin = âge calculé depuis birthYear du demandeur)
    """
    winker = get_profil_winker_raw(user_id)

    profile_text = build_winker_profile_text(winker)
    if not profile_text:
        raise HTTPException(status_code=400, detail="Profil trop vide pour recommander des winkers.")

    qvec = get_embedding(profile_text)
    if not qvec:
        raise HTTPException(status_code=500, detail="Impossible de générer l'embedding du profil.")

    user_geo = parse_geo(winker)
    if not user_geo:
        raise HTTPException(status_code=400, detail="Pas de localisation (lat/lon) sur le profil winker.")

    # âge du demandeur à partir de birthYear
    user_age = age_from_birth_year(winker.get("birthYear"))

    # ------- Filtres métier minimum -------
    must_filters: List[Dict[str, Any]] = [
        {"term": {"is_active": True}},
        {"term": {"is_banned": False}},
        {"bool": {"must_not": [{"term": {"_id": str(user_id)}}]}},  # exclure soi-même
        {"geo_distance": {"distance": f"{radius_km}km", "localisation": user_geo}},
    ]

    base_query: Dict[str, Any] = {"bool": {"filter": must_filters}}

    knn_query: Dict[str, Any] = {
        "field": "embedding_vector",
        "query_vector": qvec,
        "k": 200,                 # on récupère large
        "num_candidates": 1000,   # selon taille index
    }

    # ------- Rescore: gauss activité + gauss geo + gauss âge (+ boost) -------
    functions: List[Dict[str, Any]] = [
        # 1) Récence de connexion: plus récent = mieux
        {
            "filter": {"exists": {"field": "lastConnection"}},
            "gauss": {
                "lastConnection": {
                    "origin": "now",
                    "scale": "3d",     # à tuner (2d / 7d / 14d)
                    "offset": "0d",
                    "decay": 0.4
                }
            },
            "weight": 3.0
        },

        # 2) Proximité géographique: plus proche = mieux
        {
            "gauss": {
                "localisation": {
                    "origin": f"{user_geo['lat']},{user_geo['lon']}",
                    "scale": "10km",   # à tuner (5km / 15km / 25km)
                    "offset": "1km",
                    "decay": 0.5
                }
            },
            "weight": 2.0
        },
    ]

    # 3) Proximité d'âge: compare au champ ES "age"
    # (origin = âge calculé depuis winker.birthYear)
    if user_age > 0:
        functions.append({
            "filter": {"exists": {"field": "age"}},
            "gauss": {
                "age": {
                    "origin": user_age,
                    "scale": 5,     # tolérance ~5 ans
                    "offset": 1,    # 1 an sans pénalité
                    "decay": 0.5
                }
            },
            "weight": 1.5
        })

    # Bonus (comme avant)
    functions.append({
        "field_value_factor": {"field": "boost", "factor": 0.05, "missing": 0},
        "weight": 1.0
    })

    body: Dict[str, Any] = {
        "size": limit,
        "query": base_query,
        "knn": knn_query,
        "rescore": {
            "window_size": 200,
            "query": {
                "rescore_query": {
                    "function_score": {
                        "query": {"match_all": {}},
                        "functions": functions,
                        "boost_mode": "sum",
                        "score_mode": "sum",
                    }
                },
                "query_weight": 1.0,
                "rescore_query_weight": 1.0,
            }
        },
        "_source": False,  # on veut juste les ids (puis SQL)
    }

    resp = es.search(index="nisu_winkers", body=body)
    hits = resp.get("hits", {}).get("hits", [])

    winker_ids: List[int] = []
    for h in hits:
        try:
            winker_ids.append(int(h.get("_id")))
        except Exception:
            continue

    if not winker_ids:
        return []

    winkers_sql = fetch_winkers_by_ids(winker_ids)

    # préserver ordre ES
    by_id = {w["id"]: w for w in winkers_sql if "id" in w}
    ordered = [by_id[i] for i in winker_ids if i in by_id]

    safe_out: List[Dict[str, Any]] = []

    user_lat = float(user_geo["lat"])
    user_lon = float(user_geo["lon"])

    for w in ordered:
        lat = w.get("lat")
        lon = w.get("lon")

        distance_km = None
        try:
            if lat is not None and lon is not None:
                distance_km = round(
                    haversine_km(user_lat, user_lon, float(lat), float(lon)),
                    1,
                )
        except Exception:
            pass
        following_ids, follow_back_ids = fetch_follow_flags(user_id, winker_ids)


        safe_out.append({
            **w,
            "id": w.get("id"),
            "username": w.get("username"),
            "bio": w.get("bio"),
            "city": w.get("city"),
            "region": w.get("region"),
            "sexe": w.get("sexe"),
            "photoProfil": w.get("photoProfil"),
            "distance_km": distance_km,
            "isFollowing": int(w.get("id")) in following_ids,
            "isFollowBack": int(w.get("id")) in follow_back_ids,
        })

    return safe_out


