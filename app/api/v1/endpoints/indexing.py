from typing import List
from fastapi import APIRouter
from app.schemas import WinkerIn, EventIn
from app.repositories.winkers import index_winker, bulk_index_winkers
from app.repositories.events import index_event, bulk_index_events

router = APIRouter()


# --------- WINKERS ---------

@router.post("/winkers", tags=["indexing"])
def index_single_winker(winker: WinkerIn):
    """
    Indexe / met à jour un seul Winker dans Elasticsearch.
    Idéal pour un job Airflow incrémental (un user).
    """
    index_winker(winker)
    return {"status": "ok", "indexed_id": winker.id}


@router.post("/winkers/bulk", tags=["indexing"])
def index_winkers_bulk_endpoint(winkers: List[WinkerIn]):
    """
    Indexation bulk de plusieurs Winkers en une seule requête.
    Idéal pour un DAG Airflow qui fait un batch.
    """
    bulk_index_winkers(winkers)
    return {"status": "ok", "count": len(winkers)}


# --------- EVENTS ---------

@router.post("/events", tags=["indexing"])
def index_single_event(event: EventIn):
    """
    Indexe / met à jour un seul Event dans Elasticsearch.
    """
    index_event(event)
    return {"status": "ok", "indexed_id": event.id}


@router.post("/events/bulk", tags=["indexing"])
def index_events_bulk_endpoint(events: List[EventIn]):
    """
    Indexation bulk de plusieurs Events.
    """
    bulk_index_events(events)
    return {"status": "ok", "count": len(events)}
