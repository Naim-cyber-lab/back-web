from fastapi import APIRouter, HTTPException
from typing import List
from ...schemas import EventCreate, Event
from ...repositories.events import create_event, get_event

router = APIRouter()


@router.post("/", response_model=Event)
def create_event_endpoint(event_in: EventCreate):
    return create_event(event_in)


@router.get("/{event_id}", response_model=Event)
def get_event_endpoint(event_id: str):
    event = get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event
