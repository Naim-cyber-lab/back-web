from ..core.es import es_client
from ..core.config import INDEX_EVENTS
from ..schemas import EventIn
from typing import List


def index_event(e: EventIn) -> None:
    doc = {
        "titre": e.titre,
        "bioEvent": e.bioEvent,
        "city": e.city,
        "region": e.region,
        "subregion": e.subregion,
        "pays": e.pays,
        "codePostal": e.codePostal,
        "dateEvent": e.dateEvent,
        "datePublication": e.datePublication,
        "ageMinimum": e.ageMinimum,
        "ageMaximum": e.ageMaximum,
        "accessFille": e.accessFille,
        "accessGarcon": e.accessGarcon,
        "accessTous": e.accessTous,
        "hastagEvents": e.hastagEvents,
        "meetEligible": e.meetEligible,
        "planTripElligible": e.planTripElligible,
        "currentNbParticipants": e.currentNbParticipants,
        "maxNumberParticipant": e.maxNumberParticipant,
        "isFull": e.isFull,
    }

    if e.lat is not None and e.lon is not None:
        doc["latlon"] = {"lat": e.lat, "lon": e.lon}

    if e.vectorPreferenceEvent is not None:
        doc["vectorPreferenceEvent"] = e.vectorPreferenceEvent

    es_client.index(index=INDEX_EVENTS, id=e.get("id"), document=doc)


def bulk_index_events(events: List[EventIn]) -> None:
    if not events:
        return

    from elasticsearch.helpers import bulk

    actions = []
    for e in events:
        doc = {
            "_op_type": "index",
            "_index": INDEX_EVENTS,
            "_id": e.id,
            "titre": e.titre,
            "bioEvent": e.bioEvent,
            "city": e.city,
            "region": e.region,
            "subregion": e.subregion,
            "pays": e.pays,
            "codePostal": e.codePostal,
            "dateEvent": e.dateEvent,
            "datePublication": e.datePublication,
            "ageMinimum": e.ageMinimum,
            "ageMaximum": e.ageMaximum,
            "accessFille": e.accessFille,
            "accessGarcon": e.accessGarcon,
            "accessTous": e.accessTous,
            "hastagEvents": e.hastagEvents,
            "meetEligible": e.meetEligible,
            "planTripElligible": e.planTripElligible,
            "currentNbParticipants": e.currentNbParticipants,
            "maxNumberParticipant": e.maxNumberParticipant,
            "isFull": e.isFull,
        }

        if e.lat is not None and e.lon is not None:
            doc["latlon"] = {"lat": e.lat, "lon": e.lon}

        if e.vectorPreferenceEvent is not None:
            doc["vectorPreferenceEvent"] = e.vectorPreferenceEvent

        actions.append(doc)

    bulk(es_client, actions)
