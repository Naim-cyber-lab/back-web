from app.core.es import es_client
from app.core.config import INDEX_WINKERS
from app.schemas import WinkerIn
from typing import List


def index_winker(w: WinkerIn) -> None:
    doc = {
        "username": w.username,
        "email": w.email,
        "sexe": w.sexe,
        "age": w.age,
        "city": w.city,
        "region": w.region,
        "subregion": w.subregion,
        "pays": w.pays,
        "visible_tags": w.visible_tags,
        "meet_eligible": w.meet_eligible,
        "mails_eligible": w.mails_eligible,
    }

    if w.lat is not None and w.lon is not None:
        doc["latlon"] = {"lat": w.lat, "lon": w.lon}

    if w.preference_vector is not None:
        doc["preference_vector"] = w.preference_vector

    es.index(index=INDEX_WINKERS, id=w.id, document=doc)


def bulk_index_winkers(winkers: List[WinkerIn]) -> None:
    """
    Indexation bulk pour gagner du temps côté Airflow.
    """
    if not winkers:
        return

    actions = []
    for w in winkers:
        doc = {
            "_op_type": "index",
            "_index": INDEX_WINKERS,
            "_id": w.id,
            "username": w.username,
            "email": w.email,
            "sexe": w.sexe,
            "age": w.age,
            "city": w.city,
            "region": w.region,
            "subregion": w.subregion,
            "pays": w.pays,
            "visible_tags": w.visible_tags,
            "meet_eligible": w.meet_eligible,
            "mails_eligible": w.mails_eligible,
        }

        if w.lat is not None and w.lon is not None:
            doc["latlon"] = {"lat": w.lat, "lon": w.lon}

        if w.preference_vector is not None:
            doc["preference_vector"] = w.preference_vector

        actions.append(doc)

    # bulk helper
    from elasticsearch.helpers import bulk
    bulk(es_client, actions)
