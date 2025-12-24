from elasticsearch import Elasticsearch
from .config import (
    ELASTICSEARCH_URL,
    ELASTICSEARCH_USERNAME,
    ELASTICSEARCH_PASSWORD,
    INDEX_WINKERS,
    INDEX_EVENTS,
    INDEX_CONVERSATIONS,
)

es_client = Elasticsearch(
    ELASTICSEARCH_URL,
    basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
    verify_certs=False,  # à durcir en prod
)

WINKER_MAPPING = {
    "mappings": {
        "properties": {
            "username": {"type": "keyword"},
            "email": {"type": "keyword"},
            "sexe": {"type": "keyword"},
            "age": {"type": "integer"},
            "city": {"type": "keyword"},
            "region": {"type": "keyword"},
            "subregion": {"type": "keyword"},
            "pays": {"type": "keyword"},
            "latlon": {"type": "geo_point"},
            # préférences / tags visibles (Party, sport, voyage, etc.)
            "visible_tags": {"type": "keyword"},
            # vecteur de préférences (16 dimensions chez toi)
            "preference_vector": {"type": "dense_vector", "dims": 16},
            # flags utiles
            "meet_eligible": {"type": "boolean"},
            "mails_eligible": {"type": "boolean"},
        }
    }
}

EVENT_MAPPING = {
    "mappings": {
        "properties": {
            "titre": {"type": "text"},
            "bioEvent": {"type": "text"},
            "city": {"type": "keyword"},
            "region": {"type": "keyword"},
            "subregion": {"type": "keyword"},
            "pays": {"type": "keyword"},
            "codePostal": {"type": "keyword"},
            "latlon": {"type": "geo_point"},
            "dateEvent": {"type": "date"},
            "datePublication": {"type": "date"},
            "ageMinimum": {"type": "integer"},
            "ageMaximum": {"type": "integer"},
            "accessFille": {"type": "boolean"},
            "accessGarcon": {"type": "boolean"},
            "accessTous": {"type": "boolean"},
            "hastagEvents": {"type": "keyword"},   # hashtags splittés
            "meetEligible": {"type": "boolean"},
            "planTripElligible": {"type": "boolean"},
            "currentNbParticipants": {"type": "integer"},
            "maxNumberParticipant": {"type": "integer"},
            "isFull": {"type": "boolean"},
            "vectorPreferenceEvent": {"type": "dense_vector", "dims": 16},
        }
    }
}

CONVERSATION_MAPPING = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "description": {"type": "text"},
            "city": {"type": "keyword"},
            "region": {"type": "keyword"},
            "sexe": {"type": "keyword"},
            "age_min": {"type": "integer"},
            "age_max": {"type": "integer"},
            "is_private": {"type": "keyword"},
            "isOnline": {"type": "boolean"},
            "last_message_summary": {"type": "text"},
            "datePublication": {"type": "date"},
            "nb_waiting_count": {"type": "integer"},
        }
    }
}


def init_indices():
    if not es_client.indices.exists(index=INDEX_WINKERS):
        es_client.indices.create(index=INDEX_WINKERS, **WINKER_MAPPING)

    if not es_client.indices.exists(index=INDEX_EVENTS):
        es_client.indices.create(index=INDEX_EVENTS, **EVENT_MAPPING)

    if not es_client.indices.exists(index=INDEX_CONVERSATIONS):
        es_client.indices.create(index=INDEX_CONVERSATIONS, **CONVERSATION_MAPPING)
