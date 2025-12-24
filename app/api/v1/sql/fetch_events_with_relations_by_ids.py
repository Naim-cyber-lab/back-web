from app.core.db import *

FETCH_EVENTS_SQL = """
WITH input_ids AS (
    SELECT *
    FROM unnest(%s::int[]) WITH ORDINALITY AS t(id, ord)
)
SELECT
    e.id,
    e.titre,
    e."datePublication",
    e.adresse,
    e.city,
    e.region,
    e.subregion,
    e.pays,
    e."codePostal",
    e."bioEvent",
    e."bioEvent_fr",
    e."titre_fr",
    e.audio,
    e.lon,
    e.lat,
    e."accessComment",
    e."containReduction",
    e."prixInitial",
    e."prixReduction",
    e."firstPreference",
    e."firstPhotoLocalisation",
    e.remarque,
    e."priceEvent",
    e."horaireDebut",
    e."textForSearchingBar",
    e."vectorPreferenceEvent",
    e."hastagEvents",
    e."textReduction",
    e."currentLangue",
    e."importanteInformation",
    e."needReservation",
    e."linkReservation",
    e."isFull",
    e."detailsAddress",
    e."currentNbParticipants",
    e."maxNumberParticipant",
    e."nbComment",
    e.website,
    e."urlGoogleMapsAvis",
    e."urlAjoutGoogleMapsAvis",
    e."nb_conversations",
    e."nbStories",

    to_jsonb(cw.*) AS "creatorWinker",
    '[]'::jsonb AS participants,

    COALESCE(
        jsonb_agg(
            jsonb_build_object(
                'id', fe.id,
                'image', fe.image,
                'video', fe.video,
                'event_id', fe.event_id
            )
            ORDER BY fe.id ASC
        ) FILTER (WHERE fe.id IS NOT NULL),
        '[]'::jsonb
    ) AS "filesEvent"

FROM input_ids i
JOIN profil_event e ON e.id = i.id
LEFT JOIN profil_winker cw ON cw.id = e."creatorWinker_id"
LEFT JOIN profil_filesEvent fe ON fe.event_id = e.id

GROUP BY i.ord, e.id, cw.id
ORDER BY i.ord;
"""



def fetch_events_with_relations_by_ids(event_ids: list[int]) -> list[dict]:
    if not event_ids:
        return []

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(FETCH_EVENTS_SQL, (event_ids,))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

    return [dict(zip(columns, row)) for row in rows]

