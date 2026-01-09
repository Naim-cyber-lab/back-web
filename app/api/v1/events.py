# app/routers/events.py
import os
import uuid
from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier

from app.core.db import get_conn

router = APIRouter(prefix="/events", tags=["events"])

# Tables Django (override via env si besoin)
EVENT_TABLE = os.getenv("DJANGO_EVENT_TABLE", "profil_event")
FILESEVENT_TABLE = os.getenv("DJANGO_FILESEVENT_TABLE", "profil_filesevent")

MEDIA_ROOT = os.getenv("MEDIA_ROOT", "/app/media")
EVENT_UPLOAD_DIR = os.getenv("EVENT_UPLOAD_DIR", "events")

ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm"}
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# -------------------------
# DB dependency
# -------------------------
def conn_dep() -> Connection:
    return get_conn()

# -------------------------
# Helpers upload
# -------------------------
def _safe_ext(filename: str) -> str:
    _, ext = os.path.splitext(filename or "")
    return ext.lower().strip()

def _save_upload(upload: UploadFile, subdir: str, allowed_ext: set[str]) -> str:
    ext = _safe_ext(upload.filename)
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Extension non autorisée: {ext}. Autorisées: {sorted(allowed_ext)}",
        )

    rel_dir = os.path.join(EVENT_UPLOAD_DIR, subdir)
    abs_dir = os.path.join(MEDIA_ROOT, rel_dir)
    os.makedirs(abs_dir, exist_ok=True)

    filename = f"{uuid.uuid4().hex}{ext}"
    abs_path = os.path.join(abs_dir, filename)
    rel_path = os.path.join(rel_dir, filename).replace("\\", "/")

    with open(abs_path, "wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return rel_path

def _delete_file_quiet(rel_path: Optional[str]) -> None:
    if not rel_path:
        return
    try:
        abs_path = os.path.join(MEDIA_ROOT, rel_path)
        if os.path.exists(abs_path):
            os.remove(abs_path)
    except Exception:
        pass

# -------------------------
# Models
# -------------------------
class EventPublic(BaseModel):
    id: int

    titre: Optional[str] = None
    titre_fr: Optional[str] = None

    adresse: Optional[str] = None
    city: Optional[str] = None
    codePostal: Optional[str] = None

    region: Optional[str] = None
    subregion: Optional[str] = None
    pays: Optional[str] = None

    lat: Optional[float] = None
    lon: Optional[float] = None

    bioEvent: Optional[str] = None
    website: Optional[str] = None

    # champs socials (si tu les as en DB; sinon ils restent None)
    youtube_short: Optional[str] = None
    youtube_query: Optional[str] = None
    tiktok_video: Optional[str] = None
    tiktok_query: Optional[str] = None
    insta_video: Optional[str] = None
    insta_query: Optional[str] = None

    price: Optional[str] = None

    # confirmé (on mappe sur validated_from_web si la colonne existe)
    confirmed: bool = False

    # media Django (profil_filesevent)
    image: Optional[str] = None
    video: Optional[str] = None

class EventListResponse(BaseModel):
    items: list[EventPublic]
    total: int
    limit: int
    offset: int

class EventPatch(BaseModel):
    # édition depuis le front
    titre: Optional[str] = None
    bioEvent: Optional[str] = None
    adresse: Optional[str] = None
    city: Optional[str] = None
    codePostal: Optional[str] = None
    region: Optional[str] = None
    subregion: Optional[str] = None
    pays: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    website: Optional[str] = None

    youtube_short: Optional[str] = None
    youtube_query: Optional[str] = None
    tiktok_video: Optional[str] = None
    tiktok_query: Optional[str] = None
    insta_video: Optional[str] = None
    insta_query: Optional[str] = None

    price: Optional[str] = None

class ConfirmBody(BaseModel):
    confirmed: bool

class EventCreateResponse(BaseModel):
    event_id: int
    files_event_id: int
    video_path: str
    image_path: Optional[str] = None

# -------------------------
# Introspection util
# -------------------------
def _get_columns(conn: Connection) -> set[str]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
            """,
            (EVENT_TABLE,),
        )
        return {r["column_name"] for r in cur.fetchall()}

def _col_exists(cols: set[str], col: str) -> bool:
    return col in cols

# -------------------------
# Create (upload) - conservé + durci
# -------------------------
@router.post("", response_model=EventCreateResponse, status_code=status.HTTP_201_CREATED)
def create_event(
    creator_winker_id: int = Form(...),
    titre: str = Form(""),
    titre_fr: Optional[str] = Form(None),
    date_event: Optional[date] = Form(None),
    adresse: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
    region: Optional[str] = Form(None),
    subregion: Optional[str] = Form(None),
    pays: Optional[str] = Form("France"),
    code_postal: Optional[str] = Form(None),
    bio_event: Optional[str] = Form(None),
    lon: Optional[float] = Form(None),
    lat: Optional[float] = Form(None),
    video: UploadFile = File(...),
    image: Optional[UploadFile] = File(None),
    conn: Connection = Depends(conn_dep),
):
    if not video or not video.filename:
        raise HTTPException(status_code=400, detail="La vidéo est obligatoire.")

    video_path = _save_upload(video, subdir="videos", allowed_ext=ALLOWED_VIDEO_EXT)
    image_path = None
    if image and image.filename:
        image_path = _save_upload(image, subdir="images", allowed_ext=ALLOWED_IMAGE_EXT)

    try:
        cols = _get_columns(conn)

        # not-null defaults
        nbStories_default = 0
        nbAlreadyPublished_default = 0
        active_default = 0
        currentNbParticipants_default = 1
        maxNumberParticipant_default = 10000000
        isFull_default = False

        validated_default = False

        today = date.today()

        with conn:
            with conn.cursor() as cur:
                # Build insert safely depending on real columns present
                insert_cols = [
                    "creatorWinker_id", "titre", "titre_fr",
                    "dateEvent", "datePublication",
                    "adresse", "city", "region", "subregion", "pays", "codePostal",
                    "bioEvent", "lon", "lat",
                ]

                values = [
                    creator_winker_id, titre or "", titre_fr,
                    date_event, today,
                    adresse, city, region, subregion, pays, code_postal,
                    bio_event, lon, lat,
                ]

                # add default columns if they exist
                def add_if_exists(c: str, v: Any):
                    nonlocal insert_cols, values
                    if _col_exists(cols, c):
                        insert_cols.append(c)
                        values.append(v)

                add_if_exists("currentNbParticipants", currentNbParticipants_default)
                add_if_exists("maxNumberParticipant", maxNumberParticipant_default)
                add_if_exists("isFull", isFull_default)
                add_if_exists("active", active_default)
                add_if_exists("nbAlreadyPublished", nbAlreadyPublished_default)
                add_if_exists("nbStories", nbStories_default)
                add_if_exists("validated_from_web", validated_default)

                q = SQL("INSERT INTO {} ({}) VALUES ({}) RETURNING id").format(
                    Identifier(EVENT_TABLE),
                    SQL(", ").join(Identifier(c) for c in insert_cols),
                    SQL(", ").join(SQL("%s") for _ in insert_cols),
                )
                cur.execute(q, values)
                event_id = cur.fetchone()[0]

                q2 = SQL('INSERT INTO {} ("event_id","image","video") VALUES (%s,%s,%s) RETURNING id').format(
                    Identifier(FILESEVENT_TABLE)
                )
                cur.execute(q2, (event_id, image_path, video_path))
                files_event_id = cur.fetchone()[0]

        return EventCreateResponse(
            event_id=event_id,
            files_event_id=files_event_id,
            video_path=video_path,
            image_path=image_path,
        )

    except Exception as e:
        _delete_file_quiet(video_path)
        _delete_file_quiet(image_path)
        raise HTTPException(status_code=500, detail=f"Erreur création event: {e}")

# -------------------------
# Read - list
# -------------------------
@router.get("", response_model=EventListResponse)
def list_events(
    limit: int = 200,
    offset: int = 0,
    q: Optional[str] = None,
    conn: Connection = Depends(conn_dep),
):
    cols = _get_columns(conn)

    # map "confirmed" => validated_from_web if exists else active
    confirmed_expr = (
        SQL("COALESCE(e.validated_from_web, false)") if _col_exists(cols, "validated_from_web")
        else SQL("COALESCE(e.active, 0) = 1")
    )

    # optional columns
    def sel(colname: str) -> SQL:
        return Identifier(colname) if _col_exists(cols, colname) else SQL("NULL")

    where = SQL("TRUE")
    params: list[Any] = []

    if q and q.strip():
        where = SQL("(e.titre ILIKE %s OR e.adresse ILIKE %s OR e.city ILIKE %s)")
        like = f"%{q.strip()}%"
        params.extend([like, like, like])

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            SQL("SELECT COUNT(*) AS n FROM {} e WHERE ").format(Identifier(EVENT_TABLE)) + where,
            params,
        )
        total = int(cur.fetchone()["n"])

        query = (
            SQL(
                """
                SELECT
                  e.id,
                  e.titre, e.titre_fr,
                  e.adresse, e.city, e."codePostal" as "codePostal",
                  e.region, e.subregion, e.pays,
                  e.lat, e.lon,
                  e."bioEvent" as "bioEvent",
                  e.website,
                  {youtube_short} as youtube_short,
                  {youtube_query} as youtube_query,
                  {tiktok_video} as tiktok_video,
                  {tiktok_query} as tiktok_query,
                  {insta_video} as insta_video,
                  {insta_query} as insta_query,
                  {price} as price,
                  {confirmed} as confirmed,
                  f.image as image,
                  f.video as video
                FROM {event_table} e
                LEFT JOIN {files_table} f ON f."event_id" = e.id
                WHERE
                """
            )
            .format(
                event_table=Identifier(EVENT_TABLE),
                files_table=Identifier(FILESEVENT_TABLE),
                confirmed=confirmed_expr,
                youtube_short=sel("youtube_short"),
                youtube_query=sel("youtube_query"),
                tiktok_video=sel("tiktok_video"),
                tiktok_query=sel("tiktok_query"),
                insta_video=sel("insta_video"),
                insta_query=sel("insta_query"),
                price=sel("price"),
            )
            + where
            + SQL(" ORDER BY e.id DESC LIMIT %s OFFSET %s")
        )

        cur.execute(query, params + [limit, offset])
        rows = cur.fetchall()

    items = []
    for r in rows:
        items.append(
            EventPublic(
                id=r["id"],
                titre=r.get("titre"),
                titre_fr=r.get("titre_fr"),
                adresse=r.get("adresse"),
                city=r.get("city"),
                codePostal=r.get("codePostal"),
                region=r.get("region"),
                subregion=r.get("subregion"),
                pays=r.get("pays"),
                lat=r.get("lat"),
                lon=r.get("lon"),
                bioEvent=r.get("bioEvent"),
                website=r.get("website"),
                youtube_short=r.get("youtube_short"),
                youtube_query=r.get("youtube_query"),
                tiktok_video=r.get("tiktok_video"),
                tiktok_query=r.get("tiktok_query"),
                insta_video=r.get("insta_video"),
                insta_query=r.get("insta_query"),
                price=r.get("price"),
                confirmed=bool(r.get("confirmed") or False),
                image=r.get("image"),
                video=r.get("video"),
            )
        )

    return EventListResponse(items=items, total=total, limit=limit, offset=offset)

# -------------------------
# Read - single
# -------------------------
@router.get("/{event_id}", response_model=EventPublic)
def get_event(event_id: int, conn: Connection = Depends(conn_dep)):
    data = list_events(limit=1, offset=0, q=None, conn=conn)  # not ideal but OK
    # better: direct query, but we keep simple
    # We'll fetch by id directly:
    cols = _get_columns(conn)
    confirmed_expr = (
        SQL("COALESCE(e.validated_from_web, false)") if _col_exists(cols, "validated_from_web")
        else SQL("COALESCE(e.active, 0) = 1")
    )

    def sel(colname: str) -> SQL:
        return Identifier(colname) if _col_exists(cols, colname) else SQL("NULL")

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            SQL(
                """
                SELECT
                  e.id,
                  e.titre, e.titre_fr,
                  e.adresse, e.city, e."codePostal" as "codePostal",
                  e.region, e.subregion, e.pays,
                  e.lat, e.lon,
                  e."bioEvent" as "bioEvent",
                  e.website,
                  {youtube_short} as youtube_short,
                  {youtube_query} as youtube_query,
                  {tiktok_video} as tiktok_video,
                  {tiktok_query} as tiktok_query,
                  {insta_video} as insta_video,
                  {insta_query} as insta_query,
                  {price} as price,
                  {confirmed} as confirmed,
                  f.image as image,
                  f.video as video
                FROM {event_table} e
                LEFT JOIN {files_table} f ON f."event_id" = e.id
                WHERE e.id = %s
                """
            ).format(
                event_table=Identifier(EVENT_TABLE),
                files_table=Identifier(FILESEVENT_TABLE),
                confirmed=confirmed_expr,
                youtube_short=sel("youtube_short"),
                youtube_query=sel("youtube_query"),
                tiktok_video=sel("tiktok_video"),
                tiktok_query=sel("tiktok_query"),
                insta_video=sel("insta_video"),
                insta_query=sel("insta_query"),
                price=sel("price"),
            ),
            (event_id,),
        )
        r = cur.fetchone()

    if not r:
        raise HTTPException(status_code=404, detail="Event introuvable")

    return EventPublic(
        id=r["id"],
        titre=r.get("titre"),
        titre_fr=r.get("titre_fr"),
        adresse=r.get("adresse"),
        city=r.get("city"),
        codePostal=r.get("codePostal"),
        region=r.get("region"),
        subregion=r.get("subregion"),
        pays=r.get("pays"),
        lat=r.get("lat"),
        lon=r.get("lon"),
        bioEvent=r.get("bioEvent"),
        website=r.get("website"),
        youtube_short=r.get("youtube_short"),
        youtube_query=r.get("youtube_query"),
        tiktok_video=r.get("tiktok_video"),
        tiktok_query=r.get("tiktok_query"),
        insta_video=r.get("insta_video"),
        insta_query=r.get("insta_query"),
        price=r.get("price"),
        confirmed=bool(r.get("confirmed") or False),
        image=r.get("image"),
        video=r.get("video"),
    )

# -------------------------
# Patch - update fields
# -------------------------
@router.patch("/{event_id}", response_model=EventPublic)
def patch_event(event_id: int, patch: EventPatch, conn: Connection = Depends(conn_dep)):
    cols = _get_columns(conn)

    # allowlist -> map api fields to db columns (exact names)
    mapping: dict[str, str] = {
        "titre": "titre",
        "bioEvent": "bioEvent",
        "adresse": "adresse",
        "city": "city",
        "codePostal": "codePostal",
        "region": "region",
        "subregion": "subregion",
        "pays": "pays",
        "lat": "lat",
        "lon": "lon",
        "website": "website",
        "youtube_short": "youtube_short",
        "youtube_query": "youtube_query",
        "tiktok_video": "tiktok_video",
        "tiktok_query": "tiktok_query",
        "insta_video": "insta_video",
        "insta_query": "insta_query",
        "price": "price",
    }

    data = patch.model_dump(exclude_unset=True)
    if not data:
        return get_event(event_id, conn)

    sets: list[SQL] = []
    values: list[Any] = []

    for api_key, value in data.items():
        db_col = mapping.get(api_key)
        if not db_col:
            continue
        # only update if column exists
        if db_col not in cols and db_col not in {"bioEvent", "codePostal"}:
            continue
        # special quoted columns in Django schema
        if db_col in {"bioEvent", "codePostal"}:
            sets.append(SQL('"{}" = %s').format(SQL(db_col)))
        else:
            sets.append(SQL("{} = %s").format(Identifier(db_col)))
        values.append(value)

    if not sets:
        return get_event(event_id, conn)

    q = SQL("UPDATE {} SET ").format(Identifier(EVENT_TABLE)) + SQL(", ").join(sets) + SQL(" WHERE id = %s")
    values.append(event_id)

    with conn:
        with conn.cursor() as cur:
            cur.execute(q, values)

    return get_event(event_id, conn)

# -------------------------
# Confirm endpoint
# -------------------------
@router.post("/{event_id}/confirm", response_model=EventPublic)
def confirm_event(event_id: int, body: ConfirmBody, conn: Connection = Depends(conn_dep)):
    cols = _get_columns(conn)

    # Prefer validated_from_web if present, else active
    if _col_exists(cols, "validated_from_web"):
        q = SQL("UPDATE {} SET validated_from_web = %s WHERE id = %s").format(Identifier(EVENT_TABLE))
        vals = (body.confirmed, event_id)
    elif _col_exists(cols, "active"):
        q = SQL("UPDATE {} SET active = %s WHERE id = %s").format(Identifier(EVENT_TABLE))
        vals = (1 if body.confirmed else 0, event_id)
    else:
        raise HTTPException(status_code=400, detail="Pas de champ 'validated_from_web' ni 'active' dans la table event")

    with conn:
        with conn.cursor() as cur:
            cur.execute(q, vals)

    return get_event(event_id, conn)
