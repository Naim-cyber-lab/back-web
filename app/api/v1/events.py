# app/routers/events.py
import os
import uuid
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from psycopg import Connection

from app.core.db import get_conn

router = APIRouter(prefix="/events", tags=["events"])

# ⚠️ Adapte si tes tables ne s'appellent pas comme ça
EVENT_TABLE = os.getenv("DJANGO_EVENT_TABLE", "profil_event")
FILESEVENT_TABLE = os.getenv("DJANGO_FILESEVENT_TABLE", "profil_filesevent")

MEDIA_ROOT = os.getenv("MEDIA_ROOT", "/app/media")  # volume partagé avec Django
EVENT_UPLOAD_DIR = os.getenv("EVENT_UPLOAD_DIR", "events")  # sous-dossier dans MEDIA_ROOT

ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm"}
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def conn_dep() -> Connection:
    # get_conn() vient de ton app/core/db.py
    return get_conn()


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

    # écrit le fichier sur disque
    with open(abs_path, "wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return rel_path


class EventCreateResponse(BaseModel):
    event_id: int
    files_event_id: int
    video_path: str
    image_path: Optional[str] = None


@router.post("", response_model=EventCreateResponse, status_code=status.HTTP_201_CREATED)
def create_event(
    # ---- Champs Event (minimal + utile) ----
    creator_winker_id: int = Form(...),
    titre: str = Form(""),
    titre_fr: Optional[str] = Form(None),
    date_event: Optional[date] = Form(None),  # correspond à dateEvent
    adresse: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
    region: Optional[str] = Form(None),
    subregion: Optional[str] = Form(None),
    pays: Optional[str] = Form("France"),
    code_postal: Optional[str] = Form(None),
    bio_event: Optional[str] = Form(None),

    lon: Optional[float] = Form(None),
    lat: Optional[float] = Form(None),

    # ---- Fichiers ----
    video: UploadFile = File(...),           # ✅ obligatoire
    image: Optional[UploadFile] = File(None),# optionnel

    conn: Connection = Depends(conn_dep),
):
    # 1) Valide la vidéo
    if not video or not video.filename:
        raise HTTPException(status_code=400, detail="La vidéo est obligatoire.")

    # 2) Sauve les fichiers sur disque (chemins relatifs Django)
    video_path = _save_upload(video, subdir="videos", allowed_ext=ALLOWED_VIDEO_EXT)
    image_path = None
    if image and image.filename:
        image_path = _save_upload(image, subdir="images", allowed_ext=ALLOWED_IMAGE_EXT)

    # 3) Insert en transaction (Event puis FilesEvent)
    try:
        with conn:
            with conn.cursor() as cur:
                # NB: dans ton modèle Django, datePublication a un default côté Django,
                # mais en SQL direct ça ne s'applique PAS. Donc on la met nous-mêmes.
                today = date.today()

                cur.execute(
                    f"""
                    INSERT INTO {EVENT_TABLE}
                        ( "creatorWinker_id", "titre", "titre_fr",
                          "dateEvent", "datePublication",
                          "adresse", "city", "region", "subregion", "pays", "codePostal",
                          "bioEvent", "lon", "lat",
                          "currentNbParticipants", "maxNumberParticipant", "isFull",
                          "active", "nbAlreadyPublished"
                        )
                    VALUES
                        ( %(creator)s, %(titre)s, %(titre_fr)s,
                          %(date_event)s, %(date_pub)s,
                          %(adresse)s, %(city)s, %(region)s, %(subregion)s, %(pays)s, %(code_postal)s,
                          %(bio_event)s, %(lon)s, %(lat)s,
                          1, 10000000, false,
                          0, 0
                        )
                    RETURNING id
                    """,
                    {
                        "creator": creator_winker_id,
                        "titre": titre or "",
                        "titre_fr": titre_fr,
                        "date_event": date_event,
                        "date_pub": today,
                        "adresse": adresse,
                        "city": city,
                        "region": region,
                        "subregion": subregion,
                        "pays": pays,
                        "code_postal": code_postal,
                        "bio_event": bio_event,
                        "lon": lon,
                        "lat": lat,
                    },
                )
                event_id = cur.fetchone()[0]

                cur.execute(
                    f"""
                    INSERT INTO {FILESEVENT_TABLE}
                        ("event_id", "image", "video")
                    VALUES
                        (%(event_id)s, %(image)s, %(video)s)
                    RETURNING id
                    """,
                    {
                        "event_id": event_id,
                        "image": image_path,
                        "video": video_path,
                    },
                )
                files_event_id = cur.fetchone()[0]

        return EventCreateResponse(
            event_id=event_id,
            files_event_id=files_event_id,
            video_path=video_path,
            image_path=image_path,
        )

    except Exception as e:
        # Si erreur DB, on supprime les fichiers sauvegardés (rollback logique)
        # (évite d'accumuler des fichiers orphelins)
        try:
            abs_video = os.path.join(MEDIA_ROOT, video_path)
            if os.path.exists(abs_video):
                os.remove(abs_video)
        except Exception:
            pass

        if image_path:
            try:
                abs_img = os.path.join(MEDIA_ROOT, image_path)
                if os.path.exists(abs_img):
                    os.remove(abs_img)
            except Exception:
                pass

        raise HTTPException(status_code=500, detail=f"Erreur création event: {e}")
