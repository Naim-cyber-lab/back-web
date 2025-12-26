# app/api/v1/auth.py
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from psycopg import Connection

from app.core.db import get_conn

router = APIRouter(prefix="/auth", tags=["auth"])

# ===== Config
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
ACCESS_TTL_MIN = int(os.getenv("ACCESS_TTL_MIN", "60"))

USER_TABLE = os.getenv("DJANGO_USER_TABLE", "profil_winker")
USER_EMAIL_COL = os.getenv("DJANGO_USER_EMAIL_COL", "email")
USER_PWD_COL = os.getenv("DJANGO_USER_PWD_COL", "password")

pwd = CryptContext(
    schemes=[
        "django_pbkdf2_sha256",
        "django_pbkdf2_sha1",
        "django_argon2",
        "bcrypt",
    ],
    deprecated="auto",
)

def conn_dep() -> Connection:
    return get_conn()

# ===== Schemas
class SignupIn(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class AuthOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class MeOut(BaseModel):
    id: int
    email: EmailStr
    role: str
    name: Optional[str] = None

# ===== Helpers
def make_access_token(*, sub: str, user_id: int, role: str):
    exp = datetime.utcnow() + timedelta(minutes=ACCESS_TTL_MIN)
    payload = {"sub": sub, "uid": user_id, "role": role, "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def decode_token(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except Exception:
        raise HTTPException(status_code=401, detail="Token invalide")

# ===== Routes
@router.post("/signup", response_model=AuthOut)
def signup(body: SignupIn, conn: Connection = Depends(conn_dep)):
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Mot de passe trop court")

    password_hash = pwd.hash(body.password, scheme="django_pbkdf2_sha256")

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT id FROM {USER_TABLE} WHERE "{USER_EMAIL_COL}"=%(email)s""",
                {"email": body.email},
            )
            if cur.fetchone():
                raise HTTPException(status_code=409, detail="Email déjà utilisé")

            cur.execute(
                f"""
                INSERT INTO {USER_TABLE} ("{USER_EMAIL_COL}", "{USER_PWD_COL}")
                VALUES (%(email)s, %(pwd)s)
                RETURNING id
                """,
                {"email": body.email, "pwd": password_hash},
            )
            user_id = cur.fetchone()[0]

    token = make_access_token(sub=body.email, user_id=user_id, role="user")
    return AuthOut(access_token=token)

@router.post("/login", response_model=AuthOut)
def login(body: LoginIn, conn: Connection = Depends(conn_dep)):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, "{USER_EMAIL_COL}", "{USER_PWD_COL}"
            FROM {USER_TABLE}
            WHERE "{USER_EMAIL_COL}"=%(email)s
            """,
            {"email": body.email},
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Identifiants invalides")

    user_id, email, password_hash = row

    if not password_hash or not pwd.verify(body.password, password_hash):
        raise HTTPException(status_code=401, detail="Identifiants invalides")

    token = make_access_token(sub=email, user_id=user_id, role="user")
    return AuthOut(access_token=token)

@router.get("/me", response_model=MeOut)
def me(authorization: str | None = None, conn: Connection = Depends(conn_dep)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Non authentifié")

    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)

    uid = int(payload["uid"])
    role = payload.get("role", "user")

    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, "{USER_EMAIL_COL}" FROM {USER_TABLE} WHERE id=%(id)s""",
            {"id": uid},
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Utilisateur introuvable")

    user_id, email = row
    return MeOut(id=user_id, email=email, role=role, name=None)
