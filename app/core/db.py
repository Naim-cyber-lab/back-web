# app/core/db.py
import os
from dotenv import load_dotenv
import psycopg

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://myuser:mypassword@192.168.1.208:5432/mydatabase",
)

def get_conn():
    """
    Ouvre une connexion PostgreSQL.
    À utiliser avec `with get_conn() as conn:`
    """
    return psycopg.connect(DATABASE_URL)
