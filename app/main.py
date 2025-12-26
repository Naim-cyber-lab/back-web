from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.v1.api import api_router

app = FastAPI(title="NISU Web Service", version="0.1.0")

# ✅ CORS (autoriser ton front)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.nisu.fr",
        "https://nisu.fr",
        "http://localhost:5173",  # si tu dev en local
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")
