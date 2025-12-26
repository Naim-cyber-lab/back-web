from fastapi import APIRouter
from app.api.v1.events import router

api_router = APIRouter()
api_router.include_router(router, prefix="/events", tags=["events"])
