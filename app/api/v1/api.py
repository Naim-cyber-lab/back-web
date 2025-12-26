from fastapi import APIRouter
from app.api.v1.events import router
from app.api.v1.auth import router as auth_router

api_router = APIRouter()
api_router.include_router(router, tags=["events"])
api_router.include_router(auth_router, tags=["auth"])

