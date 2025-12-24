from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas import UserCreate, User
from app.repositories.users import create_user, get_user

router = APIRouter()


@router.post("/", response_model=User)
def create_user_endpoint(user_in: UserCreate):
    return create_user(user_in)


@router.get("/{user_id}", response_model=User)
def get_user_endpoint(user_id: str):
    user = get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
