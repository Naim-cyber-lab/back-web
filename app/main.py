from fastapi import FastAPI
from .api.v1.api import api_router

app = FastAPI(
    title="NISU Web Service",
    version="0.1.0",
)

@app.on_event("startup")
def startup():
    pass

app.include_router(api_router, prefix="/api/v1")
