from fastapi import FastAPI
from .core.es import init_indices
from .api.v1.api import api_router

app = FastAPI(
    title="NISU Recommendation Service",
    version="0.1.0",
)

@app.on_event("startup")
def startup():
    init_indices()

app.include_router(api_router, prefix="/api/v1")
