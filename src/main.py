from config import settings
import openai

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette import status

# Routers import
from proposal.router import router as proposal_router
from channel.router import router as channel_router
from rec_system.router import router as rec_router

# config
from .config import settings

# app init & router
app = FastAPI()
app.include_router(channel_router)
app.include_router(proposal_router)
# app.include_router(rec_router)


@app.get("/")
def root():
    return {"message": "Welcome to Project Joing AI Api Server!"}


@app.get("/ready")
def health_check():
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ok", "message": "Service is ready"}
    )
