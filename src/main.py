import os

from dotenv import load_dotenv
from fastapi import FastAPI

# Routers import
from proposal.router import router as proposal_router

# config
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# app init & router
app = FastAPI()
app.include_router(proposal_router)

@app.get("/")
def root():
    return {"message": "Welcome to Project Joing AI Api Server!"}