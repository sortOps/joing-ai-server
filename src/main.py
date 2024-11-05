import os

from dotenv import load_dotenv
from fastapi import FastAPI

# Routers import

# config
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to Project Joing AI Api Server!"}