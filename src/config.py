from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    YOUTUBE_DATA_API_KEY: str

    def __init__(self):
        super().__init__()
        if not self.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        if not self.YOUTUBE_DATA_API_KEY:
            raise RuntimeError("YOUTUBE_DATA_API_KEY environment variable is not set")

    model_config = {
        "extra": "allow",
        "validate_assignment": True
    }

settings = Settings()