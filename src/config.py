from typing import Optional
import boto3
from pydantic_settings import BaseSettings

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    YOUTUBE_API_KEY: Optional[str] = None
    AWS_REGION: str = "ap-northeast-2"
    PARAMETER_NAME_OPENAI: str = "/joing/ai/openai-key"
    PARAMETER_NAME_YOUTUBE: str = "/joing/ai/youtube-data-key"

    def get_parameter(self, parameter_name: str) -> str:
        try:
            ssm = boto3.client("ssm", region_name=self.AWS_REGION)
            response = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
            return response["Parameter"]["Value"]
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve '{parameter_name}': {e}")

    def setup_environment(self):
        os.environ["OPENAI_API_KEY"] = self.get_parameter(self.PARAMETER_NAME_OPENAI)
        self.YOUTUBE_API_KEY = self.get_parameter(self.PARAMETER_NAME_YOUTUBE)

# Initialize settings and set environment variables
settings = Settings()
settings.setup_environment()