import boto3
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: Optional[str] = None
    YOUTUBE_DATA_API_KEY: Optional[str] = None

    AWS_REGION: str = "ap-northeast-2"
    PARAMETER_NAME_OPENAI: str = "/joing/ai/openai-key"
    PARAMETER_NAME_YOUTUBE: str = "/joing/ai/youtube-data-key"

    def __init__(self):
        super().__init__()
        if not self.OPENAI_API_KEY and not self.YOUTUBE_DATA_API_KEY:
            self.OPENAI_API_KEY = self.get_parameter_from_aws(
                self.PARAMETER_NAME_OPENAI)
            self.YOUTUBE_DATA_API_KEY = self.get_parameter_from_aws(
                self.PARAMETER_NAME_YOUTUBE)

    def get_parameter_from_aws(self, parameter_name) -> str:
        try:
            ssm = boto3.client('ssm', region_name=self.AWS_REGION)
            response = ssm.get_parameter(
                Name=self.parameter_name,
                WithDecryption=True
            )
            return response['Parameter']['Value']
        except Exception as e:
            raise RuntimeError(
                f"Failed to get OPENAI_API_KEY from AWS Parameter Store: {str(e)}")

settings = Settings()
