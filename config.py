"""
Configuration file for the central server
"""
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_TITLE: str = "Central Server API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = Field(default=8000, validation_alias=AliasChoices("API_PORT", "PORT"))
    
    # Streamlit Configuration
    STREAMLIT_PORT: int = 8501
    
    # CORS Settings
    CORS_ORIGINS: list[str] = ["*"]
    
    # Database (if needed)
    DATABASE_URL: Optional[str] = None
    
    # ML Model Settings
    MODEL_PATH: str = "models/"
    DEFAULT_MODEL: str = "default"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
