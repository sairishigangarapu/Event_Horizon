import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_env: str = "demo"
    database_url: str = "sqlite:///./data/app.db"
    scenario_dir: str = "./data/demo"
    model_dir: str = "./models/production"
    central_source_mode: str = "replay"
    central_forecast_path: str = "./data/demo/central_forecast.csv"
    central_api_url: str = ""
    central_api_key: str = ""
    sim_seed: int = 42
    step_seconds: int = 5
    lookback_steps: int = 12  # 12 steps * 5s = 60s history window
    forecast_horizons_seconds: str = "30,60,120"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_base_url: str = "http://127.0.0.1:8000"
    dashboard_port: int = 8501
    auth_mode: str = "local_demo"
    # Strict serving: refuse Stream B forecasts unless every neural stage is trained and
    # validated (manifest training_status + metrics_summary). Off for the demo, which serves
    # untrained stages clearly labelled as experimental.
    require_validated_models: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # NOTE (F-021): 'ignore' silently drops unknown env vars (e.g. typo DATABASE_UL
        # instead of DATABASE_URL). If you suspect a misconfiguration, temporarily set
        # extra='forbid' locally to surface unexpected variables at startup.
        extra="ignore",
        protected_namespaces=("settings_",),  # avoids conflict with reserved model_ prefix
    )

    @property
    def horizons(self) -> List[int]:
        return [int(x.strip()) for x in self.forecast_horizons_seconds.split(",") if x.strip()]

settings = Settings()
