from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Config:
    exa_api_key: str = os.getenv("EXA_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    brave_api_key: str = os.getenv("BRAVE_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")


config = Config()
