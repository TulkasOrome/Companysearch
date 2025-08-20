# config/__init__.py
"""
Configuration management for the Company Search & Validation System
"""

import os
from pathlib import Path
from typing import Optional

# Try to load from .env file
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")
except ImportError:
    # dotenv not installed, use environment variables directly
    print("python-dotenv not installed, using system environment variables")


class Config:
    """Configuration manager"""

    # Azure OpenAI
    AZURE_OPENAI_KEY: str = os.getenv(
        "AZURE_OPENAI_KEY",
        "CUxPxhxqutsvRVHmGQcmH59oMim6mu55PjHTjSpM6y9UwIxwVZIuJQQJ99BFACL93NaXJ3w3AAABACOG3kI1"
    )
    AZURE_OPENAI_ENDPOINT: str = os.getenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://amex-openai-2025.openai.azure.com/"
    )
    AZURE_API_VERSION: str = os.getenv(
        "AZURE_API_VERSION",
        "2024-02-01"
    )

    # Serper
    SERPER_API_KEY: str = os.getenv(
        "SERPER_API_KEY",
        "99c44b79892f5f7499accf2d7c26d93313880937"
    )

    # GPT Deployments
    GPT_DEPLOYMENTS: list = [
        "gpt-4.1",
        "gpt-4.1-2",
        "gpt-4.1-3",
        "gpt-4.1-4",
        "gpt-4.1-5"
    ]

    # App Settings with safe parsing
    @classmethod
    def _safe_float(cls, key: str, default: str) -> float:
        """Safely parse float from environment variable"""
        try:
            value = os.getenv(key, default)
            return float(value)
        except (ValueError, TypeError):
            print(f"Warning: Invalid value for {key}='{value}', using default {default}")
            return float(default)

    @classmethod
    def _safe_int(cls, key: str, default: str) -> int:
        """Safely parse int from environment variable"""
        try:
            value = os.getenv(key, default)
            return int(value)
        except (ValueError, TypeError):
            print(f"Warning: Invalid value for {key}='{value}', using default {default}")
            return int(default)

    # App Settings using safe parsing
    MAX_VALIDATION_COST: float = None
    MAX_COST_PER_COMPANY: float = None
    CACHE_DURATION_HOURS: int = None
    MAX_PARALLEL_QUERIES: int = None
    SESSION_DIR: str = None
    SESSION_CLEANUP_DAYS: int = None

    @classmethod
    def initialize(cls):
        """Initialize configuration values with safe parsing"""
        cls.MAX_VALIDATION_COST = cls._safe_float("MAX_VALIDATION_COST", "10.0")
        cls.MAX_COST_PER_COMPANY = cls._safe_float("MAX_COST_PER_COMPANY", "0.01")
        cls.CACHE_DURATION_HOURS = cls._safe_int("CACHE_DURATION_HOURS", "24")
        cls.MAX_PARALLEL_QUERIES = cls._safe_int("MAX_PARALLEL_QUERIES", "5")
        cls.SESSION_DIR = os.getenv("SESSION_DIR", "sessions")
        cls.SESSION_CLEANUP_DAYS = cls._safe_int("SESSION_CLEANUP_DAYS", "7")

    @classmethod
    def get(cls, key: str, default: Optional[any] = None) -> any:
        """Get configuration value"""
        return getattr(cls, key, default)

    @classmethod
    def set(cls, key: str, value: any) -> None:
        """Set configuration value"""
        setattr(cls, key, value)

    @classmethod
    def to_dict(cls) -> dict:
        """Export configuration as dictionary"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value) and value is not None
        }

    @classmethod
    def debug_env(cls):
        """Debug environment variables"""
        print("\nEnvironment Variables Debug:")
        env_vars = [
            "AZURE_OPENAI_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "SERPER_API_KEY",
            "MAX_VALIDATION_COST",
            "MAX_COST_PER_COMPANY",
            "CACHE_DURATION_HOURS",
            "MAX_PARALLEL_QUERIES",
            "SESSION_DIR",
            "SESSION_CLEANUP_DAYS"
        ]

        for var in env_vars:
            value = os.getenv(var)
            if value:
                if "KEY" in var:
                    # Mask API keys
                    display_value = f"***{value[-4:]}" if len(value) > 4 else "***"
                else:
                    display_value = value
                print(f"  {var}: {display_value}")
            else:
                print(f"  {var}: Not set")


# Initialize configuration
Config.initialize()

# Export config instance
config = Config()

__all__ = [
    'Config',
    'config'
]