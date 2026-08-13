import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def load_environment() -> str | None:
    """Load GSO environment variables without overriding exported values."""
    configured_path = os.getenv("GSO_ENV_FILE")
    dotenv_path = (
        str(Path(configured_path).expanduser())
        if configured_path
        else find_dotenv(usecwd=True)
    )
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
        return dotenv_path
    return None


load_environment()


def hello() -> str:
    return "Hello from gso!"
