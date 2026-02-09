from src.config import Config
from src.models.integration import run_all


def test_imports() -> None:
    assert Config is not None
    assert run_all is not None
