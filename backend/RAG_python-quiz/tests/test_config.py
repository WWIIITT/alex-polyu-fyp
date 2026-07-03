from pathlib import Path

from app.config import Settings


def test_settings_env_file_is_project_local_absolute_path():
    env_file = Path(Settings.model_config["env_file"])

    assert env_file.is_absolute()
    assert env_file.name == ".env"
    assert env_file.parent == Path(__file__).resolve().parents[1]
