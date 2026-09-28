
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_api_files_exist():
    assert (ROOT / "app" / "main.py").is_file()
    assert (ROOT / "app" / "model_loader.py").is_file()


def test_training_files_exist():
    assert (ROOT / "training" / "config.py").is_file()
    assert (ROOT / "training" / "dataset.py").is_file()
    assert (ROOT / "training" / "lora.py").is_file()
    assert (ROOT / "training" / "train.py").is_file()


def test_docker_files_exist():
    assert (ROOT / "Dockerfile").is_file()
    assert (ROOT / "requirements-docker.txt").is_file()