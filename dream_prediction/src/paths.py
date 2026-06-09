from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"


def ensure_project_dirs():
    for path in (DATA_DIR, MODELS_DIR, FIGURES_DIR, DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def data_path(filename):
    return DATA_DIR / filename


def model_path(filename):
    return MODELS_DIR / filename


def figure_path(filename):
    return FIGURES_DIR / filename
