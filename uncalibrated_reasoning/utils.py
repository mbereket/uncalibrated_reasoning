from pathlib import Path

def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def get_cache_dir() -> Path:
    repo_root = get_repo_root()
    cache_path = repo_root / "data_cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path
