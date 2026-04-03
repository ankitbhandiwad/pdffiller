from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi.templating import Jinja2Templates


@dataclass(frozen=True)
class AppPaths:
    package_dir: Path
    data_dir: Path
    startup_dir: Path
    templates_dir: Path
    static_dir: Path


def build_paths() -> AppPaths:
    package_dir = Path(__file__).resolve().parent
    data_dir = package_dir / "data"
    startup_dir = data_dir / "startup_form"
    templates_dir = package_dir / "templates"
    static_dir = package_dir / "static"

    data_dir.mkdir(parents=True, exist_ok=True)
    startup_dir.mkdir(parents=True, exist_ok=True)

    return AppPaths(
        package_dir=package_dir,
        data_dir=data_dir,
        startup_dir=startup_dir,
        templates_dir=templates_dir,
        static_dir=static_dir,
    )


def build_templates(paths: AppPaths) -> Jinja2Templates:
    return Jinja2Templates(directory=str(paths.templates_dir))
