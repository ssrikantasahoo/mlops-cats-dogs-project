"""
Create a reproducible submission ZIP for Assignment 2 deliverables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_INCLUDE = [
    ".github",
    ".dvc",
    ".dvcignore",
    ".gitattributes",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.prod.yml",
    "kubernetes",
    "models",
    "prometheus.yml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "pyproject.toml",
    "pytest.ini",
    "setup.py",
    "src",
    "scripts",
    "tests",
    "README.md",
    "ASSIGNMENT_GUIDE.md",
    "EVIDENCE_INDEX.md",
    "SUBMISSION_CHECKLIST.md",
]


def iter_files(root: Path, relative_path: Path):
    target = root / relative_path
    if target.is_file():
        yield target
        return
    if target.is_dir():
        for file_path in target.rglob("*"):
            if file_path.is_file():
                yield file_path


def create_zip(project_root: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zf:
        for entry in DEFAULT_INCLUDE:
            rel = Path(entry)
            for file_path in iter_files(project_root, rel):
                arcname = file_path.relative_to(project_root)
                zf.write(file_path, arcname)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build assignment submission ZIP")
    parser.add_argument(
        "--output",
        default="submission/mlops_assignment2_submission.zip",
        help="Output ZIP path",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_zip = project_root / args.output
    create_zip(project_root, output_zip)
    print(f"Created submission bundle: {output_zip}")


if __name__ == "__main__":
    main()
