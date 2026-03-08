"""Validation tests for Phase 8: Project cleanup and archiving."""

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def test_legacy_files_archived():
    """Verify legacy files have been moved to archive/."""
    archive_path = PROJECT_ROOT / "archive" / "legacy_scripts_20260219"
    assert archive_path.exists(), "Archive directory should exist"

    # Check some key archived files
    archived_files = [
        "automl_benchmark.py",
        "create_bucket.py",
        "accident_model.pkl",
        "README.md.old",
        "ARCHITECTURE_REFONTE.md",
        "docker-compose.yml",
    ]

    for filename in archived_files:
        file_path = archive_path / filename
        assert file_path.exists(), f"{filename} should be archived"


def test_root_directory_clean():
    """Verify root directory only contains essential files."""
    root_files = [f.name for f in PROJECT_ROOT.iterdir() if f.is_file()]

    # Essential files that should remain

    # Files that should NOT be in root
    unwanted_patterns = [
        "accident_model.pkl",
        "all_models.pkl",
        "tmp.md",
        "TODO.md",
        ".env.old",
        "automl_benchmark.py",
        "README.md.old",
    ]

    for pattern in unwanted_patterns:
        assert pattern not in root_files, f"{pattern} should be archived, not in root"

    # Check that we have reasonable number of root files (not cluttered)
    assert len(root_files) < 25, f"Root has {len(root_files)} files - should be cleaner"


def test_temporary_directories_removed():
    """Verify temporary directories have been removed."""
    temp_dirs = [
        "build",
        "accidents.egg-info",
        "catboost_info",
    ]

    for dirname in temp_dirs:
        dir_path = PROJECT_ROOT / dirname
        assert not dir_path.exists(), f"{dirname} should be removed"


def test_gitignore_updated():
    """Verify .gitignore has been updated with proper patterns."""
    gitignore_path = PROJECT_ROOT / ".gitignore"
    assert gitignore_path.exists(), ".gitignore should exist"

    content = gitignore_path.read_text()

    # Check essential patterns
    essential_patterns = [
        "*.pkl",
        "data/",
        "*.csv",
        ".env",
        "__pycache__",
        ".venv",
        "archive/",
        "mlruns/",
        ".pytest_cache/",
    ]

    for pattern in essential_patterns:
        assert pattern in content, f"Pattern {pattern} should be in .gitignore"


def test_project_structure_intact():
    """Verify essential project directories still exist."""
    essential_dirs = [
        "src",
        "apps",
        "tests",
        "docs",
        "k8s",
        "infra",
        "pipeline",
        "data",
        "archive",
        "src",
    ]

    for dirname in essential_dirs:
        dir_path = PROJECT_ROOT / dirname
        assert dir_path.exists(), f"{dirname} directory should exist"


def test_documentation_complete():
    """Verify all documentation files are present."""
    doc_files = [
        "README.md",
        "docs/architecture.md",
        "docs/workflow.md",
        "docs/deployment.md",
    ]

    for doc_file in doc_files:
        file_path = PROJECT_ROOT / doc_file
        assert file_path.exists(), f"{doc_file} should exist"

        # Check files are not empty
        content = file_path.read_text()
        assert len(content) > 1000, f"{doc_file} should have substantial content"
