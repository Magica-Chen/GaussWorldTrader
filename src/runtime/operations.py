"""Consistent SQLite backups and verified restore into a new offline destination."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _readonly_database(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{quote(str(path.resolve()))}?mode=ro", uri=True)


def _integrity(path: Path) -> int:
    with _readonly_database(path) as connection:
        results = [row[0] for row in connection.execute("PRAGMA integrity_check")]
        if results != ["ok"]:
            raise ValueError(f"SQLite integrity check failed: {results}")
        return connection.execute("PRAGMA user_version").fetchone()[0]


def backup_state(
    database_path: str | Path,
    destination: str | Path,
    *,
    evidence_directory: str | Path | None = None,
) -> dict:
    """Use SQLite's online backup API so committed WAL state is included."""
    source = Path(database_path).resolve()
    target = Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError("backup destination must be new")
    evidence = Path(evidence_directory).resolve() if evidence_directory is not None else None
    if evidence is not None:
        if not evidence.is_dir():
            raise FileNotFoundError(evidence)
        if target == evidence or evidence in target.parents:
            raise ValueError("backup destination must be outside the evidence directory")
    target.mkdir(parents=True, mode=0o700)
    backup_database = target / "session.db"
    try:
        with _readonly_database(source) as reader, sqlite3.connect(backup_database) as writer:
            reader.backup(writer)
        os.chmod(backup_database, 0o600)
        schema_version = _integrity(backup_database)
        files = {"session.db": _digest(backup_database)}
        if evidence is not None:
            for item in sorted(evidence.rglob("*")):
                if item.is_symlink():
                    raise ValueError("evidence backup refuses symbolic links")
                if not item.is_file():
                    continue
                relative = Path("evidence") / item.relative_to(evidence)
                copied = target / relative
                copied.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copyfile(item, copied)
                os.chmod(copied, 0o600)
                files[relative.as_posix()] = _digest(copied)
        manifest = {
            "schema_version": 1,
            "sqlite_user_version": schema_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
            "restore_requires_offline_runtime": True,
        }
        manifest_path = target / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.chmod(manifest_path, 0o600)
        return {"path": str(target), **manifest}
    except Exception:
        # Keep the incomplete artifact for diagnosis; it has no valid manifest.
        raise


def verify_backup(backup_directory: str | Path) -> dict:
    source = Path(backup_directory).resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or "session.db" not in manifest.get("files", {}):
        raise ValueError("unsupported or incomplete backup manifest")
    for name, expected_hash in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("backup manifest contains an unsafe relative path")
        item = source / relative
        if item.is_symlink() or source not in item.resolve().parents:
            raise ValueError("backup references a symbolic link or external file")
        if not item.is_file() or _digest(item) != expected_hash:
            raise ValueError(f"backup content hash mismatch: {name}")
    if _integrity(source / "session.db") != manifest["sqlite_user_version"]:
        raise ValueError("backup SQLite schema version mismatch")
    return manifest


def restore_state(
    backup_directory: str | Path,
    destination: str | Path,
    *,
    expected_schema_version: int | None = None,
) -> dict:
    """Restore to a new directory; starting a broker-connected service is separate."""
    source = Path(backup_directory).resolve()
    manifest = verify_backup(source)
    if (
        expected_schema_version is not None
        and manifest["sqlite_user_version"] != expected_schema_version
    ):
        raise ValueError("backup schema is incompatible with the requested deployment")
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError("restore destination must be new; active state is never overwritten")
    target.mkdir(parents=True, mode=0o700)
    for name in manifest["files"]:
        copied = target / name
        copied.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source / name, copied)
        os.chmod(copied, 0o600)
    _integrity(target / "session.db")
    return {
        "database_path": str(target / "session.db"),
        "evidence_directory": str(target / "evidence"),
        "files_restored": len(manifest["files"]),
        "broker_connected": False,
        "execution_armed": False,
    }
