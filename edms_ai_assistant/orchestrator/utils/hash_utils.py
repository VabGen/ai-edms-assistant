# orchestrator/utils/hash_utils.py
"""Утилиты хэширования для кэш-ключей и идентификаторов файлов."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def sha256_file(file_path: str, chunk_size: int = 65536) -> str:
    """Асинхронно вычисляет SHA-256 файла по чанкам."""
    import aiofiles
    h = hashlib.sha256()
    async with aiofiles.open(file_path, "rb") as f:
        while chunk := await f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
