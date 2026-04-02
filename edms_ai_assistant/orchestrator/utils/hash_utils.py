# edms_ai_assistant\utils\hash_utils.py
import hashlib


def get_file_hash(file_path: str) -> str:
    """Генерирует SHA-256 хэш содержимого файла."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()
