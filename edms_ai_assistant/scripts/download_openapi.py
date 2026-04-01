# edms_ai_assistant\scripts\download_openapi.py
import json
import logging
import re
import subprocess
import sys

import httpx

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def download_openapi_spec(url: str, output_file: str) -> bool:
    """
    Загружает спецификацию OpenAPI с заданного URL и сохраняет ее в файл.
    """
    logger.info(f"Downloading OpenAPI spec from {url}...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()

            if "application/json" in response.headers.get("content-type", ""):
                spec_data = response.json()
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(spec_data, f, ensure_ascii=False, indent=2)
                logger.info(
                    f"OpenAPI spec successfully downloaded and saved to {output_file}"
                )
                return True
            else:
                logger.error(
                    f"Error: Response is not JSON. Content-Type: {response.headers.get('content-type')}"
                )
                return False
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
            return False
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {e}")
            return False
        except Exception as e:
            logger.error(f"An error occurred during download: {e}")
            return False


def run_datamodel_codegen(input_file: str, output_file: str) -> bool:
    """
    Запускает datamodel-codegen для генерации Pydantic-моделей из OpenAPI-спецификации.
    """
    logger.info(f"Running datamodel-codegen on {input_file}...")
    cmd = [
        sys.executable,
        "-m",
        "datamodel_code_generator",
        "--input",
        input_file,
        "--output",
        output_file,
        "--encoding",
        "utf-8",
        "--use-unique-items-as-set",
        "--target-python-version",
        "3.13",
        "--use-annotated",
        "--use-union-operator",
        "--reuse-models",
        "--use-standard-collections",
        "--use-schema-description",
        "--collapse-root-models",
    ]

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, encoding="utf-8"
        )
        logger.info("datamodel-code-generator executed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"datamodel-code-generator failed with return code {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(
            "Error: 'datamodel-code-generator' is not installed in the current environment."
        )
        return False
    except Exception as e:
        logger.error(f"An error occurred while running datamodel-code-generator: {e}")
        return False


RegexPatch = tuple[str, str]

POST_GENERATION_PATCHES: list[RegexPatch] = [
    (r",\s*unique_items=True(?=\s*(?:,|\)))", ""),
    (r"unique_items=True\s*,?", ""),
    (r",\s*,", ","),
    (r"Field\(\s*,", "Field("),
    (r"(Annotated\[)list\[UUID\](\s*\| None,\s*Field\(.*?)\]", r"\1set[UUID]\2]"),
    (
        r"list\[UUID\] \| None = Field\((.*?)\)",
        lambda m: f"set[UUID] | None = Field({m.group(1)})",
    ),
    (
        r"class JsonNode\(BaseModel\):\s*\n\s*__root__:\s*Any",
        "class JsonNode(RootModel[Any]):\n    pass",
    ),
]


def apply_regex_patches(content: str, patches: list[RegexPatch]) -> str:
    """Применяет список регулярных выражений к содержимому файла."""
    for pattern, replacement in patches:
        if isinstance(replacement, str):
            content = re.sub(pattern, replacement, content)
        else:
            content = re.sub(pattern, replacement, content)
    return content


def fix_generated_file(file_path: str):
    """
    Применяет необходимые исправления к сгенерированному файлу Pydantic-моделей.
    """
    logger.info(f"Applying fixes to {file_path}...")
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    pydantic_import_pattern = r"from pydantic import [^\n]+\n"
    pydantic_import_matches = re.findall(pydantic_import_pattern, content)

    needed_pydantic_items = set()

    for match in pydantic_import_matches:
        content = content.replace(match, "", 1)

        items_str = match.replace("from pydantic import ", "").replace("\n", "")
        items = [item.strip() for item in items_str.split(",")]
        for item in items:
            item_clean = item.strip()
            if item_clean:
                needed_pydantic_items.add(item_clean)

    pydantic_import_line = ""
    if needed_pydantic_items:
        items_list = sorted(list(needed_pydantic_items))
        pydantic_import_line = f"from pydantic import {', '.join(items_list)}\n"

    future_import_pattern = r"(from __future__ import [^\n]+\n(?:\s*#[^\n]*\n)*)"
    generated_comment_pattern = r"#[^\n]*generated[^\n]*\n(?:#[^\n]*\n)*"

    future_match = re.search(future_import_pattern, content)
    future_block = future_match.group(1) if future_match else ""
    if future_match:
        content = content.replace(future_block, "", 1)

    generated_comment_match = re.search(generated_comment_pattern, content)
    generated_comment_block = (
        generated_comment_match.group(0) if generated_comment_match else ""
    )
    if generated_comment_match:
        content = content.replace(generated_comment_block, "", 1)

    new_start = future_block + generated_comment_block + pydantic_import_line
    content = new_start + content

    content = apply_regex_patches(content, POST_GENERATION_PATCHES)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"All fixes applied to {file_path}.")


async def main():
    """
    Основная функция, оркестрирующая процесс генерации DTO.
    """

    def check_datamodel_codegen():
        try:
            result = subprocess.run(
                [sys.executable, "-m", "datamodel_code_generator", "--version"],
                capture_output=True,
                text=True,
            )
            logger.info(f"datamodel-codegen version: {result.stdout.strip()}")
        except Exception:
            logger.error(
                "datamodel-codegen not found. Install with: uv pip install datamodel-code-generator"
            )
            sys.exit(1)

    openapi_url: str = "http://127.0.0.1:8098/public-resources/openapi"
    spec_file: str = "../../openapi_spec.json"
    dto_file: str = "../generated/resources_openapi.py"

    check_datamodel_codegen()

    download_success = await download_openapi_spec(openapi_url, spec_file)
    if not download_success:
        logger.error("Download failed. Stopping process.")
        return

    generation_success = run_datamodel_codegen(spec_file, dto_file)
    if not generation_success:
        logger.error("Model generation failed.")
        sys.exit(1)

    fix_generated_file(dto_file)

    logger.info(
        "OpenAPI spec downloaded, Pydantic models generated, and fixes applied successfully! ✅"
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
