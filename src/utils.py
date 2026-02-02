import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from jinja2 import Environment, FileSystemLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def document_to_sentences(document: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$", document)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


def normalize_punctuation(text: str) -> str:
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.,!?;:])(?=[^\s])", r"\1 ", text)
    return text


def strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    return text


def parse_document(document: str) -> Tuple[str, List[str]]:
    document = document.replace("\n", " ")
    document = normalize_punctuation(document)
    sentences = document_to_sentences(document)
    return document, sentences


def get_prompt_environment(language_pair: str) -> Environment:
    prompts_dir = Path(__file__).resolve().parent / "prompts" / language_pair
    return Environment(loader=FileSystemLoader(prompts_dir))


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_paths(config: dict, language_pair: str):
    data_config = config.get("data", {})
    root_dir = PROJECT_ROOT / (data_config.get("root") or "data")

    base = root_dir / language_pair
    unprocessed_dir = base / data_config["unprocessed_dir"]
    processed_dir = base / data_config["processed_dir"]
    synth_source_dir = base / data_config["synth_source_dir"]
    translation_dir = base / data_config["translation_dir"]

    root_dir.mkdir(exist_ok=True)
    base.mkdir(exist_ok=True)
    unprocessed_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    synth_source_dir.mkdir(exist_ok=True)
    translation_dir.mkdir(exist_ok=True)

    return {
        "base_dir": base,
        "unprocessed_dir": unprocessed_dir,
        "processed_dir": processed_dir,
        "synth_source_dir": synth_source_dir,
        "translation_dir": translation_dir,
    }
