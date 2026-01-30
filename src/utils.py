import os
import re
from typing import Any, Dict, List, Tuple

import yaml
from jinja2 import Environment, FileSystemLoader


def document_to_sentences(document: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$", document)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


def normalize_punctuation(text: str) -> str:
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.,!?;:])(?=[^\s])", r"\1 ", text)
    return text


def parse_document(document: str) -> Tuple[str, List[str]]:
    document = document.replace("\n", " ")
    document = normalize_punctuation(document)
    sentences = document_to_sentences(document)
    return document, sentences


def get_prompt_environment(language_pair: str) -> Environment:
    prompts_dir = os.path.join(os.path.dirname(__file__), f"prompts/{language_pair}")
    return Environment(loader=FileSystemLoader(prompts_dir))


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
