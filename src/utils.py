import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import nltk
from nltk.tokenize import sent_tokenize
import yaml
from jinja2 import Environment, FileSystemLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
nltk.download("punkt")
nltk.download("punkt_tab")


def document_to_sentences(document: str) -> List[str]:
    sentences = sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


def normalize_text(
    text: str, min_line_length: int = 10, min_alpha_ratio: float = 0.3
) -> str:
    text = re.sub(r"[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        total_chars = len(stripped)
        alpha_chars = sum(1 for c in stripped if c.isalpha())
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        is_equation = bool(re.search(r"[=<>≤≥±×÷²∑∏∫]", stripped))

        if (
            (total_chars >= min_line_length and alpha_ratio >= min_alpha_ratio)
            or (alpha_ratio >= 0.7 and total_chars >= 5)
            or is_equation
        ):
            cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = "\n".join(document_to_sentences(text))
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def parse_document(document: str) -> Tuple[str, List[str]]:
    document = document.replace("\n", " ")
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
    backtranslated_dir = base / data_config["backtranslated_dir"]
    translation_dir = base / data_config["translation_dir"]
    baseline_dir = base / data_config["baseline_dir"]

    root_dir.mkdir(exist_ok=True)
    base.mkdir(exist_ok=True)
    unprocessed_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    backtranslated_dir.mkdir(exist_ok=True)
    translation_dir.mkdir(exist_ok=True)
    baseline_dir.mkdir(exist_ok=True)

    return {
        "base_dir": base,
        "unprocessed_dir": unprocessed_dir,
        "processed_dir": processed_dir,
        "backtranslated_dir": backtranslated_dir,
        "translation_dir": translation_dir,
        "baseline_dir": baseline_dir,
    }
