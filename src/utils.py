import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import nltk
from nltk.tokenize import sent_tokenize
import yaml
from jinja2 import Environment, FileSystemLoader
from lingua import LanguageDetectorBuilder

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_punkt():
    # Only download punkt resources if missing to avoid noisy import logs.
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_punkt()
detector = LanguageDetectorBuilder.from_all_languages().build()


def detect_language(text: str):
    result = detector.detect_language_of(text)
    return result.iso_code_639_1.name.lower() if result else None


def document_to_sentences(document: str) -> List[str]:
    sentences = sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


def normalize_text(
    text: str,
    ignore_lang: str = None,
    min_line_length: int = 10,
    min_alpha_ratio: float = 0.3,
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
    if ignore_lang:
        sents = document_to_sentences(text)
        sents = [sent for sent in sents if detect_language(sent) != ignore_lang]
        text = "\n".join(sents)
    else:
        sents = document_to_sentences(text)
        text = "\n".join(sents)
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
    raw_pdfs_dir = base / data_config["raw_pdfs_dir"]
    raw_documents_dir = base / data_config["raw_documents_dir"]
    graph_dir = base / data_config["graph_dir"]
    translation_dir = base / data_config["translation_dir"]
    baseline_dir = base / data_config["baseline_dir"]

    root_dir.mkdir(exist_ok=True)
    base.mkdir(exist_ok=True)
    raw_pdfs_dir.mkdir(exist_ok=True)
    raw_documents_dir.mkdir(exist_ok=True)
    graph_dir.mkdir(exist_ok=True)
    translation_dir.mkdir(exist_ok=True)
    baseline_dir.mkdir(exist_ok=True)

    return {
        "base_dir": base,
        "raw_pdfs_dir": raw_pdfs_dir,
        "raw_documents_dir": raw_documents_dir,
        "graph_dir": graph_dir,
        "translation_dir": translation_dir,
        "baseline_dir": baseline_dir,
    }
