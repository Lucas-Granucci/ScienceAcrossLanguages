import logging
from pathlib import Path
from typing import List

import fitz
import nltk
from lingua import LanguageDetectorBuilder
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


def detect_language(text: str, detector):
    result = detector.detect_language_of(text)
    return result.iso_code_639_1.name.lower() if result else None


def _ensure_punkt():
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def extract_text(pdf_path: Path) -> str:
    try:
        doc = fitz.open(pdf_path)
        full_text = []

        for page in doc:
            blocks = page.get_text("blocks", sort=True)
            for b in blocks:
                clean_block = b[4].replace("\n", " ").strip()
                full_text.append(clean_block)
        doc_text = "\n".join(full_text)
        doc.close()

        if not isinstance(doc_text, str):
            return ""

        return doc_text

    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        return ""


def clean_text(document_text: str, target_lang: str, ignore_lang: str) -> List[str]:

    # Check entire doc first
    detector = LanguageDetectorBuilder.from_all_languages().build()

    detected_code = detect_language(document_text, detector)
    if detected_code != target_lang:
        return [""]

    # Split into sentences and clean
    sentences = document_to_sentences(document_text)
    sentences = [
        sent for sent in sentences if detect_language(sent, detector) != ignore_lang
    ]
    sentences = [sent.strip() for sent in sentences]
    return sentences


def document_to_sentences(document: str) -> List[str]:
    _ensure_punkt()
    sentences = sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences
