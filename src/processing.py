import json
import os
import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Discourse:
    source_txt: str
    target_txt: str | None
    memory_incident: dict
    memory_local: dict


@dataclass
class Instance:
    name: str

    document_source: str
    document_source_sentences: List[str]
    document_translation_output: str | None
    document_reference_translation: str | None

    source_lang: str
    target_lang: str
    source_ext: str
    target_ext: str

    discourses: List[Discourse]
    edges: List[Tuple[int, int]]


def normalize_punctuation(text: str) -> str:
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.,!?;:])(?=[^\s])", r"\1 ", text)
    return text


def document_to_sentences(document: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$", document)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


def parse_document(document: str) -> Tuple[str, List[str]]:
    document = document.replace("\n", " ")
    document = normalize_punctuation(document)
    sentences = document_to_sentences(document)
    return document, sentences


def instance_to_dict(instance) -> dict:
    return {
        "name": instance.name,
        "document_source": instance.document_source,
        "document_source_sentences": instance.document_source_sentences,
        "document_translation_output": instance.document_translation_output,
        "document_reference_translation": instance.document_reference_translation,
        "source_lang": instance.source_lang,
        "target_lang": instance.target_lang,
        "source_ext": instance.source_ext,
        "target_ext": instance.target_ext,
        "discourses": [
            {
                "source_txt": discourse.source_txt,
                "target_txt": discourse.target_txt,
                "memory_incident": discourse.memory_incident,
                "memory_local": discourse.memory_local,
            }
            for discourse in instance.discourses
        ],
        "edges": instance.edges,
    }


def save_data(path: str, instances: List[Instance]) -> None:
    os.makedirs(path, exist_ok=True)
    for instance in instances:
        file_path = os.path.join(path, f"{instance.name}.json")
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(
                instance_to_dict(instance), json_file, ensure_ascii=False, indent=4
            )


def load_data(
    path: str, source_lang: str, target_lang: str, source_code: str, target_code: str
) -> List[Instance]:
    filenames = [
        fl.split(".")[-2] for fl in os.listdir(path) if fl.endswith(source_code)
    ]

    instances = []
    for filename in filenames:
        source_file_path = os.path.join(path, filename + f".{source_code}")

        with open(source_file_path, "r", encoding="utf-8") as fp:
            document_source, document_source_sentences = parse_document(fp.read())

        instance = Instance(
            name=filename,
            document_source=document_source,
            document_source_sentences=document_source_sentences,
            document_reference_translation=None,
            document_translation_output=None,
            source_lang=source_lang,
            target_lang=target_lang,
            source_ext=source_code,
            target_ext=target_code,
            discourses=list(),
            edges=list(),
        )
        instances.append(instance)
    return instances
