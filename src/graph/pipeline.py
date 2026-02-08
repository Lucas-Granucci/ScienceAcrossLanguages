import json
import os

from openai import OpenAI

from agents import (
    DependencyGraphAgent,
    MemoryAgent,
    TranslationAgent,
    TerminologyAgent,
    RAGAgent,
)
from graph.state import DiscourseUnit, GraphState
from graph.graph_builder import GraphBuilder
from pathlib import Path


def load_graph_state(graph_path: Path, source_field: str = "translated_txt"):
    """Load a saved graph and prepare discourses for another run.

    source_field controls which field from each discourse becomes the new source.
    Typical options: "translated_txt" (backtranslation) or "source_txt".
    """
    with graph_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    discourses: list[DiscourseUnit] = []
    for d in data.get("discourses", []):
        source_for_next = d.get(source_field) or (
            d.get("source_txt") if source_field != "source_txt" else ""
        )
        discourses.append(
            DiscourseUnit(
                id=d["idx"],
                source_text=source_for_next,
                target_text=None,
                incident_memory={},
                local_memory={},
                terminology_context={},
                rag_context=[],
            )
        )

    translated_segments = [d.get("translated_txt") for d in data.get("discourses", [])]
    joined_translation = " ".join(filter(None, translated_segments))

    return {
        "discourses": discourses,
        "edges": data.get("edges", []),
        "source_document": data.get("final_document") or joined_translation,
        "language_pair": data.get("language_pair"),
    }


def build_translation_pipeline(
    source_document: str,
    language_pair: str,
    config: dict,
    graph_save_dir: Path,
    preset: str = "base",
    preloaded_state: dict | None = None,
):
    preset_cfg = config.get("presets", {}).get(preset)
    if preset_cfg is None:
        raise ValueError(f"Unkown preset '{preset}'")

    processing_client = OpenAI(
        base_url=config["processing"]["base_url"],
        api_key=config["processing"]["api_key"],
    )
    translation_client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))

    processing_model_name = config["processing"]["model_name"]
    translation_model_name = config["translation"]["model_name"]

    # ---- Initialize agents ----
    dependency_graph_agent = DependencyGraphAgent(
        processing_client,
        processing_model_name,
        language_pair,
    )

    memory_agent = MemoryAgent(processing_client, processing_model_name, language_pair)

    translation_agent = TranslationAgent(
        translation_client,
        translation_model_name,
        language_pair,
    )

    builder = GraphBuilder(dependency_graph_agent, memory_agent, translation_agent)
    modules = preset_cfg.get("modules", [])
    if "terminology" in modules:
        term = TerminologyAgent(processing_client, processing_model_name, language_pair)
        builder.with_terminology(term, position="before:translate")
    if "rag" in modules:
        rag = RAGAgent(processing_client, processing_model_name, language_pair)
        builder.with_rag(rag, position="before:translate")

    app = builder.build()

    initial_state = GraphState(
        source_document=(preloaded_state or {}).get("source_document")
        or source_document
        or "",
        language_pair=language_pair,
        current_index=0,
        discourses=(preloaded_state or {}).get("discourses", []),
        edges=(preloaded_state or {}).get("edges", []),
        final_document="",
        graph_save_dir=graph_save_dir,
    )

    return app, initial_state
