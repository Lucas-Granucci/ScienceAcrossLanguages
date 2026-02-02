import os

from openai import OpenAI

from agents import (
    DependencyGraphAgent,
    MemoryAgent,
    TranslationAgent,
    TerminologyAgent,
    RAGAgent,
)
from graph.state import GraphState
from graph.graph_builder import GraphBuilder


def build_translation_pipeline(
    source_document: str,
    language_pair: str,
    config: dict,
    preset: str = "base",
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
        source_document=source_document,
        language_pair=language_pair,
        current_index=0,
        discourses=[],
        edges=[],
        final_document="",
    )

    return app, initial_state
