import os
from openai import OpenAI
from pathlib import Path
from agents import (
    DependencyGraphAgent,
    MemoryAgent,
    TranslationAgent,
    TerminologyAgent,
    RAGAgent,
)
from graph.graph_builder import GraphBuilder, GraphState
from graph.pipeline import load_graph_state
from utils import load_config


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
    translation_client = OpenAI(
        base_url=config["processing"]["base_url"],
        api_key=config["processing"]["api_key"],
    )

    processing_model_name = config["processing"]["model_name"]
    translation_model_name = config["processing"]["model_name"]

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
        print("built")
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


config = load_config()

preloaded_state = load_graph_state(
    Path(
        r"C:\Users\Admin\Desktop\2025-2026_LucasGranucci\ScienceAcrossLanguages\data\en-vi\graphs\0000_7f19d671.json"
    ),
    source_field="translated_txt",
)

app, initial_state = build_translation_pipeline(
    source_document=None,
    language_pair="en-vi",
    config=config,
    graph_save_dir=Path(
        r"C:\Users\Admin\Desktop\2025-2026_LucasGranucci\ScienceAcrossLanguages\data\en-vi\graphs_temp\out.json"
    ),
    preset="term_only",
    preloaded_state=preloaded_state,
)
_ = app.invoke(initial_state)
