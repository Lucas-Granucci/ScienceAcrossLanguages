import os
import re
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from agents import (
    DiscourseAgent,
    EdgeAgent,
    MemoryAgent,
    PlannerAgent,
    TranslationAgent,
)
from graph.state import GraphState
from graph.workflow import create_translation_graph


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


def main():
    load_dotenv()
    # client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))
    # model_name = "gpt-5-nano-2025-08-07"

    processing_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    processing_model_name = "qwen2.5:7b-instruct"

    translation_client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))
    translation_model_name = "gpt-5-mini"

    source_lang = "English"
    target_lang = "Vietnamese"
    language_pair = "en-vi"

    # Initialize agents
    discourse_agent = DiscourseAgent(
        processing_client,
        processing_model_name,
        source_lang,
        target_lang,
        language_pair,
    )
    edge_agent = EdgeAgent(
        processing_client,
        processing_model_name,
        source_lang,
        target_lang,
        language_pair,
    )
    planner_agent = PlannerAgent(discourse_agent, edge_agent)
    memory_agent = MemoryAgent(processing_client, processing_model_name, language_pair)

    translation_agent = TranslationAgent(
        translation_client,
        translation_model_name,
        source_lang,
        target_lang,
        language_pair,
    )

    # Compile graph
    app = create_translation_graph(planner_agent, memory_agent, translation_agent)

    # Load data
    doc_file_path = "data/en-vi/0.en"
    with open(doc_file_path, "r", encoding="utf-8") as fp:
        document_source = normalize_punctuation(fp.read())

    initial_state = GraphState(
        source_document=document_source,
        source_lang=source_lang,
        target_lang=target_lang,
        language_pair=language_pair,
        current_index=0,
        discourses=[],
        edges=[],
        final_document="",
    )

    # Run
    print("Starting translation workflow...")
    output = app.invoke(initial_state)

    print("\nFinal Translation:")
    print(output["final_document"])


if __name__ == "__main__":
    main()
