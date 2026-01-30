import os

from dotenv import load_dotenv
from openai import OpenAI

from agents import (
    DependencyGraphAgent,
    MemoryAgent,
    TranslationAgent,
)
from graph.state import GraphState
from graph.workflow import create_translation_graph
from utils import load_config, normalize_punctuation


def main():
    load_dotenv()
    config = load_config()

    processing_client = OpenAI(
        base_url=config["processing"]["base_url"],
        api_key=config["processing"]["api_key"],
    )
    translation_client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))

    processing_model_name = config["processing"]["model_name"]
    translation_model_name = config["translation"]["model_name"]

    source_lang = config["languages"]["source"]
    target_lang = config["languages"]["target"]
    language_pair = config["languages"]["pair"]

    # Initialize agents
    dependency_graph_agent = DependencyGraphAgent(
        processing_client,
        processing_model_name,
        source_lang,
        target_lang,
        language_pair,
    )

    memory_agent = MemoryAgent(processing_client, processing_model_name, language_pair)

    translation_agent = TranslationAgent(
        translation_client,
        translation_model_name,
        source_lang,
        target_lang,
        language_pair,
    )

    # Compile graph
    app = create_translation_graph(
        dependency_graph_agent, memory_agent, translation_agent
    )

    # Load data
    with open(config["data"]["input_file"], "r", encoding="utf-8") as fp:
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
