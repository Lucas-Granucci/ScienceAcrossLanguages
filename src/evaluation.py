import os

from dotenv import load_dotenv
from openai import OpenAI

from graph.state import GraphState
from graph.workflow import create_translation_graph
from utils import load_config, normalize_text, load_paths

from baselines.google_translate import gtranslate_document


def main():
    load_dotenv()
    config = load_config()

    load_dotenv()
    config = load_config()

    for lang_pair, lang_config in config["languages"].items():
        lang_data_paths = load_paths(config, lang_pair)

        # Generate baseline translations

        # Calculate metrics

        for backtranslated_path in sorted(
            lang_data_paths["backtranslated_dir"].glob("*.txt")
        ):
            with backtranslated_path.open("r", encoding="utf-8") as fp:
                source_text = fp.read()
