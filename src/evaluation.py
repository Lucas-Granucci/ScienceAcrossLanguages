import os

from dotenv import load_dotenv
from openai import OpenAI

from graph.state import GraphState
from graph.workflow import create_translation_graph
from utils import load_config, normalize_punctuation

from baselines.google_translate import gtranslate_document


def main():
    load_dotenv()
    config = load_config()

    translation_client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))

    processing_model_name = config["processing"]["model_name"]
    translation_model_name = config["translation"]["model_name"]

    source_lang = config["languages"]["source"]
    target_lang = config["languages"]["target"]
    language_pair = config["languages"]["pair"]

    # Load data
    source_docs = [
        path for path in os.listdir("data/processed") if path.endswith(".md")
    ]
