import os

from dotenv import load_dotenv
from utils import load_config, normalize_punctuation
from graph.pipeline import build_translation_pipeline


def main():
    load_dotenv()
    config = load_config()

    # Load data
    with open(config["data"]["input_file"], "r", encoding="utf-8") as fp:
        document_source = normalize_punctuation(fp.read())

    # app, initial_state = build_translation_pipeline(document_source, config)

    # # Run
    # print("Starting translation workflow...")
    # output = app.invoke(initial_state)

    # print("\nFinal Translation:")
    # print(output["final_document"])


if __name__ == "__main__":
    main()
