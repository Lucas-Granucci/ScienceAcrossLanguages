from pathlib import Path

from ingestion.downloader import download_pdfs
from ingestion.metadata import download_metadata
from ingestion.parser import process_document
from utils import load_paths, load_config, normalize_text
from graph.pipeline import build_translation_pipeline
from dotenv import load_dotenv


def translate(
    source_path: Path, language_pair: str, translation_dir: Path, config
) -> None:
    with source_path.open("r", encoding="utf-8") as fp:
        source_text = fp.read()

    presets = ["base", "term_only", "rag_and_term"]

    for preset in presets:
        app, initial_state = build_translation_pipeline(
            source_text, language_pair, config, preset=preset
        )
        output = app.invoke(initial_state)
        translation_dir = translation_dir.parent / (
            translation_dir.stem + preset + ".txt"
        )
        translation_dir.write_text(output["final_document"], encoding="utf-8")


def main():
    load_dotenv()
    config = load_config()

    for lang_pair, _ in config["languages"].items():
        lang_data_paths = load_paths(config, lang_pair)

        for source_path in sorted(lang_data_paths["backtranslated_dir"].glob("*.txt")):
            translated_path = (
                lang_data_paths["translation_dir"] / f"{source_path.stem}.txt"
            )
            translate(source_path, lang_pair, translated_path)


if __name__ == "__main__":
    main()
