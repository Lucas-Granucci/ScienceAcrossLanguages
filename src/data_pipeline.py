from pathlib import Path

from ingestion.downloader import download_pdfs
from ingestion.metadata import download_metadata
from ingestion.parser import process_document
from utils import load_paths, load_config
from graph.pipeline import build_translation_pipeline
from dotenv import load_dotenv


def backtranslate(
    processed_path: Path,
    language_pair: str,
    graph_save_dir: Path,
    config,
) -> None:
    with processed_path.open("r", encoding="utf-8") as fp:
        document_source = fp.read()

    app, initial_state = build_translation_pipeline(
        document_source,
        language_pair,
        config,
        graph_save_dir,
        preset="base",
    )
    _ = app.invoke(
        initial_state
    )  # JSON containing constructed graph is saved to save_dir


def collect_articles(
    lang_code: str, target_articles: int, max_articles: int, data_paths: dict
) -> None:
    article_metadata = download_metadata(
        lang_code, max_articles, data_paths["base_dir"]
    )
    download_pdfs(article_metadata, data_paths["raw_pdfs_dir"])
    process_document(
        data_paths["raw_pdfs_dir"],
        lang_code,
        target_articles,
        data_paths["raw_documents_dir"],
    )


def main():
    load_dotenv()
    config = load_config()

    for lang_pair, lang_config in config["languages"].items():
        lang_data_paths = load_paths(config, lang_pair)
        collect_articles(lang_config["target_code"], 10, 100, lang_data_paths)

        reversed_lang_pair = "-".join(reversed(lang_pair.split("-")))

        for processed_path in sorted(
            lang_data_paths["raw_documents_dir"].glob("*.txt")
        ):
            graph_save_path = (
                lang_data_paths["graph_dir"] / f"{processed_path.stem}.json"
            )
            backtranslate(
                processed_path,
                reversed_lang_pair,
                graph_save_path,
                config,
            )


if __name__ == "__main__":
    main()
