import argparse
import json
import logging
import os
from pathlib import Path

import yaml
from tqdm import tqdm

from src.core.graph_builder import TranslationPipeline

# Setup logging
format = "%(asctime)s - %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=format)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Backtranslate fetched docs -> source")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        content = os.path.expandvars(f.read())
    config = yaml.safe_load(content)

    # Setup directory
    source_code = config["language"]["source_code"]
    target_code = config["language"]["target_code"]
    language_pair = f"{source_code}-{target_code}"

    base_dir = Path(config["paths"]["base_dir"]) / language_pair
    proc_dir = base_dir / Path(config["paths"]["processed_data"])
    backtrans_dir = base_dir / Path(config["paths"]["backtranslated"])
    backtrans_dir.mkdir(parents=True, exist_ok=True)

    if not proc_dir.exists():
        raise ValueError("Use run_ingestion to collect documents first")

    # Flip source/target because backtranslation
    translator = TranslationPipeline(
        source_lang=config["language"]["target"],
        target_lang=config["language"]["source"],
        config=config,
    )

    for doc_path in tqdm(proc_dir.glob("*.jsonl"), desc="Backtranslating documents..."):
        with open(doc_path, "r", encoding="utf-8") as f:
            document = [json.loads(json_str) for json_str in f.readlines()]
        sentences = [line["text"] for line in document]

        graph_save_dir = backtrans_dir / f"{doc_path.stem}.json"

        translator.run(
            source_sentences=sentences,
            graph_save_dir=graph_save_dir,
            preloaded_state=None,
        )


if __name__ == "__main__":
    main()
