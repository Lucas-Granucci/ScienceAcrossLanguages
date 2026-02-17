import argparse
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
    parser = argparse.ArgumentParser(description="Run translation graph")
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
    backtrans_dir = base_dir / Path(config["paths"]["backtranslated"])
    translated_dir = base_dir / Path(config["paths"]["translated"])
    translated_dir.mkdir(parents=True, exist_ok=True)

    if not backtrans_dir.exists():
        raise ValueError("Use run_backtranslation to prepare documents first")

    # Flip source/target because backtranslation
    translator = TranslationPipeline(
        source_lang=config["language"]["source"],
        target_lang=config["language"]["target"],
        config=config,
    )

    for input_graph_path in tqdm(
        backtrans_dir.glob("*.json"), desc="Translating documents..."
    ):
        # Loading backtranslated data, so have to swap direction
        input_data = translator.load_from_json(input_graph_path, swap_direction=True)
        graph_save_dir = translated_dir / f"{input_graph_path.stem}.json"

        translator.run(
            source_sentences=input_data["source_sentences"],
            graph_save_dir=graph_save_dir,
            preloaded_state=input_data,
        )


if __name__ == "__main__":
    main()
