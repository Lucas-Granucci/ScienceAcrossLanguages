import argparse
import hashlib
import json
import logging
from pathlib import Path

import yaml
from tqdm import tqdm

from src.ingestion.downloader import download_pdf, setup_pdf_driver
from src.ingestion.metadata import fetch_openalex_metadata
from src.ingestion.parser import clean_text, extract_text

# Setup logging
format = "%(asctime)s - %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=format)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fetch papers from OpenAlex")
    parser.add_argument("--num_docs", type=int, required=True, help="# documents")
    parser.add_argument("--limit", type=int, default=100, help="Max # of docs to fetch")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup directory
    source_code = config["language"]["source_code"]
    target_code = config["language"]["target_code"]
    language_pair = f"{source_code}-{target_code}"

    base_dir = Path(config["paths"]["base_dir"]) / language_pair
    raw_dir = base_dir / Path(config["paths"]["raw_data"])
    proc_dir = base_dir / Path(config["paths"]["processed_data"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching {args.num_docs} documents for {target_code} from OpenAlex")
    documents = fetch_openalex_metadata(target_code, args.limit)

    # Download PDFs
    driver = setup_pdf_driver(raw_dir)
    for doc in tqdm(documents, desc="Downloading documents"):
        download_pdf(doc["pdf_url"], driver)

    # Process PDFs and convert to sentences
    kept_count = 0
    for pdf_path in tqdm(raw_dir.glob("*.pdf"), desc="Parsing documents..."):
        doc_text = extract_text(pdf_path)
        sentences = clean_text(doc_text, target_code, source_code)

        if len(sentences) < 10:
            pdf_path.unlink()
            continue

        id = hashlib.md5(pdf_path.stem.encode("utf-8")).hexdigest()[:8]
        doc_path = proc_dir / f"{kept_count:04d}_{id}.jsonl"

        with open(doc_path, "w", encoding="utf-8") as f:
            for sent in sentences:
                f.write(json.dumps({"text": sent}, ensure_ascii=False) + "\n")

        new_pdf_path = raw_dir / f"{kept_count:04d}_{id}.pdf"
        pdf_path.rename(new_pdf_path)

        kept_count += 1
        if kept_count >= args.num_docs:
            break

    logger.info(f"Downloaded {kept_count} documents")


if __name__ == "__main__":
    main()
