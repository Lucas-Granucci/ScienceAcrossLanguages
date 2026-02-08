from pathlib import Path
from dotenv import load_dotenv
from ..utils import load_paths, load_config
from comet import download_model, load_from_checkpoint
import json


def run_doc_comet():
    load_dotenv()
    config = load_config()

    # model_path = download_model("Unbabel/XCOMET-XL")

    for lang_pair, _ in config["languages"].items():
        lang_data_paths = load_paths(config, lang_pair)

        preset = "base"
        translated_dir = lang_data_paths["translation_dir"] / preset

        src = []
        mt = []
        ref = []
        for translation_path in sorted(translated_dir.glob("*.json")):
            input_discourse_path = (
                lang_data_paths["graph_dir"] / f"{translation_path.stem}.json"
            )

            with open(translation_path, "r", encoding="utf-8") as f:
                translation_data = json.load(f)
                for d in translation_data["discourses"]:
                    src.append(d["source_txt"])
                    mt.append(d["translated_txt"])

            with open(input_discourse_path, "r", encoding="utf-8") as f:
                input_discourses = json.load(f)
                for d in input_discourses["discourses"]:
                    ref.append(d["source_txt"])

        print(len(src))
        print(len(mt))
        print(len(ref))
