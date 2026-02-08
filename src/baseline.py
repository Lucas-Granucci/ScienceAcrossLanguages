from dotenv import load_dotenv
from utils import load_paths, load_config
from baselines.google_translate import gtranslate_sentences
from baselines.nllb import nllb_translate_sentences
import json


BASELINES = {
    "google_translate": gtranslate_sentences,
    "nllb": nllb_translate_sentences,
}

SELECTED_BASELINES = ["nllb"]


def main():
    load_dotenv()
    config = load_config()

    for lang_pair, _ in config["languages"].items():
        lang_data_paths = load_paths(config, lang_pair)

        target_lang_code = config["languages"][lang_pair]["target_code"]
        for baseline_name in SELECTED_BASELINES:
            translator = BASELINES.get(baseline_name)
            if translator is None:
                raise ValueError(f"Unknown baseline: {baseline_name}")

            output_dir = lang_data_paths["baseline_dir"] / baseline_name
            output_dir.mkdir(parents=True, exist_ok=True)

            for input_graph_path in sorted(lang_data_paths["graph_dir"].glob("*.json")):
                with input_graph_path.open("r", encoding="utf-8") as f:
                    graph_data = json.load(f)

                discourses = graph_data.get("discourses", [])

                # Use the translated text from the graph as the English source for back-translation.
                src_segments = [
                    (d.get("translated_txt") or "").strip() for d in discourses
                ]

                translations = translator(src_segments, target_lang_code)
                translations = [(t or "").strip() for t in translations]

                final_document = " ".join([t for t in translations if t])

                output = {
                    "language_pair": lang_pair,
                    "source_document": graph_data.get("final_document", ""),
                    "final_document": final_document,
                    "discourses": [
                        {
                            "idx": d.get("idx", i),
                            "source_txt": src_segments[i],
                            "translated_txt": translations[i],
                        }
                        for i, d in enumerate(discourses)
                    ],
                    "edges": graph_data.get("edges", []),
                }

                output_path = output_dir / f"{input_graph_path.stem}.json"
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)

                print(f"Saved {baseline_name} translation to {output_path}")


if __name__ == "__main__":
    main()
