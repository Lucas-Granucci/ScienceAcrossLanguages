from pathlib import Path
from dotenv import load_dotenv
from utils import load_paths, load_config
from graph.pipeline import build_translation_pipeline, load_graph_state


def translate(
    source_path: Path,
    language_pair: str,
    preset: str,
    output_graph_path: Path,
    input_json_path: Path,
    config,
) -> None:
    if input_json_path:
        preloaded_state = load_graph_state(
            input_json_path, source_field="translated_txt"
        )
        app, initial_state = build_translation_pipeline(
            None,
            language_pair,
            config,
            graph_save_dir=output_graph_path,
            preset=preset,
            preloaded_state=preloaded_state,
        )
        _ = app.invoke(initial_state)

    elif source_path:
        with source_path.open("r", encoding="utf-8") as fp:
            source_text = fp.read()

        app, initial_state = build_translation_pipeline(
            source_text,
            language_pair,
            config,
            graph_save_dir=output_graph_path,
            preset=preset,
        )
        _ = app.invoke(initial_state)


def main():
    load_dotenv()
    config = load_config()

    for lang_pair, _ in config["languages"].items():
        lang_data_paths = load_paths(config, lang_pair)

        preset = "base"

        for input_discourse_json_path in sorted(
            lang_data_paths["graph_dir"].glob("*.json")
        ):
            translated_dir = lang_data_paths["translation_dir"] / preset
            translated_dir.mkdir(exist_ok=True)

            output_graph_path = (
                translated_dir / f"{input_discourse_json_path.stem}.json"
            )
            translate(
                None,
                lang_pair,
                preset,
                output_graph_path,
                input_discourse_json_path,
                config,
            )


if __name__ == "__main__":
    main()
