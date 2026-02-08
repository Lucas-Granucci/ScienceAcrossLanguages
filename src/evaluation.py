import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from comet import download_model, load_from_checkpoint
from dotenv import load_dotenv

from utils import load_config, load_paths


# Default COMET settings
DEFAULT_MODEL = "Unbabel/wmt22-comet-da"
DEFAULT_BATCH_SIZE = 8
DEFAULT_GPUS = 1


def _load_triplets(
    system_dir: Path, graph_dir: Path
) -> Tuple[List[str], List[str], List[str]]:
    src: List[str] = []
    mt: List[str] = []
    ref: List[str] = []

    def _clean_segment(text: str) -> str:
        # Collapse all whitespace/newlines so each segment maps to exactly one line.
        return " ".join((text or "").split())

    for translation_path in sorted(system_dir.glob("*.json")):
        graph_path = graph_dir / f"{translation_path.stem}.json"
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Missing graph for {translation_path.name}; expected {graph_path.name}"
            )

        with translation_path.open("r", encoding="utf-8") as f:
            translation_data = json.load(f)
        with graph_path.open("r", encoding="utf-8") as f:
            graph_data = json.load(f)

        translation_discourses = translation_data.get("discourses", [])
        graph_discourses = graph_data.get("discourses", [])
        if len(translation_discourses) != len(graph_discourses):
            raise ValueError(f"Mismatched discourse counts for {translation_path.name}")

        for td, gd in zip(translation_discourses, graph_discourses):
            src.append(_clean_segment(td.get("source_txt") or ""))
            mt.append(_clean_segment(td.get("translated_txt") or ""))
            ref.append(_clean_segment(gd.get("source_txt") or ""))

    return src, mt, ref


def _align_triplets(
    src: List[str],
    ref: List[str],
    system_mt: Dict[str, List[str]],
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    if not system_mt:
        return [], [], {}

    lengths = {len(v) for v in system_mt.values()}
    lengths.add(len(src))
    lengths.add(len(ref))
    if len(lengths) != 1:
        raise ValueError("Source, reference, and MT lengths must match across systems")

    keep_indices = []
    for i in range(len(src)):
        if not src[i] or not ref[i]:
            continue
        if any(not mt_list[i] for mt_list in system_mt.values()):
            continue
        keep_indices.append(i)

    aligned_src = [src[i] for i in keep_indices]
    aligned_ref = [ref[i] for i in keep_indices]
    aligned_mt = {name: [mt[i] for i in keep_indices] for name, mt in system_mt.items()}
    return aligned_src, aligned_ref, aligned_mt


def _build_comet_dataset(
    src: List[str], mt: List[str], ref: List[str]
) -> List[Dict[str, str]]:
    return [{"src": s, "mt": m, "ref": r} for s, m, r in zip(src, mt, ref)]


def _score_systems(
    model_name: str,
    batch_size: int,
    gpus: int,
    comet_inputs: Dict[str, List[Dict[str, str]]],
) -> Dict[str, float]:
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    scores: Dict[str, float] = {}
    for system_name, samples in comet_inputs.items():
        output = model.predict(samples, batch_size=batch_size, gpus=gpus)
        scores[system_name] = float(output.system_score)
    return scores


def _write_plaintext_corpora(
    run_dir: Path, src: List[str], ref: List[str], aligned_mt: Dict[str, List[str]]
) -> Dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "src.txt").write_text("\n".join(src), encoding="utf-8")
    (run_dir / "ref.txt").write_text("\n".join(ref), encoding="utf-8")

    mt_paths: Dict[str, Path] = {}
    for system_name, lines in aligned_mt.items():
        path = run_dir / f"mt_{system_name}.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        mt_paths[system_name] = path
    return mt_paths


def _run_comet_compare(
    model_name: str,
    gpus: int,
    run_dir: Path,
    system_names: List[str],
) -> Tuple[int, str, str]:
    cmd = [
        "comet-compare",
        "-s",
        str(run_dir / "src.txt"),
        "-t",
        *[str(run_dir / f"mt_{name}.txt") for name in system_names],
        "-r",
        str(run_dir / "ref.txt"),
        "--model",
        model_name,
        "--gpus",
        str(gpus),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, encoding="utf-8"
        )
    except FileNotFoundError:
        return 127, "", "comet-compare not found on PATH"

    return result.returncode, result.stdout, result.stderr


def _discover_all_system_dirs(lang_paths: Dict[str, Path]) -> Dict[str, Path]:
    """Discover every system folder under translation/ (presets) and baseline/."""
    systems: Dict[str, Path] = {}
    for root in (lang_paths["translation_dir"], lang_paths["baseline_dir"]):
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir():
                systems[child.name] = child
    return systems


def run():
    load_dotenv()
    config = load_config()

    lang_pairs = list(config.get("languages", {}).keys())
    model_name = DEFAULT_MODEL
    batch_size = DEFAULT_BATCH_SIZE
    gpus = DEFAULT_GPUS

    for lang_pair in lang_pairs:
        lang_paths = load_paths(config, lang_pair)
        system_dirs = _discover_all_system_dirs(lang_paths)

        if not system_dirs:
            print(f"No system outputs found for {lang_pair}; skipping")
            continue

        # Deterministic ordering of systems for consistent comparison output
        system_dirs = dict(sorted(system_dirs.items()))

        canonical_src: List[str] = []
        canonical_ref: List[str] = []
        system_mt: Dict[str, List[str]] = {}

        for name, system_dir in system_dirs.items():
            src, mt, ref = _load_triplets(system_dir, lang_paths["graph_dir"])
            if not canonical_src:
                canonical_src, canonical_ref = src, ref
            else:
                if len(src) != len(canonical_src) or len(ref) != len(canonical_ref):
                    raise ValueError(
                        f"Length mismatch for system {name} in {lang_pair}"
                    )
            system_mt[name] = mt

        aligned_src, aligned_ref, aligned_mt = _align_triplets(
            canonical_src, canonical_ref, system_mt
        )

        if not aligned_src:
            print(f"No usable segments for {lang_pair}; skipping")
            continue

        comet_inputs = {
            name: _build_comet_dataset(aligned_src, mt, aligned_ref)
            for name, mt in aligned_mt.items()
        }

        run_root = (
            lang_paths["base_dir"]
            / "eval"
            / "comet"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        run_root.mkdir(parents=True, exist_ok=True)

        scores = _score_systems(
            model_name=model_name,
            batch_size=batch_size,
            gpus=gpus,
            comet_inputs=comet_inputs,
        )

        mt_paths = _write_plaintext_corpora(
            run_root, aligned_src, aligned_ref, aligned_mt
        )

        results = {
            "lang_pair": lang_pair,
            "model": model_name,
            "gpus": gpus,
            "batch_size": batch_size,
            "n_segments": len(aligned_src),
            "systems": scores,
        }
        (run_root / "scores.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if len(aligned_mt) >= 2:
            retcode, stdout, stderr = _run_comet_compare(
                model_name=model_name,
                gpus=gpus,
                run_dir=run_root,
                system_names=sorted(aligned_mt.keys()),
            )
            comparison = {
                "returncode": retcode,
                "stdout": stdout,
                "stderr": stderr,
                "command": "comet-compare",
                "systems": list(aligned_mt.keys()),
            }
            (run_root / "compare.txt").write_text(stdout, encoding="utf-8")
            (run_root / "compare.json").write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if retcode != 0:
                print(
                    f"comet-compare failed for {lang_pair} (exit {retcode}); see compare.json"
                )

        if len(aligned_mt) < 2:
            print(f"Only one system for {lang_pair}; comet-compare skipped")


if __name__ == "__main__":
    run()
