import fitz
import hashlib
from tqdm import tqdm
from pathlib import Path
from utils import normalize_text
from lingua import LanguageDetectorBuilder


def detect_language(text: str, detector):
    result = detector.detect_language_of(text)
    return result.iso_code_639_1.name.lower() if result else None


def process_document(
    pdf_data_path: Path,
    target_lang_code: str,
    target_articles: int,
    processed_output_path: Path,
):
    detector = LanguageDetectorBuilder.from_all_languages().build()

    kept_count = 0
    pdf_files = sorted(pdf_data_path.glob("*.pdf"))

    for pdf_path in tqdm(pdf_files, desc="Converting PDFs to markdown..."):
        try:
            doc = fitz.open(pdf_path)
            full_text = []

            for page in doc:
                blocks = page.get_text("blocks", sort=True)
                for b in blocks:
                    clean_block = b[4].replace("\n", " ").strip()
                    full_text.append(clean_block)
            doc_text = "\n".join(full_text)
            doc.close()

            if not isinstance(doc_text, str):
                continue

            # Verify language
            detected_code = detect_language(doc_text, detector)
            if detected_code == target_lang_code:
                content_hash = hashlib.md5(doc_text.encode("utf-8")).hexdigest()[:8]

                # Rename pdf and save markdown with hash
                new_pdf_name = f"{kept_count:04d}_{content_hash}.pdf"
                new_pdf_path = pdf_data_path / new_pdf_name
                pdf_path.rename(new_pdf_path)

                doc_name = f"{kept_count:04d}_{content_hash}.txt"
                doc_path = processed_output_path / doc_name
                doc_path.write_text(
                    normalize_text(doc_text, ignore_lang="en"), encoding="utf-8"
                )
                kept_count += 1

                if kept_count >= target_articles:
                    for leftover in pdf_files[pdf_files.index(pdf_path) + 1 :]:
                        leftover.unlink(missing_ok=True)
                    break
            else:
                pdf_path.unlink(missing_ok=True)

        except Exception as e:
            print(f"Error converting PDF to markdown: {e}")
            pdf_path.unlink(missing_ok=True)
