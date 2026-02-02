import pathlib
import pymupdf4llm
import re
import strip_markdown


import fitz  # PyMuPDF

doc = fitz.open("data/en-vi/unprocessed/0000_e581ca7d.pdf")
full_text = []

for page in doc:
    # 'blocks' sorts the text by columns and rows automatically
    blocks = page.get_text("blocks", sort=True)
    for b in blocks:
        # b[4] is the actual text content of the block
        clean_block = b[4].replace("\n", " ").strip()
        full_text.append(clean_block)
pathlib.Path("output_fitz.txt").write_text("\n".join(full_text), encoding="utf-8")
# Result: A list of clean, ordered paragraphs.

md_text = pymupdf4llm.to_markdown(
    "data/en-vi/unprocessed/0000_e581ca7d.pdf", write_images=True
)

# def strip_markdown(text: str) -> str:
# text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
# text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
# text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
# text = re.sub(r"\*(.+?)\*", r"\1", text)
# text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
# text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
# return text


if isinstance(md_text, str):
    pathlib.Path("output.md").write_text(md_text, encoding="utf-8")
    pathlib.Path("output.txt").write_text(
        strip_markdown.strip_markdown(md_text), encoding="utf-8"
    )
