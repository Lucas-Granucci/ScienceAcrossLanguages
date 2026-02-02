import asyncio
from tqdm import tqdm
from googletrans import Translator
from utils import document_to_sentences, normalize_punctuation


async def translate_and_write_batches(
    sentences, target_lang_code, batch_size, output_file
):
    async with Translator() as translator:
        loop = asyncio.get_event_loop()

        for i in tqdm(range(0, len(sentences), batch_size)):
            sent_batch = sentences[i : i + batch_size]
            sent_text = [sent["source_text"] for sent in sent_batch]

            translations = await translator.translate(sent_text, dest=target_lang_code)
            translated_texts = [translation.text for translation in translations]

            await loop.run_in_executor(
                None, write_to_file, output_file, translated_texts
            )


def write_to_file(path, lines):
    with open(path, "a", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")


def gtranslate_document(
    source_file_path: str, target_lang_code: str, output_file_path: str
):
    with open(source_file_path, "r", encoding="utf-8") as f:
        source_text = f.read()
    source_text = normalize_punctuation(source_text)
    sentences = document_to_sentences(source_text)

    asyncio.run(
        translate_and_write_batches(
            sentences, target_lang_code, batch_size=10, output_file=output_file_path
        )
    )
