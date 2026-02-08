import asyncio
from typing import List

from googletrans import Translator


async def _translate_async(sentences: List[str], target_lang_code: str) -> List[str]:
    async with Translator() as translator:
        translations = await translator.translate(sentences, dest=target_lang_code)
        return [translation.text for translation in translations]


def gtranslate_sentences(sentences: List[str], target_lang_code: str) -> List[str]:
    """Translate a list of sentences and return the translated texts."""
    return asyncio.run(_translate_async(sentences, target_lang_code))
