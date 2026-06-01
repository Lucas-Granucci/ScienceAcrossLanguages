"""Baseline translation using OpenAI Chat Completions."""

import os
from typing import List

from openai import OpenAI

from dotenv import load_dotenv

MODEL_NAME = "gpt-4.1-2025-04-14"

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))


def _build_system_prompt(target_lang_code: str) -> str:
    return (
        "You are a precise translation engine. Translate any user message into the "
        f"language identified by ISO 639-1 code '{target_lang_code}'. Return only the "
        "translated sentence with no extra text."
    )


def chatgpt_translate_sentences(
    sentences: List[str], target_lang_code: str
) -> List[str]:

    if not sentences:
        return []

    system_prompt = _build_system_prompt(target_lang_code)
    translations: List[str] = []

    for sentence in sentences:
        if not sentence:
            translations.append("")
            continue

        try:
            response = _client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": sentence},
                ],
            )
            message = response.choices[0].message.content if response.choices else ""
            translations.append(message.strip() if message else "")
        except Exception as exc:
            print(f"OpenAI translation failed: {exc}")
            translations.append("")

    return translations
