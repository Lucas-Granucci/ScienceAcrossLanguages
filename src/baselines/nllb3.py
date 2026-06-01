from typing import List
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-3.3B"
_SOURCE_LANG_ISO = "en"

_LANG_CODE_MAP = {
    "en": "eng_Latn",
    "vi": "vie_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "sw": "swh_Latn",
}


device = "cuda"


def _get_lang_code(iso_code: str) -> str:
    try:
        return _LANG_CODE_MAP[iso_code]
    except KeyError as exc:
        raise ValueError(f"Unsupported ISO language code: {iso_code}") from exc


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)


def nllb_translate_sentences(sentences: List[str], target_lang_code: str) -> List[str]:
    src_lang = _get_lang_code(_SOURCE_LANG_ISO)
    tgt_lang = _get_lang_code(target_lang_code)

    tokenizer.src_lang = src_lang
    BATCH_SIZE = 32
    final_translations = []

    for i in range(0, len(sentences), BATCH_SIZE):
        print(f"Processing batch... {i}/{len(sentences) // BATCH_SIZE}")
        batch_sents = sentences[i : i + BATCH_SIZE]

        inputs = tokenizer(
            batch_sents, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            )
        tran_sents = [
            text.strip()
            for text in tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
        ]
        final_translations.extend(tran_sents)
        del inputs, generated_tokens
        torch.cuda.empty_cache()

    return final_translations
