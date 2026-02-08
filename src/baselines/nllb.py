from typing import List
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"
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
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_lang_code(iso_code: str) -> str:
    try:
        return _LANG_CODE_MAP[iso_code]
    except KeyError as exc:
        raise ValueError(f"Unsupported ISO language code: {iso_code}") from exc


def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    return tokenizer, model


def nllb_translate_sentences(sentences: List[str], target_lang_code: str) -> List[str]:
    tokenizer, model = _load_model()
    src_lang = _get_lang_code(_SOURCE_LANG_ISO)
    tgt_lang = _get_lang_code(target_lang_code)

    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        sentences, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
    )

    return [
        text.strip()
        for text in tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    ]
