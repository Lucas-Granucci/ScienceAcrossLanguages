import os

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from processing import Instance, load_data, save_data
from workflow import workflow

load_dotenv()


def translate_document(
    instance: Instance,
    source_lang: str,
    target_lang: str,
    source_code: str,
    target_code: str,
    model_name: str,
) -> Instance:
    client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))
    instance = workflow(
        client,
        model_name,
        instance,
        source_lang,
        target_lang,
        f"{source_code}-{target_code}",
    )
    return instance


def process_files(
    input_path: str,
    output_path: str,
    source_lang: str,
    target_lang: str,
    source_code: str,
    target_code: str,
    model_name: str,
):
    instances = load_data(
        input_path, source_lang, target_lang, source_code, target_code
    )
    processed_instances = []
    for instance in tqdm(instances, desc="Processing documents..."):
        processed_instance = translate_document(
            instance, source_lang, target_lang, source_code, target_code, model_name
        )
        processed_instances.append(processed_instance)

    save_data(output_path, processed_instances)


def main():
    process_files(
        input_path="data/en-vi",
        output_path="data/en-vi",
        source_lang="English",
        target_lang="Vietnamese",
        source_code="en",
        target_code="vi",
        model_name="gpt-5-mini-2025-08-07",
    )


if __name__ == "__main__":
    main()
