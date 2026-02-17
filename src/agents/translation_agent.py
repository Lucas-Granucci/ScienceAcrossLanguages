from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI


class TranslationAgent:
    def __init__(self, client: OpenAI, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

        prompts_dir = Path("config/prompts")
        env = Environment(loader=FileSystemLoader(prompts_dir))
        self.translation_prompt_template = env.get_template("translation.jinja")

    def translate(
        self,
        discourse: str,
        source_lang: str,
        target_lang: str,
        memory_str: str,
        terminology_str: Optional[str] = None,
        rag_snippets_str: Optional[str] = None,
    ) -> str:

        prompt = self.translation_prompt_template.render(
            source_lang=source_lang,
            target_lang=target_lang,
            incident_memory=memory_str,
            source_discourse=discourse,
            terminology=terminology_str,
            rag_context=rag_snippets_str,
        )

        print(len(prompt))
        print(len(prompt.split(" ")))
        print()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": prompt}],
                max_completion_tokens=4096,
            )
            message = response.choices[0].message.content
            return message.strip() if message else ""
        except Exception as e:
            print(f"Error during translation: {e}")
            return ""
