import json
import os
from typing import Dict

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI


class TranslationAgent:
    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        source_lang: str,
        target_lang: str,
        language_pair: str,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        env = Environment(
            loader=FileSystemLoader(
                os.path.join(os.path.dirname(__file__), f"../prompts/{language_pair}")
            )
        )
        self.user_prompt_template = env.get_template("translation_agent/user.jinja")

    def translate(self, discourse: str, memory: Dict[str, str]) -> str:
        prompt = self.user_prompt_template.render(
            incident_memory=json.dumps(memory, indent=2, ensure_ascii=False),
            source_discourse=discourse,
        )

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
