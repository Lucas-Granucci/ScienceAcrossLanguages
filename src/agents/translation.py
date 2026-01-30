import json
import os
from typing import Dict, List, Optional

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

    def translate(
        self,
        discourse: str,
        memory: Dict[str, str],
        terminology: Optional[Dict[str, str]] = None,
        rag_snippets: Optional[List[str]] = None,
    ) -> str:
        print("translating discourse")
        # Prepare context strings for the prompt
        terminology_str = (
            json.dumps(terminology, indent=2, ensure_ascii=False)
            if terminology
            else "{}"
        )
        rag_str = "\n".join(rag_snippets) if rag_snippets else ""

        prompt = self.user_prompt_template.render(
            incident_memory=json.dumps(memory, indent=2, ensure_ascii=False),
            source_discourse=discourse,
            terminology=terminology_str,
            rag_context=rag_str,
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
