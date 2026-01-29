import os
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI


class MemoryAgent:
    def __init__(self, client: OpenAI, model_name: str, language_pair: str) -> None:
        self.client = client
        self.model_name = model_name

        # prompt template, simple (False) or dict (True) response, display string
        self.all_keys = [
            (
                "target_noun_target_pronoun",
                True,
                "Target noun→pronoun mappings:",
            ),
            (
                "source_entity_target_entity",
                True,
                "Source entity→target entity mappings:",
            ),
            (
                "discourse_connectives",
                False,
                "Discourse connectives at end of connected discourses:",
            ),
            (
                "source_phrase_target_phrase",
                True,
                "Source phrase→target phrase mappings:",
            ),
            ("translation_summary", False, "Combined translation summary:"),
            ("context_summary", False, "Combined translation summary:"),
        ]

        env = Environment(
            loader=FileSystemLoader(
                os.path.join(os.path.dirname(__file__), f"../prompts/{language_pair}")
            )
        )

        key_names = [name for name, _, _ in self.all_keys]
        self.prompts = {
            name: env.get_template(f"memory_agent/{name}.jinja") for name in key_names
        }

    def reset_memory(self) -> None:
        self.memory = dict()

    def get_client_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=8192,
        )
        message = response.choices[0].message.content
        if message and message.strip():
            return message.strip()
        return ""

    def process_simple_response(self, output: str):
        s = output.strip()
        return "" if s == "(none)" else s

    def process_dict_response(self, output: str):
        lines = [
            line.strip()
            for line in output.splitlines()
            if line.strip() and line.strip() != "(none)"
        ]
        return dict(line.split(":", 1) for line in lines)

    def get_local_memory(self, discourse: str, translation: str) -> dict:
        local_memory = {
            "target_noun_target_pronoun_mapping": {},
            "source_entity_target_entity_mapping": {},
            "source_phrase_target_phrase_mapping": {},
            "discourse_connectives": "",
            "translation_summary": "",
            "context_summary": "",
        }

        for batch in self.all_keys:
            name, dict_struct, _ = batch

            try:
                prompt = self.prompts[name].render(
                    source_discourse=discourse, target_discourse=translation
                )
                response = self.get_client_response(prompt)
                content = (
                    self.process_dict_response(response)
                    if dict_struct
                    else self.process_simple_response(response)
                )

                local_memory[name] = content
            except Exception as _:
                pass

        return local_memory

    def get_incident_memory(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        combined = {
            "target_noun_target_pronoun_mapping": {},
            "source_entity_target_entity_mapping": {},
            "discourse_connectives": "",
            "source_phrase_target_phrase_mapping": {},
            "translation_summary": "",
            "context_summary": "",
        }

        summaries = []
        contexts = []
        for mem in memories:
            combined["target_noun_target_pronoun_mapping"].update(
                mem.get("target_noun_target_pronoun_mapping", {})
            )
            combined["source_entity_target_entity_mapping"].update(
                mem.get("source_entity_target_entity_mapping", {})
            )
            combined["source_phrase_target_phrase_mapping"].update(
                mem.get("source_phrase_target_phrase_mapping", {})
            )
            combined["discourse_connectives"] = mem.get("discourse_connectives", "")
            summary = mem.get("translation_summary", "").strip()
            if summary:
                summaries.append(summary)
            context = mem.get("context_summary", "").strip()
            if context:
                contexts.append(context)
        combined["translation_summary"] = " ".join(summaries) or "(none)"
        combined["context_summary"] = " ".join(contexts) or "(none)"
        return combined

    def encode_memory(self, memory: Dict[str, Any]) -> str:
        """Encode memory into a string suited for LLM consumption"""
        lines = []

        for batch in self.all_keys:
            name, dict_struct, display_string = batch

            lines.append(display_string)
            if memory[name]:
                if dict_struct:
                    for item1, item2 in memory[name].items():
                        lines.append(f"- {item1} → {item2}")
                else:
                    lines.append(memory[name])
            else:
                lines.append("- (none)")

        return "\n".join(lines)
