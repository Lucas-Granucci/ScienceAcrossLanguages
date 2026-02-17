import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
from pydantic import BaseModel


@dataclass
class MemoryComponent:
    name: str
    returns_mapping: bool
    display_string: str
    response_format: Type[BaseModel]


# Response formats
class Entity(BaseModel):
    source_term: str
    target_term: str


class EntityMappingResponse(BaseModel):
    entity_map: List[Entity]


class DiscourseConnectiveResponse(BaseModel):
    connective: Optional[str]


class ContextSummaryResponse(BaseModel):
    summary: str


class MemoryAgent:
    def __init__(self, client: OpenAI, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

        # prompt template, simple (False) or dict (True) response, display string, response format
        self.components = [
            MemoryComponent(
                name="entity_mapping",
                returns_mapping=True,
                display_string="Entity mappings:",
                response_format=EntityMappingResponse,
            ),
            MemoryComponent(
                name="discourse_connectives",
                returns_mapping=False,
                display_string="Discourse connectives:",
                response_format=DiscourseConnectiveResponse,
            ),
            MemoryComponent(
                name="context_summary",
                returns_mapping=False,
                display_string="Context summary:",
                response_format=ContextSummaryResponse,
            ),
        ]

        self._load_prompts()
        self.memory: Dict[str, Any] = {}

    def _load_prompts(self):
        prompts_dir = Path("config/prompts")
        env = Environment(loader=FileSystemLoader(prompts_dir))
        self.prompts = {
            component.name: env.get_template(f"memory/{component.name}.jinja")
            for component in self.components
        }

    def reset_memory(self) -> None:
        self.memory = dict()

    def _get_structured_response(self, prompt: str, response_format: Type[BaseModel]):
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format,
            )
            return response.choices[0].message.parsed
        except Exception as e:
            print(f"Error getting structured response (memory): {e}")
            return None

    def _extract_entity_mapping(self, prompt: str) -> Dict[str, str]:
        result = self._get_structured_response(prompt, EntityMappingResponse)
        if isinstance(result, EntityMappingResponse) and result.entity_map:
            return {
                entity.source_term: entity.target_term for entity in result.entity_map
            }
        return {}

    def _extract_discourse_connective(self, prompt: str) -> str:
        result = self._get_structured_response(prompt, DiscourseConnectiveResponse)
        if isinstance(result, DiscourseConnectiveResponse) and result.connective:
            return result.connective
        return ""

    def _extract_context_summary(self, prompt: str) -> str:
        result = self._get_structured_response(prompt, ContextSummaryResponse)
        if isinstance(result, ContextSummaryResponse) and result.summary:
            return result.summary
        return ""

    def get_local_memory(
        self, discourse: str, translation: str, source_lang: str, target_lang: str
    ) -> dict:
        local_memory = {
            "entity_mapping": {},
            "discourse_connectives": "",
            "context_summary": "",
        }

        for component in self.components:
            try:
                prompt = self.prompts[component.name].render(
                    source_discourse=discourse,
                    target_discourse=translation,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )

                if component.response_format == EntityMappingResponse:
                    content = self._extract_entity_mapping(prompt)
                elif component.response_format == DiscourseConnectiveResponse:
                    content = self._extract_discourse_connective(prompt)
                elif component.response_format == ContextSummaryResponse:
                    content = self._extract_context_summary(prompt)
                else:
                    content = {} if component.returns_mapping else ""

                key_name = (
                    f"{component.name}_mapping"
                    if component.returns_mapping
                    else component.name
                )
                local_memory[key_name] = content

            except Exception as e:
                print(f"Error getting memory components: {e}")
                pass

        return local_memory

    def get_incident_memory(
        self, memories: List[Dict[str, Any]], max_summary_history: int = 3
    ) -> Dict[str, Any]:
        combined = {
            "entity_mapping": {},
            "discourse_connectives": "",
            "context_summary": "",
        }

        context_summaries = []
        for mem in memories:
            combined["entity_mapping"].update(mem.get("entity_mapping", {}))
            combined["discourse_connectives"] = mem.get("discourse_connectives", "")
            if context := mem.get("context_summary", "").strip():
                context_summaries.append(context)

        recent_summaries = context_summaries[-max_summary_history:]
        combined["context_summary"] = " ".join(recent_summaries) or "(none)"
        return combined

    def encode_memory(self, memory: Dict[str, Any], max_summary_len: int = 200) -> str:
        """Encode memory into a string suited for LLM consumption"""
        parts = []
        for component in self.components:
            key_name = (
                f"{component.name}_mapping"
                if component.returns_mapping
                else component.name
            )
            content = memory.get(key_name)

            if not content:
                continue

            if component.returns_mapping and isinstance(content, dict):
                items = content.items()
                json_str = json.dumps(
                    dict(items), separators=(",", ":"), ensure_ascii=False
                )
                parts.append(f"{component.display_string}:{json_str}")
            elif isinstance(content, str):
                clean_content = content.strip()
                if (
                    component.name == "context_summary"
                    and len(clean_content) > max_summary_len
                ):
                    clean_content = "..." + clean_content[-max_summary_len:]
                parts.append(f"{component.display_string}:[{clean_content}]")
        return " | ".join(parts)
