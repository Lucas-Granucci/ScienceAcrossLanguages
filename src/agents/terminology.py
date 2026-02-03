from typing import List, Optional, Dict
from openai import OpenAI
from utils import get_prompt_environment
from pydantic import BaseModel
import requests


class TermsResponse(BaseModel):
    terms: List[str]


class WikidataGlossary:
    def __init__(self):
        self.endpoint_url = "https://www.wikidata.org/w/api.php"
        self.headers = {"User-Agent": "ScienceAcrossLanguages (029725@mtka.org)"}

    def get_translation(
        self, term: str, target_lang_code: str
    ) -> Optional[Dict[str, str]]:
        search_params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": term,
            "limit": 1,
        }

        try:
            r = requests.get(
                self.endpoint_url,
                params=search_params,
                headers=self.headers,
            )
            r.raise_for_status()
            data = r.json()
            search_results = data.get("search", [])

            if not search_results:
                return

            first_result = search_results[0]
            entity_id = first_result["id"]

            entity_params = {
                "action": "wbgetentities",
                "format": "json",
                "ids": entity_id,
                "props": "labels|descriptions",
            }

            r = requests.get(self.endpoint_url, entity_params, headers=self.headers)
            r.raise_for_status()
            entity_data = r.json()
            entity = entity_data.get("entities", {}).get(entity_id, {})
            labels = entity.get("labels", {})
            descriptions = entity.get("descriptions", {})

            if target_lang_code in labels:
                translation = labels[target_lang_code]["value"]
                description = descriptions.get(target_lang_code, {}).get(
                    "value",
                    descriptions.get("en", {}).get("value", "No definition found."),
                )

                return {
                    "term": term,
                    "translation": translation,
                    "definition": description,
                }

            else:
                return None

        except Exception as e:
            print(f"Wikidata lookup failed for {term}: {e}")
            return None


class TerminologyAgent:
    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        language_pair: str,
    ) -> None:
        self.client = client
        self.model_name = model_name
        env = get_prompt_environment(language_pair)
        self.user_prompt_template = env.get_template("terminology/user.jinja")
        self.wikidata = WikidataGlossary()

    def extract_terms(self, discourse: str, target_lang_code: str) -> str:
        try:
            prompt = self.user_prompt_template.render(source_discourse=discourse)
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=TermsResponse,
            )
            extracted_terms = response.choices[0].message.parsed.terms
            glossary_entries = []

            for term in extracted_terms:
                wiki_result = self.wikidata.get_translation(term, target_lang_code)
                if wiki_result:
                    glossary_entries.append(wiki_result)

        except Exception as e:
            print(f"Error during term extraction: {e}")
            return {}
