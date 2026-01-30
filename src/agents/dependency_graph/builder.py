from typing import List, Tuple

from openai import OpenAI

from .discourse import DiscourseAgent
from .edge import EdgeAgent


class DependencyGraphAgent:
    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        source_lang: str,
        target_lang: str,
        language_pair: str,
    ):
        self.discourse_agent = DiscourseAgent(
            client, model_name, source_lang, target_lang, language_pair
        )
        self.edge_agent = EdgeAgent(
            client, model_name, source_lang, target_lang, language_pair
        )

    def generate_dependency_graph(
        self, document_sentences: List[str]
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        print("Generating discourses...")
        discourses = self.discourse_agent(document_sentences)

        print("Generating edges...")
        edges = self.edge_agent(discourses)
        return discourses, edges
