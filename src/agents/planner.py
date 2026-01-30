from typing import List, Tuple

from .discourse import DiscourseAgent
from .edge import EdgeAgent


class PlannerAgent:
    def __init__(self, discourse_agent: DiscourseAgent, edge_agent: EdgeAgent):
        self.discourse_agent = discourse_agent
        self.edge_agent = edge_agent

    def plan(
        self, document_sentences: List[str]
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        print("Generating discourses...")
        discourses = self.discourse_agent(document_sentences)

        print("Generating edges...")
        edges = self.edge_agent(discourses)
        return discourses, edges
