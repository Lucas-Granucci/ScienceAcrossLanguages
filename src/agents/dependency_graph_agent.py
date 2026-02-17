from pathlib import Path
from typing import List, Tuple

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm


class DiscourseDecision(BaseModel):
    decision: bool


class EdgeDecision(BaseModel):
    decision: bool


class DependencyGraphAgent:
    def __init__(
        self, client: OpenAI, model_name: str, max_discourse_length: int = 2048
    ):
        self.client = client
        self.model_name = model_name
        self.max_discourse_length = max_discourse_length

        prompts_dir = Path("config/prompts")
        env = Environment(loader=FileSystemLoader(prompts_dir))
        self.discourse_prompt_template = env.get_template("discourse.jinja")
        self.edge_prompt_template = env.get_template("edge.jinja")

    def generate_dependency_graph(
        self, document_sentences: List[str]
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        discourses = self._segment_discourses(document_sentences)
        edges = self._find_edges(discourses)
        return discourses, edges

    def _segment_discourses(self, document_sentences: List[str]) -> List[str]:
        discourses = []
        curr_sent_idx = 0
        sentences = document_sentences

        with tqdm(total=len(sentences), desc="Segmenting document") as pbar:
            while curr_sent_idx < len(sentences):
                discourse = [sentences[curr_sent_idx]]
                discourse_end_idx = curr_sent_idx + 1

                while discourse_end_idx < len(sentences):
                    if len(" ".join(discourse)) >= self.max_discourse_length:
                        break

                    try:
                        prompt = self.discourse_prompt_template.render(
                            discourse=" ".join(discourse),
                            next_sentence=sentences[discourse_end_idx],
                        )
                        response = self.client.beta.chat.completions.parse(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            response_format=DiscourseDecision,
                        )
                        result = response.choices[0].message.parsed
                        if result.decision:
                            discourse.append(sentences[discourse_end_idx])
                            discourse_end_idx += 1
                        else:
                            break
                    except Exception as e:
                        print(f"Error during segmentation: {e}")
                        break
                discourses.append(" ".join(discourse))
                pbar.update(discourse_end_idx - curr_sent_idx)
                curr_sent_idx = discourse_end_idx
        return discourses

    def _find_edges(self, discourses: List[str]) -> List[Tuple[str, str]]:
        edges = list()
        n = len(discourses)
        total_comparisons = (n * (n - 1)) // 2 + (n - 1)

        with tqdm(total=total_comparisons, desc="Finding edges") as pbar:
            for uid in range(len(discourses)):
                if uid < len(discourses) - 1:
                    edges.append((uid, uid + 1))
                    pbar.update(1)

                for vid in range(uid + 2, len(discourses)):
                    try:
                        prompt = self.edge_prompt_template.render(
                            discourse_1=discourses[uid][: self.max_discourse_length],
                            discourse_2=discourses[vid][: self.max_discourse_length],
                        )
                        response = self.client.beta.chat.completions.parse(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            response_format=EdgeDecision,
                        )
                        result = response.choices[0].message.parsed
                        if result:
                            if result.decision:
                                edges.append((uid, vid))

                    except Exception as e:
                        print(f"Error during edge generation: {e}")

                    finally:
                        pbar.update(1)
        return edges
