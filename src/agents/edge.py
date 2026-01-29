import os
from typing import List, Tuple

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI


class EdgeAgent:
    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        source_lang: str,
        target_lang: str,
        language_pair: str,
        max_discourse_length: int = 2048,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.max_discourse_length = max_discourse_length
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.language_pair = language_pair
        env = Environment(
            loader=FileSystemLoader(
                os.path.join(os.path.dirname(__file__), f"../prompts/{language_pair}")
            )
        )
        self.user_prompt_template = env.get_template("edge_agent/user.jinja")

    def __call__(self, discourses: List[str]) -> List[Tuple[int, int]]:
        edges = list()

        for uid in range(len(discourses)):
            u_discourse = discourses[uid]
            if uid < len(discourses) - 1:
                edges.append((uid, uid + 1))

            for vid in range(uid + 2, len(discourses)):
                v_discourse = discourses[vid]

                try:
                    prompt = self.user_prompt_template.render(
                        discourse_1=u_discourse, discourse_2=v_discourse
                    )
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        # max_completion_tokens=1,
                    )
                    message = response.choices[0].message.content
                    include_edge = message.strip().lower() if message else ""
                    print("edge message: ", include_edge)
                    if include_edge == "yes":
                        edges.append((uid, vid))
                except Exception as e:
                    print(f"Error during edge generation: {e}")
        return edges
