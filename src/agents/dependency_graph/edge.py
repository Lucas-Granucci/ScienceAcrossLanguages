from typing import List, Tuple

from openai import OpenAI
from pydantic import BaseModel

from utils import get_prompt_environment


class EdgeDecision(BaseModel):
    decision: bool


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
        env = get_prompt_environment(language_pair)
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
        return edges
