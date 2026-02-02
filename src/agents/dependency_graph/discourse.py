from typing import List

from openai import OpenAI
from pydantic import BaseModel

from utils import get_prompt_environment


class DiscourseDecision(BaseModel):
    decision: bool


class DiscourseAgent:
    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        language_pair: str,
        max_discourse_length: int = 2048,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.max_discourse_length = max_discourse_length
        self.language_pair = language_pair
        env = get_prompt_environment(language_pair)
        self.user_prompt_template = env.get_template("discourse_agent/user.jinja")

    def __call__(self, document_sentences: List[str]) -> List[str]:
        discourses = []
        curr_sent_idx = 0
        sentences = document_sentences
        while curr_sent_idx < len(sentences):
            discourse = [sentences[curr_sent_idx]]
            discourse_end_idx = curr_sent_idx + 1

            while discourse_end_idx < len(sentences):
                try:
                    prompt = self.user_prompt_template.render(
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
            curr_sent_idx = discourse_end_idx
        return discourses
