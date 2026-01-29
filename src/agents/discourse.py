import os
from typing import List

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI


class DiscourseAgent:
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
                    print("Getting discourse response...")
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        # max_completion_tokens=1,
                    )
                    include_sentence = (
                        response.choices[0].message.content.strip().lower()
                    )
                    print("discourse_message: ", include_sentence)
                    if include_sentence == "yes":
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
