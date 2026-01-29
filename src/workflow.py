from openai import OpenAI
from tqdm import tqdm

from agents import DiscourseAgent, EdgeAgent, MemoryAgent, TranslationAgent
from processing import Discourse, Instance


def workflow(
    client: OpenAI,
    model_name: str,
    instance: Instance,
    source_lang: str,
    target_lang: str,
    language_pair: str,
) -> Instance:
    discourse_agent = DiscourseAgent(
        client, model_name, source_lang, target_lang, language_pair
    )
    edge_agent = EdgeAgent(client, model_name, source_lang, target_lang, language_pair)

    translation_agent = TranslationAgent(
        client, model_name, source_lang, target_lang, language_pair
    )

    memory_agent = MemoryAgent(client, model_name, language_pair)
    memory_agent.reset_memory()

    print("Generating discourses...")
    discourses = discourse_agent(instance.document_source_sentences)
    instance.discourses = [
        Discourse(
            source_txt=source_txt,
            target_txt=None,
            memory_incident=dict(),
            memory_local=dict(),
        )
        for source_txt in discourses
    ]

    print("Generating edges...")
    edges = edge_agent([discourse.source_txt for discourse in instance.discourses])
    instance.edges = edges

    translations = list()
    for did, discourse in tqdm(
        enumerate(instance.discourses), desc="Processing discourses..."
    ):
        # Generate incident memory
        incident_nodes = [uid for uid, vid in instance.edges if vid == did]
        incident_memories = [
            instance.discourses[uid].memory_local for uid in incident_nodes
        ]
        discourse.memory_incident = memory_agent.get_incident_memory(incident_memories)

        # Translate discourse
        translation = translation_agent.translate(
            discourse.source_txt, discourse.memory_incident
        )
        translations.append(translation)

        # Update local memory
        discourse.target_txt = translation
        discourse.memory_local = memory_agent.get_local_memory(
            discourse.source_txt, discourse.target_txt
        )
        memory_agent.reset_memory()

    instance.document_translation_output = " ".join(translations)
    return instance
