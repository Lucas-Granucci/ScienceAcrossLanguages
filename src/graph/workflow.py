import re
from typing import List

from langgraph.graph import END, StateGraph

from agents import MemoryAgent, PlannerAgent, TranslationAgent
from graph.state import DiscourseUnit, GraphState


def document_to_sentences(document: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$", document)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


def create_translation_graph(
    planner_agent: PlannerAgent,
    memory_agent: MemoryAgent,
    translation_agent: TranslationAgent,
    # terminology_agent: TerminologyAgent,
    # rag_agent: RAGAgent,
    # reviewer_agent: ReviewerAgent
):
    # --- Node Definitions ---
    def planner_node(state: GraphState):
        sentences = document_to_sentences(state["source_document"])
        discourses, edges = planner_agent.plan(sentences)

        discourse_units = [
            DiscourseUnit(
                id=i,
                source_text=txt,
                target_text=None,
                incident_memory={},
                local_memory={},
                terminology_context={},
                rag_context=[],
            )
            for i, txt in enumerate(discourses)
        ]

        return {"discourses": discourse_units, "edges": edges, "current_index": 0}

    def prepare_memory_node(state: GraphState):
        idx = state["current_index"]

        # Calculate dependencies
        incident_indices = [uid for uid, vid in state["edges"] if vid == idx]
        incident_memories = [
            state["discourses"][uid]["local_memory"] for uid in incident_indices
        ]

        # Consolidate memory
        incident_memory = memory_agent.get_incident_memory(incident_memories)

        # Update discourse
        state["discourses"][idx]["incident_memory"] = incident_memory
        return {"discourses": state["discourses"]}

    def terminology_node(state: GraphState):
        return {"discourses": state["discourses"]}

    def rag_node(state: GraphState):
        return {"discourses": state["discourses"]}

    def translation_node(state: GraphState):
        idx = state["current_index"]
        discourse = state["discourses"][idx]

        # TODO: Maybe extract prompt creation logic?
        translation = translation_agent.translate(
            discourse=discourse["source_text"],
            memory=discourse["incident_memory"],
            terminology=discourse["terminology_context"],
            rag_snippets=discourse["rag_context"],
        )

        state["discourses"][idx]["target_text"] = translation

        # Generate local memory immediately
        local_mem = memory_agent.get_local_memory(discourse["source_text"], translation)
        state["discourses"][idx]["local_memory"] = local_mem

        memory_agent.reset_memory()

        return {"discourses": state["discourses"]}

    def increment_node(state: GraphState):
        return {"current_index": state["current_index"] + 1}

    def reviwer_node(state: GraphState):
        translations = [d["target_text"] for d in state["discourses"]]
        translations = list(filter(None, translations))
        full_doc = " ".join(translations)
        return {"final_document": full_doc}

    # --- Conditional Logic ---

    def check_done(state: GraphState):
        if state["current_index"] < len(state["discourses"]):
            return "process_segment"
        return "finalize"

    # --- Graph Construction ---

    workflow = StateGraph(GraphState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("prepare_memory", prepare_memory_node)
    workflow.add_node("terminology", terminology_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("translate", translation_node)
    workflow.add_node("next_segment", increment_node)
    workflow.add_node("reviwer", reviwer_node)

    # Entry
    workflow.set_entry_point("planner")

    # Planner -> Loop start
    workflow.add_edge("planner", "prepare_memory")

    # Segment processing chain
    workflow.add_edge("prepare_memory", "terminology")
    workflow.add_edge("terminology", "rag")
    workflow.add_edge("rag", "translate")
    workflow.add_edge("translate", "next_segment")

    # Loop condition
    workflow.add_conditional_edges(
        "next_segment",
        check_done,
        {"process_segment": "prepare_memory", "finalize": "reviwer"},
    )

    workflow.add_edge("reviwer", END)
    return workflow.compile()
