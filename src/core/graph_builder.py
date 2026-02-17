import json
import logging
from pathlib import Path
from typing import List, Optional

from langgraph.graph import END, StateGraph
from openai import OpenAI

from src.agents.dependency_graph_agent import DependencyGraphAgent
from src.agents.memory_agent import MemoryAgent
from src.agents.translation_agent import TranslationAgent
from src.core.state import DiscourseUnit, GraphState

logger = logging.getLogger(__name__)


class TranslationPipeline:
    def __init__(self, source_lang: str, target_lang: str, config: dict):
        self.config = config
        self.source_lang = source_lang
        self.target_lang = target_lang

        # Setup clients
        self.proc_client = OpenAI(
            base_url=config["processing"]["base_url"],
            api_key=config["processing"]["api_key"],
        )
        self.trans_client = OpenAI(
            base_url=config["translation"]["base_url"],
            api_key=config["translation"]["api_key"],
        )

        # Initialize core agents
        model_proc = config["processing"]["model_name"]
        model_trans = config["translation"]["model_name"]

        self.dep_agent = DependencyGraphAgent(self.proc_client, model_proc)
        self.mem_agent = MemoryAgent(self.proc_client, model_proc)
        self.trans_agent = TranslationAgent(self.trans_client, model_trans)

        # Initialize optional agents based on config
        self.active_modules = config.get("modules", [])

        self.term_agent = None
        if "terminology" in self.active_modules:
            # Implement terminology
            pass

        self.rag_agent = None
        if "rag" in self.active_modules:
            # Implement RAG agent
            pass

        self.app = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph(GraphState)

        # --- Add Nodes ---
        workflow.add_node("dependency_graph", self.node_dependency)
        workflow.add_node("prepare_memory", self.node_memory)
        workflow.add_node("translate", self.node_translate)
        workflow.add_node("finalize", self.node_finalize)

        if self.term_agent:
            workflow.add_node("terminology", self.node_terminology)
        if self.rag_agent:
            workflow.add_node("rag", self.node_rag)

        # --- Define edges ---
        workflow.set_entry_point("dependency_graph")
        workflow.add_edge("dependency_graph", "prepare_memory")
        current_node = "prepare_memory"

        if self.term_agent:
            workflow.add_edge(current_node, "terminology")
            current_node = "terminology"

        if self.rag_agent:
            workflow.add_edge(current_node, "rag")
            current_node = "rag"

        workflow.add_edge(current_node, "translate")

        workflow.add_conditional_edges(
            "translate",
            self._check_if_done,
            {"continue": "prepare_memory", "done": "finalize"},
        )
        workflow.add_edge("finalize", END)
        return workflow.compile()

    # ---- Node Implementations ----
    def node_dependency(self, state: GraphState):
        if state.get("discourses"):
            logger.info(f"Using {len(state['discourses'])} preloaded discourses")
            return {
                "discourses": state["discourses"],
                "edges": state["edges"],
                "current_index": 0,
            }

        sentences = state["source_sentences"]
        discourses, edges = self.dep_agent.generate_dependency_graph(sentences)

        units = [
            DiscourseUnit(
                id=i,
                source_text=txt,
                target_text=None,
                incident_memory={},
                local_memory={},
            )
            for i, txt in enumerate(discourses)
        ]

        logger.info(f"Created {len(units)} discourse units")
        return {
            "discourses": units,
            "edges": edges,
            "current_index": 0,
        }

    def node_memory(self, state: GraphState):
        idx = state["current_index"]
        incident_indices = [uid for uid, vid in state["edges"] if vid == idx]
        incident_mems = [
            state["discourses"][uid]["local_memory"] for uid in incident_indices
        ]
        aggregated_mem = self.mem_agent.get_incident_memory(incident_mems)
        state["discourses"][idx]["incident_memory"] = aggregated_mem
        return {"discourses": state["discourses"]}

    def node_terminology(self, state: GraphState):
        raise NotImplementedError

    def node_rag(self, state: GraphState):
        raise NotImplementedError

    def node_translate(self, state: GraphState):
        idx = state["current_index"]
        unit = state["discourses"][idx]

        memory_str = self.mem_agent.encode_memory(unit["incident_memory"])

        terminology_str = (
            self.term_agent.encode_terminology() if self.term_agent else None
        )
        rag_str = self.rag_agent.encode_rag() if self.rag_agent else None

        translation = self.trans_agent.translate(
            discourse=unit["source_text"],
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            memory_str=memory_str,
            terminology_str=terminology_str,
            rag_snippets_str=rag_str,
        )
        state["discourses"][idx]["target_text"] = translation

        # Extract local memory
        local_mem = self.mem_agent.get_local_memory(
            discourse=unit["source_text"],
            translation=translation,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
        )
        state["discourses"][idx]["local_memory"] = local_mem

        if (idx + 1) % 10 == 0:
            logger.info(f"Translated segment {idx + 1}/{len(state['discourses'])}")

        return {"discourses": state["discourses"], "current_index": idx + 1}

    def node_finalize(self, state: GraphState):
        translations = [d["target_text"] for d in state["discourses"]]
        translations = list(filter(None, translations))

        source_doc = " ".join(state["source_sentences"])
        target_doc = " ".join(translations)

        output_data = {
            "source_sentences": state["source_sentences"],
            "target_sentences": translations,
            "source_document": source_doc,
            "target_document": target_doc,
            "discourses": [d for d in state["discourses"]],
            "edges": state["edges"],
        }

        # Save to disk
        with open(state["graph_save_dir"], "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        return {"target_document": target_doc, "target_sentences": translations}

    def _check_if_done(self, state: GraphState):
        if state["current_index"] < len(state["discourses"]):
            return "continue"
        return "done"

    # ---- Helpers ----
    def run(
        self,
        source_sentences: List[str],
        graph_save_dir: Path,
        preloaded_state: Optional[dict] = None,
    ):

        initial_state = GraphState(
            source_sentences=source_sentences,
            discourses=(preloaded_state or {}).get("discourses") or [],
            edges=(preloaded_state or {}).get("edges") or [],
            current_index=0,
            target_document="",
            target_sentences=[],
            graph_save_dir=graph_save_dir,
        )

        return self.app.invoke(initial_state)

    @staticmethod
    def load_from_json(json_path: Path, swap_direction: bool = False) -> dict:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        discourses = []
        for d in data.get("discourses", []):
            source_txt = d.get("source_txt", "")
            target_txt = d.get("target_txt", "")

            final_source = target_txt if swap_direction else source_txt
            final_target = source_txt if swap_direction else target_txt

            discourses.append(
                DiscourseUnit(
                    id=d["idx"],
                    source_text=final_source,
                    target_text=final_target,
                    incident_memory={},
                    local_memory={},
                )
            )

        source_sentences = data.get("source_sentences", [])
        target_sentences = data.get("target_sentences", [])

        final_source_sents = target_sentences if swap_direction else source_sentences
        final_target_sents = source_sentences if swap_direction else target_sentences

        return {
            "source_sentences": final_source_sents,
            "target_sentences": final_target_sents,
            "discourses": discourses,
            "edges": data.get("edges", []),
        }
