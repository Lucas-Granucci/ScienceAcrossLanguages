import logging

from agents import (
    DependencyGraphAgent,
    MemoryAgent,
    TranslationAgent,
    TerminologyAgent,
    RAGAgent,
)
from langgraph.graph import END, StateGraph
from graph.state import DiscourseUnit, GraphState
from utils import document_to_sentences
from graph.state import DiscourseUnit, GraphState


class GraphBuilder:
    def __init__(
        self,
        dep_agent: DependencyGraphAgent,
        memory_agent: MemoryAgent,
        translation_agent: TranslationAgent,
    ):
        self.dep_agent = dep_agent
        self.memory_agent = memory_agent
        self.translation_agent = translation_agent
        self.modules: list[dict] = []
        self.logger = logging.getLogger(__name__)

    def _log(self, message: str) -> None:
        self.logger.info(message)

    def with_terminology(
        self, terminology_agent: TerminologyAgent, position="before:translate"
    ):
        self.modules.append(
            {"name": "terminology", "handler": terminology_agent, "position": position}
        )
        return self

    def with_rag(self, rag_agent: RAGAgent, position="before:translate"):
        self.modules.append({"name": "rag", "handler": rag_agent, "position": position})
        return self

    # ---- Build Helpers ----
    def _base_order(self):
        return ["dependency_graph", "prepare_memory", "translate", "next_segment"]

    def _apply_modules(self, order: list[str]) -> list[str]:
        new_order = list(order)
        for mod in self.modules:
            position = mod["position"]
            if ":" not in position:
                raise ValueError(
                    f"Position must be 'before:x' or 'afer:x', got {position}"
                )
            direction, anchor = position.split(":", 1)
            if anchor not in new_order:
                raise ValueError(f"Anchor '{anchor}' not in base order: {new_order}")
            idx = new_order.index(anchor)
            insert_at = idx if direction == "before" else idx + 1
            new_order.insert(insert_at, mod["name"])
        if len(new_order) != len(set(new_order)):
            raise ValueError(f"Duplicate node names in order: {new_order}")
        return new_order

    # ---- Node factories ----
    def _dependency_graph_node(self):
        dep_agent = self.dep_agent
        log = self._log

        def node(state: GraphState):
            log("dependency_graph: building dependency graph")
            sentences = document_to_sentences(state["source_document"])
            discourses, edges = dep_agent.generate_dependency_graph(sentences)

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

            log(
                f"dependency_graph: created {len(discourse_units)} discourses and {len(edges)} edges"
            )
            return {"discourses": discourse_units, "edges": edges, "current_index": 0}

        return node

    def _prepare_memory_node(self):
        memory_agent = self.memory_agent
        log = self._log

        def node(state: GraphState):
            idx = state["current_index"]
            incident_indices = [uid for uid, vid in state["edges"] if vid == idx]
            incident_memories = [
                state["discourses"][uid]["local_memory"] for uid in incident_indices
            ]
            incident_memory = memory_agent.get_incident_memory(incident_memories)

            # Update discourse
            state["discourses"][idx]["incident_memory"] = incident_memory
            log(f"prepare_memory: idx={idx}, incident_count={len(incident_indices)}")
            return {"discourses": state["discourses"]}

        return node

    def _terminology_node(self, terminology_agent: TerminologyAgent):
        log = self._log

        def node(state: GraphState):
            # TODO: Adjust based on agent
            log(f"terminology: idx={state['current_index']}")
            return {"discourses": state["discourses"]}

        return node

    def _rag_node(self, rag_agent: RAGAgent):
        log = self._log

        def node(state: GraphState):
            # TODO: Adjust based on agent
            log(f"rag: idx={state['current_index']}")
            return {"discourses": state["discourses"]}

        return node

    def _translate_node(self):
        translation_agent = self.translation_agent
        memory_agent = self.memory_agent
        log = self._log

        def node(state: GraphState):
            idx = state["current_index"]
            discourse = state["discourses"][idx]

            translation = translation_agent.translate(
                discourse=discourse["source_text"],
                memory=discourse["incident_memory"],
                terminology=discourse["terminology_context"],
                rag_snippets=discourse["rag_context"],
            )
            state["discourses"][idx]["target_text"] = translation

            # Generate local memory immediately
            local_mem = memory_agent.get_local_memory(
                discourse["source_text"], translation
            )
            state["discourses"][idx]["local_memory"] = local_mem
            memory_agent.reset_memory()

            log(
                f"translate: completed segment {idx + 1}/{len(state['discourses'])}"
            )
            return {"discourses": state["discourses"]}

        return node

    def _finalize_node(self):
        log = self._log

        def node(state: GraphState):
            translations = [d["target_text"] for d in state["discourses"]]
            translations = list(filter(None, translations))
            full_doc = " ".join(translations)
            log(f"finalize: concatenated {len(translations)} segments")
            return {"final_document": full_doc}

        return node

    def build(self):
        order = self._base_order()
        order = self._apply_modules(order)

        workflow = StateGraph(GraphState)

        # Node registry assembly
        workflow.add_node("dependency_graph", self._dependency_graph_node())
        workflow.add_node("prepare_memory", self._prepare_memory_node())
        workflow.add_node("translate", self._translate_node())
        workflow.add_node("finalize", self._finalize_node())

        module_handlers = {m["name"]: m for m in self.modules}

        for name in order:
            if name in ("dependency_graph", "prepare_memory", "translate"):
                continue
            mod = module_handlers.get(name)
            if not mod:
                continue
            name = mod["name"]
            if name == "terminology":
                workflow.add_node(name, self._terminology_node(mod["handler"]))
            elif name == "rag":
                workflow.add_node(name, self._rag_node(mod["handler"]))

        # Increment node
        def increment_node(state: GraphState):
            next_idx = state["current_index"] + 1
            return {"current_index": next_idx}

        workflow.add_node("next_segment", increment_node)

        # Entry point
        workflow.set_entry_point("dependency_graph")

        # Build edges for loop chain
        next_idx = order.index("next_segment")
        loop_chain = order[: next_idx + 1]
        for src, dest in zip(loop_chain, loop_chain[1:]):
            workflow.add_edge(src, dest)

        # Conditional branching
        loop_start = "prepare_memory"

        def check_done(state: GraphState):
            if state["current_index"] < len(state["discourses"]):
                return "process_segment"
            return "finalize"

        # Loop condition
        workflow.add_conditional_edges(
            "next_segment",
            check_done,
            {"process_segment": "prepare_memory", "finalize": "finalize"},
        )

        workflow.add_edge("finalize", END)
        return workflow.compile()
