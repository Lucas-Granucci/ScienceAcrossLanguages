from typing import Any, Dict, List, Optional, TypedDict


class DiscourseUnit(TypedDict):
    id: int
    source_text: str
    target_text: Optional[str]
    incident_memory: Dict[str, Any]  # Memory from dependencies
    local_memory: Dict[str, Any]  # Memory generated from this unit

    # Additional context from modular agents
    terminology_context: Optional[Dict[str, str]]
    rag_context: Optional[List[str]]


class GraphState(TypedDict):
    source_document: str
    source_lang: str
    target_lang: str
    language_pair: str

    # Graph structure
    discourses: List[DiscourseUnit]
    edges: List[tuple[int, int]]  # (uid, vid)

    # Loop state
    current_index: int  # Pointer to current discourse being processed

    final_document: str
