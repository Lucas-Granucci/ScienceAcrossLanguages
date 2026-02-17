from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict


class DiscourseUnit(TypedDict):
    id: int
    source_text: str
    target_text: Optional[str]
    incident_memory: Dict[str, Any]  # Memory from dependencies
    local_memory: Dict[str, Any]  # Memory generated from this unit


class GraphState(TypedDict):
    source_sentences: List[str]

    # Graph structure
    discourses: List[DiscourseUnit]
    edges: List[tuple[int, int]]  # (uid, vid)
    current_index: int

    # Output
    target_document: str
    target_sentences: List[str]
    graph_save_dir: Path
