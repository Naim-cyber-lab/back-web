from typing import List
from app.core.es import es
from app.core.config import INDEX_WINKERS, INDEX_EVENTS
from app.schemas import RecommendedWinker, RecommendedEvent, WinkerOut, EventOut


def recommend_winkers_for_winker(winker_id: int, size: int = 10) -> List[RecommendedWinker]:
    return []

def recommend_events_for_winker(winker_id: int, size: int = 10) -> List[RecommendedEvent]:
    return []
