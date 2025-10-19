from pydantic import BaseModel
from typing import Optional, Tuple

class DetectionEvent(BaseModel):
    type: str                # "choking", "bleeding", "unresponsive"
    confidence: float
    coords: Optional[Tuple[float, float]] = None  # (x, y) coordinates on frame
    frame_id: Optional[int] = None