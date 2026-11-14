from dataclasses import dataclass


@dataclass
class Detection:
    class_name: str
    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2

        return cx, cy