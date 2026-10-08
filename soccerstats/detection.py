"""Detección y tracking de jugadores, porteros, árbitros y balón con YOLO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

# Clases internas del proyecto (coinciden con el orden del modelo de fútbol de Roboflow).
BALL, GOALKEEPER, PLAYER, REFEREE = 0, 1, 2, 3
CLASS_NAMES = {BALL: "balon", GOALKEEPER: "portero", PLAYER: "jugador", REFEREE: "arbitro"}

# Cómo se traducen los nombres de clase de cada modelo a las clases internas.
_NAME_MAP = {
    "ball": BALL, "sports ball": BALL, "balon": BALL,
    "goalkeeper": GOALKEEPER, "portero": GOALKEEPER,
    "player": PLAYER, "person": PLAYER, "jugador": PLAYER,
    "referee": REFEREE, "arbitro": REFEREE,
}


@dataclass
class Detections:
    """Detecciones de un frame. Todas las matrices tienen N filas."""

    xyxy: np.ndarray                # (N, 4) cajas en píxeles
    confidence: np.ndarray          # (N,)
    class_id: np.ndarray            # (N,) clases internas
    tracker_id: np.ndarray          # (N,) -1 si no tiene ID

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.empty((0, 4), np.float32), np.empty(0, np.float32),
                   np.empty(0, int), np.empty(0, int))

    def __len__(self) -> int:
        return len(self.xyxy)

    def __getitem__(self, mask) -> "Detections":
        return Detections(self.xyxy[mask], self.confidence[mask],
                          self.class_id[mask], self.tracker_id[mask])

    @property
    def feet(self) -> np.ndarray:
        """Punto de apoyo en el suelo: centro del borde inferior de la caja."""
        return np.stack([(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, self.xyxy[:, 3]], axis=1)

    @property
    def center(self) -> np.ndarray:
        return np.stack([(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2,
                         (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2], axis=1)


class FrameDetector(Protocol):
    def __call__(self, frame: np.ndarray) -> Detections: ...


class YoloTracker:
    """Detecta y sigue objetos con Ultralytics YOLO (+ BoT-SORT o ByteTrack).

    Funciona con:
      * un modelo afinado para fútbol (clases ball/goalkeeper/player/referee), o
      * un modelo genérico COCO (person -> jugador, sports ball -> balón).
    """

    def __init__(self, model_path: str = "yolo11l.pt", tracker: str = "botsort.yaml",
                 conf: float = 0.25, imgsz: int = 1280, device: str | None = None):
        from ultralytics import YOLO  # importación diferida: pesada y opcional en tests

        self.model = YOLO(model_path)
        self.tracker = tracker
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        names = {i: n.lower() for i, n in self.model.names.items()}
        self.class_map = {i: _NAME_MAP[n] for i, n in names.items() if n in _NAME_MAP}
        if not self.class_map:
            raise ValueError(f"El modelo no tiene clases reconocibles: {names}")
        self.is_soccer_model = any(n in ("player", "goalkeeper", "referee") for n in names.values())

    def __call__(self, frame: np.ndarray) -> Detections:
        result = self.model.track(frame, persist=True, tracker=self.tracker, conf=self.conf,
                                  imgsz=self.imgsz, device=self.device,
                                  classes=list(self.class_map), verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return Detections.empty()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = (boxes.id.cpu().numpy().astype(int) if boxes.id is not None
               else np.full(len(cls), -1, dtype=int))
        return Detections(
            xyxy=boxes.xyxy.cpu().numpy().astype(np.float32),
            confidence=boxes.conf.cpu().numpy().astype(np.float32),
            class_id=np.array([self.class_map[c] for c in cls], dtype=int),
            tracker_id=ids,
        )


def best_ball(dets: Detections) -> Detections:
    """Se queda solo con la detección de balón más segura (hay un único balón)."""
    balls = dets[dets.class_id == BALL]
    if len(balls) <= 1:
        return balls
    return balls[np.array([int(np.argmax(balls.confidence))])]
