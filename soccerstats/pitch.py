"""Geometría del campo: medidas, puntos de referencia, homografía y minimapa.

Sistema de coordenadas del campo (en metros):
    origen (0, 0) en la esquina superior izquierda vista desde la cámara,
    eje X a lo largo del campo (de portería a portería, 0..largo),
    eje Y a lo ancho (banda superior = 0, banda inferior = ancho).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PitchSpec:
    """Medidas del terreno de juego en metros."""

    length: float = 105.0
    width: float = 68.0
    box_depth: float = 16.5        # profundidad del área grande
    box_width: float = 40.32       # anchura del área grande
    small_box_depth: float = 5.5
    small_box_width: float = 18.32
    penalty_spot: float = 11.0
    center_radius: float = 9.15

    @classmethod
    def futbol7(cls) -> "PitchSpec":
        """Medidas habituales de fútbol 7 (pueden variar según el campo)."""
        return cls(length=65.0, width=45.0, box_depth=13.0, box_width=26.0,
                   small_box_depth=4.0, small_box_width=12.0, penalty_spot=9.0,
                   center_radius=6.0)

    def keypoints(self) -> dict[str, tuple[float, float]]:
        """Puntos de referencia con nombre, útiles para calibrar la cámara."""
        L, W = self.length, self.width
        by0, by1 = (W - self.box_width) / 2, (W + self.box_width) / 2
        sy0, sy1 = (W - self.small_box_width) / 2, (W + self.small_box_width) / 2
        return {
            "esquina_sup_izq": (0.0, 0.0),
            "esquina_sup_der": (L, 0.0),
            "esquina_inf_izq": (0.0, W),
            "esquina_inf_der": (L, W),
            "medio_sup": (L / 2, 0.0),
            "medio_inf": (L / 2, W),
            "centro": (L / 2, W / 2),
            "circulo_sup": (L / 2, W / 2 - self.center_radius),
            "circulo_inf": (L / 2, W / 2 + self.center_radius),
            "area_izq_linea_sup": (0.0, by0),
            "area_izq_linea_inf": (0.0, by1),
            "area_izq_sup": (self.box_depth, by0),
            "area_izq_inf": (self.box_depth, by1),
            "area_der_linea_sup": (L, by0),
            "area_der_linea_inf": (L, by1),
            "area_der_sup": (L - self.box_depth, by0),
            "area_der_inf": (L - self.box_depth, by1),
            "area_peq_izq_sup": (self.small_box_depth, sy0),
            "area_peq_izq_inf": (self.small_box_depth, sy1),
            "area_peq_der_sup": (L - self.small_box_depth, sy0),
            "area_peq_der_inf": (L - self.small_box_depth, sy1),
            "penalti_izq": (self.penalty_spot, W / 2),
            "penalti_der": (L - self.penalty_spot, W / 2),
        }

    def lines(self) -> list[np.ndarray]:
        """Líneas del campo como polilíneas en metros (para dibujar)."""
        L, W = self.length, self.width
        by0, by1 = (W - self.box_width) / 2, (W + self.box_width) / 2
        sy0, sy1 = (W - self.small_box_width) / 2, (W + self.small_box_width) / 2
        bd, sd = self.box_depth, self.small_box_depth
        polys = [
            [(0, 0), (L, 0), (L, W), (0, W), (0, 0)],
            [(L / 2, 0), (L / 2, W)],
            [(0, by0), (bd, by0), (bd, by1), (0, by1)],
            [(L, by0), (L - bd, by0), (L - bd, by1), (L, by1)],
            [(0, sy0), (sd, sy0), (sd, sy1), (0, sy1)],
            [(L, sy0), (L - sd, sy0), (L - sd, sy1), (L, sy1)],
        ]
        t = np.linspace(0, 2 * np.pi, 64)
        circle = np.stack([L / 2 + self.center_radius * np.cos(t),
                           W / 2 + self.center_radius * np.sin(t)], axis=1)
        return [np.asarray(p, dtype=np.float32) for p in polys] + [circle.astype(np.float32)]


@dataclass
class Calibration:
    """Relación imagen <-> campo calculada a partir de puntos marcados a mano."""

    pitch: PitchSpec
    image_points: np.ndarray            # (N, 2) píxeles
    pitch_points: np.ndarray            # (N, 2) metros
    names: list[str] = field(default_factory=list)
    H: np.ndarray = field(init=False)   # imagen -> campo
    H_inv: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if len(self.image_points) < 4:
            raise ValueError(
                f"Hacen falta al menos 4 puntos rellenados para calibrar el campo y hay "
                f"{len(self.image_points)}. Sustituye los [0, 0] de la plantilla por las "
                "coordenadas (x, y) en píxeles de cada punto en el frame con cuadrícula.")
        H, _ = cv2.findHomography(self.image_points.astype(np.float32),
                                  self.pitch_points.astype(np.float32), method=0)
        if H is None:
            raise ValueError("No se pudo calcular la homografía: revisa que los puntos "
                             "no estén alineados en una misma recta.")
        self.H = H
        self.H_inv = np.linalg.inv(H)

    @classmethod
    def from_json(cls, path: str | Path) -> "Calibration":
        """Carga un fichero como `config/calibracion_ejemplo.json`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        pitch = pitch_from_config(data.get("campo", {}))
        refs = pitch.keypoints()
        names, img, world = [], [], []
        for name, xy in data["puntos"].items():
            if name.startswith("_") or xy is None or list(xy) == [0, 0]:
                continue  # [0, 0] = punto de la plantilla sin rellenar
            if name not in refs:
                raise KeyError(f"Punto desconocido '{name}'. Válidos: {', '.join(refs)}")
            names.append(name)
            img.append(xy)
            world.append(refs[name])
        return cls(pitch, np.asarray(img, dtype=np.float32),
                   np.asarray(world, dtype=np.float32), names)

    def to_pitch(self, points: np.ndarray) -> np.ndarray:
        """Píxeles (N, 2) -> metros (N, 2)."""
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, points: np.ndarray) -> np.ndarray:
        """Metros (N, 2) -> píxeles (N, 2)."""
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    def reprojection_error(self) -> float:
        """Error medio (en metros) al proyectar los puntos marcados."""
        return float(np.linalg.norm(self.to_pitch(self.image_points) - self.pitch_points,
                                    axis=1).mean())

    def inside_pitch(self, pitch_xy: np.ndarray, margin: float = 3.0) -> np.ndarray:
        """Máscara de puntos dentro del campo (con margen para jugadores en la banda)."""
        x, y = pitch_xy[:, 0], pitch_xy[:, 1]
        return ((x >= -margin) & (x <= self.pitch.length + margin)
                & (y >= -margin) & (y <= self.pitch.width + margin))

    def draw_overlay(self, frame: np.ndarray, color=(0, 255, 255)) -> np.ndarray:
        """Dibuja las líneas del campo proyectadas sobre el frame (para verificar)."""
        out = frame.copy()
        for line in self.pitch.lines():
            dense = _densify(line, step=0.5)
            pts = self.to_image(dense).round().astype(np.int32)
            cv2.polylines(out, [pts], isClosed=False, color=color, thickness=2,
                          lineType=cv2.LINE_AA)
        for (u, v), name in zip(self.image_points, self.names):
            cv2.circle(out, (int(u), int(v)), 6, (0, 0, 255), -1)
            cv2.putText(out, name, (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return out


def pitch_from_config(campo: dict) -> PitchSpec:
    """Construye las medidas del campo a partir del bloque "campo" de la calibración."""
    pitch = PitchSpec.futbol7() if campo.get("tipo") == "futbol7" else PitchSpec()
    overrides = {k: v for k, v in campo.items() if k in PitchSpec.__dataclass_fields__}
    return PitchSpec(**{**pitch.__dict__, **overrides}) if overrides else pitch


def _densify(poly: np.ndarray, step: float) -> np.ndarray:
    """Añade puntos intermedios para que la perspectiva curve bien las líneas."""
    out = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(int(np.linalg.norm(b - a) / step), 1)
        for i in range(1, n + 1):
            out.append(a + (b - a) * i / n)
    return np.asarray(out, dtype=np.float32)


class PitchCanvas:
    """Dibuja un campo 2D visto desde arriba y convierte metros a píxeles."""

    def __init__(self, pitch: PitchSpec, scale: float = 8.0, padding: int = 30):
        self.pitch = pitch
        self.scale = scale
        self.padding = padding
        self.size = (int(pitch.length * scale) + 2 * padding,
                     int(pitch.width * scale) + 2 * padding)  # (ancho, alto)

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        return (np.asarray(xy, dtype=np.float32) * self.scale + self.padding).round().astype(np.int32)

    def blank(self, grass=(60, 140, 60), line=(255, 255, 255)) -> np.ndarray:
        img = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        img[:] = grass
        thickness = max(1, int(self.scale / 4))
        for poly in self.pitch.lines():
            cv2.polylines(img, [self.to_px(poly)], isClosed=False, color=line,
                          thickness=thickness, lineType=cv2.LINE_AA)
        for spot in (self.pitch.keypoints()["penalti_izq"], self.pitch.keypoints()["penalti_der"],
                     self.pitch.keypoints()["centro"]):
            cv2.circle(img, tuple(self.to_px(spot)), thickness + 1, line, -1)
        return img

    def draw_points(self, img: np.ndarray, xy: np.ndarray, colors, labels=None,
                    radius: int | None = None) -> np.ndarray:
        radius = radius or max(4, int(self.scale * 0.9))
        for i, p in enumerate(self.to_px(xy)):
            color = colors[i] if isinstance(colors, (list, np.ndarray)) else colors
            cv2.circle(img, tuple(p), radius, tuple(int(c) for c in color), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p), radius, (0, 0, 0), 1, cv2.LINE_AA)
            if labels is not None:
                cv2.putText(img, str(labels[i]), (p[0] - radius, p[1] - radius - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        return img
