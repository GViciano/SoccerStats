"""Separación automática de equipos por el color de la camiseta."""

from __future__ import annotations

from collections import Counter, defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans

UNKNOWN = -1


def jersey_color(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """Color medio (espacio Lab) del torso del jugador, ignorando el césped.

    Se toma la franja del 15 % al 50 % de la altura de la caja y el 50 % central
    de la anchura, que suele ser la camiseta.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1, cx2 = int(x1 + 0.25 * w), int(x2 - 0.25 * w)
    cy1, cy2 = int(y1 + 0.15 * h), int(y1 + 0.50 * h)
    H, W = frame.shape[:2]
    cx1, cx2 = max(cx1, 0), min(cx2, W)
    cy1, cy2 = max(cy1, 0), min(cy2, H)
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255)) > 0
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    keep = ~grass.reshape(-1)
    if keep.sum() < 4:
        return None
    return lab[keep].mean(axis=0)


class TeamClassifier:
    """Agrupa los colores de camiseta en dos equipos (KMeans, sin etiquetar nada).

    Los colores que quedan muy lejos de ambos equipos (árbitros, porteros con
    otra equipación) se marcan como UNKNOWN.
    """

    def __init__(self, outlier_factor: float = 3.0, min_threshold: float = 20.0,
                 random_state: int = 0):
        self.outlier_factor = outlier_factor
        self.min_threshold = min_threshold   # distancia Lab mínima para considerar "otro color"
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=random_state)
        self.threshold: float | None = None

    @property
    def fitted(self) -> bool:
        return self.threshold is not None

    def fit(self, colors: np.ndarray) -> "TeamClassifier":
        colors = np.asarray(colors, dtype=np.float32)
        if len(colors) < 10:
            raise ValueError("Pocas muestras de camisetas para separar equipos "
                             f"({len(colors)}); usa más frames de muestra.")
        self.kmeans.fit(colors)
        # Ordena los equipos de forma estable: equipo 0 = el más oscuro (menor L).
        order = np.argsort(self.kmeans.cluster_centers_[:, 0])
        self.kmeans.cluster_centers_ = self.kmeans.cluster_centers_[order]
        dists = self._distances(colors).min(axis=1)
        self.threshold = max(float(np.median(dists) * self.outlier_factor), self.min_threshold)
        return self

    def _distances(self, colors: np.ndarray) -> np.ndarray:
        c = self.kmeans.cluster_centers_
        return np.linalg.norm(colors[:, None, :] - c[None, :, :], axis=2)

    def predict(self, colors: np.ndarray) -> np.ndarray:
        if len(colors) == 0:
            return np.empty(0, dtype=int)
        d = self._distances(np.asarray(colors, dtype=np.float32))
        team = d.argmin(axis=1)
        team[d.min(axis=1) > self.threshold] = UNKNOWN
        return team

    def team_colors_bgr(self) -> list[tuple[int, int, int]]:
        """Color representativo de cada equipo, en BGR, para dibujar."""
        lab = self.kmeans.cluster_centers_.clip(0, 255).astype(np.uint8).reshape(1, -1, 3)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).reshape(-1, 3)
        return [tuple(int(v) for v in c) for c in bgr]


class TeamVoter:
    """Estabiliza el equipo de cada track_id con una votación acumulada.

    Así un frame con mala iluminación u oclusión no cambia de equipo al jugador.
    """

    def __init__(self) -> None:
        self.votes: dict[int, Counter] = defaultdict(Counter)

    def update(self, tracker_ids: np.ndarray, teams: np.ndarray) -> np.ndarray:
        out = np.empty(len(tracker_ids), dtype=int)
        for i, (tid, team) in enumerate(zip(tracker_ids, teams)):
            if tid < 0:
                out[i] = team
                continue
            self.votes[tid][int(team)] += 1
            out[i] = self.votes[tid].most_common(1)[0][0]
        return out


def assign_goalkeepers(gk_pitch_x: np.ndarray, player_pitch_x: np.ndarray,
                       player_teams: np.ndarray) -> np.ndarray:
    """Asigna cada portero al equipo cuyo centro de gravedad está más cerca.

    El portero suele vestir distinto, así que el color no sirve; pero siempre
    está en el lado del campo donde se coloca su equipo.
    """
    out = np.full(len(gk_pitch_x), UNKNOWN, dtype=int)
    means = {t: player_pitch_x[player_teams == t].mean()
             for t in (0, 1) if np.any(player_teams == t)}
    if len(means) < 2:
        return out
    for i, x in enumerate(gk_pitch_x):
        out[i] = min(means, key=lambda t: abs(means[t] - x))
    return out
