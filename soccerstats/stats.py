"""Estadísticas a partir de ``tracks.csv``: distancias, velocidades, posesión y mapas de calor."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .pitch import PitchCanvas, PitchSpec
from .teams import UNKNOWN

MAX_SPEED = 12.0        # m/s (~43 km/h): saltos mayores son errores de tracking
POSSESSION_RADIUS = 2.0  # m: distancia máxima jugador-balón para considerar posesión


def load_tracks(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.dropna(subset=["x_m", "y_m"])


def smooth_positions(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Suaviza la trayectoria de cada jugador con una mediana móvil (quita temblores)."""
    df = df.sort_values(["track_id", "frame"]).copy()
    for col in ("x_m", "y_m"):
        df[col] = df.groupby("track_id")[col].transform(
            lambda s: _symmetric_median(s.to_numpy(), window))
    return df


def _symmetric_median(v: np.ndarray, window: int) -> np.ndarray:
    """Mediana móvil centrada que se estrecha simétricamente en los extremos.

    Así los bordes no se desplazan (una ventana asimétrica falsearía velocidades).
    """
    half = window // 2
    n = len(v)
    out = np.empty(n, dtype=float)
    for i in range(n):
        k = min(half, i, n - 1 - i)
        out[i] = np.median(v[i - k:i + k + 1])
    return out


def player_summary(df: pd.DataFrame, smooth_window: int = 5, speed_window_s: float = 1.0
                   ) -> pd.DataFrame:
    """Una fila por track_id (jugador/portero) con su equipo, tiempo, distancia y velocidades."""
    people = df[df["clase"].isin(["jugador", "portero"]) & (df["track_id"] >= 0)]
    if people.empty:
        return pd.DataFrame(columns=["track_id", "clase", "equipo", "segundos", "distancia_m",
                                     "vel_media_kmh", "vel_max_kmh", "sprints"])
    people = smooth_positions(people, smooth_window)
    out = []
    for tid, g in people.groupby("track_id"):
        g = g.sort_values("t")
        dt = g["t"].diff().to_numpy()[1:]
        step = np.hypot(g["x_m"].diff(), g["y_m"].diff()).to_numpy()[1:]
        valid = (dt > 0) & (step / np.where(dt > 0, dt, np.inf) <= MAX_SPEED)
        dist = float(step[valid].sum())
        secs = float(g["t"].iloc[-1] - g["t"].iloc[0])

        # Velocidad sobre ventanas de ~1 s para que el ruido no infle el máximo.
        speeds = _windowed_speeds(g["t"].to_numpy(), g["x_m"].to_numpy(), g["y_m"].to_numpy(),
                                  speed_window_s)
        speeds = speeds[speeds <= MAX_SPEED]
        out.append({
            "track_id": int(tid),
            "clase": g["clase"].mode().iat[0],
            "equipo": int(g["equipo"].mode().iat[0]),
            "segundos": round(secs, 1),
            "distancia_m": round(dist, 1),
            "vel_media_kmh": round(dist / secs * 3.6, 1) if secs > 0 else 0.0,
            "vel_max_kmh": round(float(speeds.max()) * 3.6, 1) if len(speeds) else 0.0,
            "sprints": _count_sprints(speeds),
        })
    return pd.DataFrame(out).sort_values(["equipo", "distancia_m"], ascending=[True, False])


def _windowed_speeds(t, x, y, window_s) -> np.ndarray:
    j = np.searchsorted(t, t + window_s)
    j = np.clip(j, 0, len(t) - 1)
    dt = t[j] - t
    ok = dt >= window_s * 0.8
    return np.hypot(x[j] - x, y[j] - y)[ok] / dt[ok]


def _count_sprints(speeds: np.ndarray, threshold_kmh: float = 25.0) -> int:
    """Número de veces que se supera el umbral de sprint (flancos de subida)."""
    fast = speeds * 3.6 >= threshold_kmh
    return int(np.sum(fast[1:] & ~fast[:-1]) + (1 if len(fast) and fast[0] else 0))


def possession(df: pd.DataFrame, radius: float = POSSESSION_RADIUS) -> pd.Series:
    """Porcentaje de posesión por equipo.

    En cada frame, el balón es del jugador más cercano si está a menos de
    ``radius`` metros. Los frames sin dueño claro no cuentan.
    """
    ball = df[df["clase"] == "balon"][["frame", "x_m", "y_m"]]
    people = df[df["clase"].isin(["jugador", "portero"]) & (df["equipo"] != UNKNOWN)]
    m = people.merge(ball, on="frame", suffixes=("", "_b"))
    if m.empty:
        return pd.Series(dtype=float, name="posesion_%")
    m["d"] = np.hypot(m["x_m"] - m["x_m_b"], m["y_m"] - m["y_m_b"])
    owner = m.loc[m.groupby("frame")["d"].idxmin()]
    owner = owner[owner["d"] <= radius]
    if owner.empty:
        return pd.Series(dtype=float, name="posesion_%")
    pct = owner["equipo"].value_counts(normalize=True).sort_index() * 100
    return pct.round(1).rename("posesion_%")


def heatmap(df: pd.DataFrame, pitch: PitchSpec, scale: float = 8.0, sigma_m: float = 2.5
            ) -> np.ndarray:
    """Mapa de calor (imagen BGR) de las posiciones de ``df`` sobre el campo."""
    canvas = PitchCanvas(pitch, scale=scale)
    base = canvas.blank()
    acc = np.zeros(base.shape[:2], dtype=np.float32)
    if len(df):
        px = canvas.to_px(df[["x_m", "y_m"]].to_numpy())
        h, w = acc.shape
        ok = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        np.add.at(acc, (px[ok, 1], px[ok, 0]), 1.0)
        acc = cv2.GaussianBlur(acc, (0, 0), sigma_m * scale)
    if acc.max() > 0:
        # Raíz cuadrada: así un jugador quieto (p. ej. el portero) no apaga al resto.
        norm = (np.sqrt(acc / acc.max()) * 255).astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        alpha = (norm.astype(np.float32) / 255 * 0.75)[..., None]
        base = (base * (1 - alpha) + color * alpha).astype(np.uint8)
        # redibuja las líneas encima para que se sigan viendo
        for poly in pitch.lines():
            cv2.polylines(base, [canvas.to_px(poly)], False, (255, 255, 255),
                          max(1, int(scale / 4)), cv2.LINE_AA)
    return base
