"""Pipeline completo: vídeo -> detecciones -> tracks -> equipos -> campo 2D -> salidas."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from .detection import BALL, CLASS_NAMES, GOALKEEPER, PLAYER, REFEREE, Detections, FrameDetector, best_ball
from .pitch import Calibration, PitchCanvas
from .teams import UNKNOWN, TeamClassifier, TeamVoter, assign_goalkeepers, jersey_color

REFEREE_COLOR = (0, 215, 255)     # amarillo (BGR)
UNKNOWN_COLOR = (200, 200, 200)
BALL_COLOR = (255, 255, 255)

CSV_FIELDS = ["frame", "t", "track_id", "clase", "equipo", "conf",
              "x_px", "y_px", "x_m", "y_m"]


@dataclass
class PipelineConfig:
    stride: int = 1                 # procesa 1 de cada `stride` frames
    max_frames: int | None = None   # límite de frames (útil para pruebas rápidas)
    team_sample_frames: int = 30    # frames muestreados para aprender los colores
    minimap: bool = True
    minimap_scale: float = 4.0
    draw_ids: bool = True


def read_frames(path: str | Path, stride: int = 1, max_frames: int | None = None
                ) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se puede abrir el vídeo: {path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames is not None and idx >= max_frames):
                break
            if idx % stride == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def video_info(path: str | Path) -> tuple[float, int, int, int]:
    """(fps, ancho, alto, nº frames)."""
    cap = cv2.VideoCapture(str(path))
    info = (cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def fit_teams(video: str | Path, detector_factory: Callable[[], FrameDetector],
              n_frames: int, calibration: Calibration | None = None) -> TeamClassifier:
    """Primera pasada: aprende los dos colores de camiseta en frames repartidos."""
    _, _, _, total = video_info(video)
    stride = max(total // max(n_frames, 1), 1)
    detector = detector_factory()   # detector propio: no contamina el tracking de la 2.ª pasada
    colors = []
    for _, frame in read_frames(video, stride=stride, max_frames=stride * n_frames):
        dets = detector(frame)
        dets = dets[dets.class_id == PLAYER]
        if calibration is not None and len(dets):
            dets = dets[calibration.inside_pitch(calibration.to_pitch(dets.feet))]
        for box in dets.xyxy:
            c = jersey_color(frame, box)
            if c is not None:
                colors.append(c)
    return TeamClassifier().fit(np.asarray(colors))


def run(video: str | Path, output_dir: str | Path,
        detector_factory: Callable[[], FrameDetector],
        calibration: Calibration | None = None,
        config: PipelineConfig | None = None,
        progress: Callable[[int, int], None] | None = None) -> dict[str, Path]:
    """Procesa un vídeo y genera:

    * ``anotado.mp4``  vídeo con jugadores, equipos, IDs, balón y minimapa.
    * ``tracks.csv``   una fila por objeto y frame (base de las estadísticas).
    """
    config = config or PipelineConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fps, width, height, total = video_info(video)
    if config.max_frames:
        total = min(total, config.max_frames)

    teams = fit_teams(video, detector_factory, config.team_sample_frames, calibration)
    team_colors = teams.team_colors_bgr()
    voter = TeamVoter()
    detector = detector_factory()

    canvas = PitchCanvas(calibration.pitch, scale=config.minimap_scale) if calibration else None
    blank_map = canvas.blank() if canvas else None

    video_path, csv_path = out / "anotado.mp4", out / "tracks.csv"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / config.stride, (width, height))
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        rows = csv.writer(fh)
        rows.writerow(CSV_FIELDS)
        for idx, frame in read_frames(video, config.stride, config.max_frames):
            dets = detector(frame)
            people = dets[dets.class_id != BALL]
            ball = best_ball(dets)

            pitch_xy = calibration.to_pitch(people.feet) if calibration else None
            if calibration is not None and len(people):
                inside = calibration.inside_pitch(pitch_xy)   # descarta público y banquillos
                people, pitch_xy = people[inside], pitch_xy[inside]

            team = _assign_teams(frame, people, pitch_xy, teams, voter)
            ball_xy = calibration.to_pitch(ball.center) if (calibration and len(ball)) else None

            t = idx / fps
            for i in range(len(people)):
                _write_row(rows, idx, t, people, i, int(team[i]),
                           pitch_xy[i] if pitch_xy is not None else None)
            for i in range(len(ball)):
                _write_row(rows, idx, t, ball, i, UNKNOWN,
                           ball_xy[i] if ball_xy is not None else None, use_center=True)

            annotated = _annotate(frame, people, team, ball, team_colors, config.draw_ids)
            if canvas is not None:
                mini = _minimap(canvas, blank_map, pitch_xy, people, team, ball_xy, team_colors)
                annotated = _paste_minimap(annotated, mini)
            writer.write(annotated)
            if progress:
                progress(idx, total)
    writer.release()
    return {"video": video_path, "tracks": csv_path}


def _assign_teams(frame, people: Detections, pitch_xy, teams: TeamClassifier,
                  voter: TeamVoter) -> np.ndarray:
    team = np.full(len(people), UNKNOWN, dtype=int)
    is_player = people.class_id == PLAYER
    if is_player.any():
        colors = [jersey_color(frame, b) for b in people.xyxy[is_player]]
        raw = np.array([teams.predict(c[None])[0] if c is not None else UNKNOWN
                        for c in colors], dtype=int)
        team[is_player] = voter.update(people.tracker_id[is_player], raw)
    is_gk = people.class_id == GOALKEEPER
    if is_gk.any() and pitch_xy is not None:
        known = is_player & (team != UNKNOWN)
        gk = assign_goalkeepers(pitch_xy[is_gk, 0], pitch_xy[known, 0], team[known])
        team[is_gk] = voter.update(people.tracker_id[is_gk], gk)
    return team


def _write_row(rows, idx, t, dets: Detections, i, team, xy_m, use_center=False):
    px = (dets.center if use_center else dets.feet)[i]
    rows.writerow([idx, f"{t:.3f}", int(dets.tracker_id[i]), CLASS_NAMES[int(dets.class_id[i])],
                   team, f"{dets.confidence[i]:.3f}", f"{px[0]:.1f}", f"{px[1]:.1f}",
                   "" if xy_m is None else f"{xy_m[0]:.2f}",
                   "" if xy_m is None else f"{xy_m[1]:.2f}"])


def _color_for(cls: int, team: int, team_colors) -> tuple[int, int, int]:
    if cls == REFEREE:
        return REFEREE_COLOR
    if team == UNKNOWN:
        return UNKNOWN_COLOR
    return team_colors[team]


def _annotate(frame, people: Detections, team, ball: Detections, team_colors, draw_ids) -> np.ndarray:
    img = frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(people.xyxy):
        color = _color_for(int(people.class_id[i]), int(team[i]), team_colors)
        cx, w = int((x1 + x2) / 2), int(x2 - x1)
        cv2.ellipse(img, (cx, int(y2)), (max(w // 2, 6), max(w // 6, 3)), 0, -45, 235,
                    color, 2, cv2.LINE_AA)
        if draw_ids and people.tracker_id[i] >= 0:
            label = str(int(people.tracker_id[i]))
            if people.class_id[i] == GOALKEEPER:
                label = "GK " + label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            org = (cx - tw // 2, int(y2) + th + 8)
            cv2.rectangle(img, (org[0] - 3, org[1] - th - 3), (org[0] + tw + 3, org[1] + 3), color, -1)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    for x1, y1, x2, y2 in ball.xyxy:
        cx, top = int((x1 + x2) / 2), int(y1)
        tri = np.array([[cx, top - 4], [cx - 8, top - 18], [cx + 8, top - 18]], np.int32)
        cv2.fillPoly(img, [tri], BALL_COLOR)
        cv2.polylines(img, [tri], True, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def _minimap(canvas: PitchCanvas, blank, pitch_xy, people: Detections, team, ball_xy,
             team_colors) -> np.ndarray:
    img = blank.copy()
    if pitch_xy is not None and len(people):
        colors = [_color_for(int(c), int(t), team_colors) for c, t in zip(people.class_id, team)]
        canvas.draw_points(img, pitch_xy, colors)
    if ball_xy is not None and len(ball_xy):
        canvas.draw_points(img, ball_xy, BALL_COLOR, radius=max(3, int(canvas.scale * 0.6)))
    return img


def _paste_minimap(frame: np.ndarray, mini: np.ndarray, max_frac: float = 0.3) -> np.ndarray:
    """Pega el minimapa abajo en el centro, ocupando como mucho un 30 % del ancho."""
    H, W = frame.shape[:2]
    scale = min(max_frac * W / mini.shape[1], 1.0)
    mini = cv2.resize(mini, (int(mini.shape[1] * scale), int(mini.shape[0] * scale)))
    h, w = mini.shape[:2]
    x0, y0 = (W - w) // 2, H - h - 10
    if y0 < 0:
        return frame
    roi = frame[y0:y0 + h, x0:x0 + w]
    frame[y0:y0 + h, x0:x0 + w] = cv2.addWeighted(roi, 0.15, mini, 0.85, 0)
    return frame
