"""Tests sin GPU ni modelo: vídeo sintético + detector simulado."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soccerstats.detection import BALL, GOALKEEPER, PLAYER, REFEREE, Detections, best_ball
from soccerstats.pipeline import PipelineConfig, run
from soccerstats.pitch import Calibration, PitchSpec
from soccerstats.stats import heatmap, player_summary, possession
from soccerstats.teams import UNKNOWN, TeamClassifier, TeamVoter, assign_goalkeepers, jersey_color

W, H, FPS = 960, 540, 10
PITCH = PitchSpec()
# Cámara "virtual": dónde aparecen las cuatro esquinas del campo en la imagen.
CORNERS_IMG = np.float32([[180, 120], [780, 120], [-60, 500], [1020, 500]])
CORNERS_M = np.float32([[0, 0], [PITCH.length, 0], [0, PITCH.width], [PITCH.length, PITCH.width]])
H_M2IMG = cv2.getPerspectiveTransform(CORNERS_M, CORNERS_IMG)

RED, BLUE, YELLOW = (40, 40, 220), (220, 60, 30), (0, 220, 255)


def m2img(xy):
    return cv2.perspectiveTransform(np.float32(xy).reshape(-1, 1, 2), H_M2IMG).reshape(-1, 2)


def calibration_dict():
    names = ["esquina_sup_izq", "esquina_sup_der", "esquina_inf_izq", "esquina_inf_der",
             "centro", "area_izq_sup"]
    refs = PITCH.keypoints()
    pts = m2img([refs[n] for n in names])
    return {"campo": {"tipo": "futbol11"},
            "puntos": {n: [float(u), float(v)] for n, (u, v) in zip(names, pts)}}


@pytest.fixture
def calib(tmp_path):
    p = tmp_path / "calib.json"
    p.write_text(json.dumps(calibration_dict()))
    return Calibration.from_json(p)


# --- Escena sintética -----------------------------------------------------------------

def scene_objects(frame_idx):
    """Posiciones en metros: 5 rojos, 5 azules, 1 árbitro, 2 porteros y el balón."""
    t = frame_idx / FPS
    objs = []
    for i in range(5):
        objs.append((i + 1, PLAYER, RED, (20 + 6 * i + 3 * t, 10 + 11 * i)))
        objs.append((i + 11, PLAYER, BLUE, (60 + 6 * i - 3 * t, 12 + 11 * i)))
    objs.append((21, REFEREE, YELLOW, (52.5, 30 + t)))
    objs.append((31, GOALKEEPER, (30, 30, 30), (4, 34)))       # portero rojo (lado izquierdo)
    objs.append((32, GOALKEEPER, (200, 200, 200), (101, 34)))  # portero azul (lado derecho)
    # El balón va pegado al primer jugador rojo.
    objs.append((-1, BALL, (255, 255, 255), (20 + 3 * t + 0.5, 10)))
    return objs


def box_for(xy_m, cls):
    u, v = m2img([xy_m])[0]
    if cls == BALL:
        return np.float32([u - 3, v - 6, u + 3, v])
    h = 22 + 18 * (v - 120) / 380      # más grande cuanto más cerca de la cámara
    w = h * 0.4
    return np.float32([u - w / 2, v - h, u + w / 2, v])


def render(frame_idx):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (50, 140, 50)  # césped
    for line in PITCH.lines():
        cv2.polylines(img, [m2img(line).round().astype(np.int32)], False, (255, 255, 255), 1)
    for _, cls, color, xy in scene_objects(frame_idx):
        x1, y1, x2, y2 = box_for(xy, cls).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    # nº de frame en binario (bloques blancos/negros) para el detector simulado
    for bit in range(8):
        img[:16, bit * 16:(bit + 1) * 16] = 255 if (frame_idx >> bit) & 1 else 0
    return img


class FakeDetector:
    """Devuelve exactamente los objetos dibujados, con su ID, como haría YOLO + tracker."""

    def __call__(self, frame):
        idx = sum(1 << b for b in range(8) if frame[4:12, b * 16 + 4:b * 16 + 12].mean() > 127)
        objs = scene_objects(idx)
        return Detections(
            xyxy=np.stack([box_for(xy, c) for _, c, _, xy in objs]),
            confidence=np.full(len(objs), 0.9, np.float32),
            class_id=np.array([c for _, c, _, _ in objs]),
            tracker_id=np.array([tid for tid, _, _, _ in objs]),
        )


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "sintetico.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(40):
        wr.write(render(i))
    wr.release()
    return path


# --- Tests ----------------------------------------------------------------------------

def test_calibration_roundtrip(calib):
    assert calib.reprojection_error() < 0.05
    pts_m = np.float32([[10, 10], [52.5, 34], [90, 60]])
    back = calib.to_pitch(calib.to_image(pts_m))
    np.testing.assert_allclose(back, pts_m, atol=0.01)
    assert calib.inside_pitch(np.float32([[50, 30], [-10, 30], [50, 80]])).tolist() == [True, False, False]


def test_calibration_rejects_unknown_point(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"puntos": {"no_existe": [1, 2], "centro": [3, 4],
                                        "medio_sup": [5, 6], "medio_inf": [7, 8]}}))
    with pytest.raises(KeyError):
        Calibration.from_json(p)


def test_futbol7_keypoints_inside_pitch():
    p = PitchSpec.futbol7()
    for x, y in p.keypoints().values():
        assert 0 <= x <= p.length and 0 <= y <= p.width


def test_jersey_color_ignores_grass():
    img = np.zeros((100, 100, 3), np.uint8)
    img[:] = (50, 140, 50)
    img[20:50, 40:60] = RED
    c = jersey_color(img, np.float32([30, 10, 70, 90]))
    red_lab = cv2.cvtColor(np.uint8([[RED]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    assert np.linalg.norm(c - red_lab) < 5


def test_team_classifier_separates_and_flags_referee():
    rng = np.random.default_rng(0)
    lab = lambda bgr: cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    reds = lab(RED) + rng.normal(0, 3, (30, 3))
    blues = lab(BLUE) + rng.normal(0, 3, (30, 3))
    clf = TeamClassifier().fit(np.vstack([reds, blues]))
    pr, pb = clf.predict(reds), clf.predict(blues)
    assert len(set(pr)) == 1 and len(set(pb)) == 1 and pr[0] != pb[0]
    assert clf.predict(lab(YELLOW)[None])[0] == UNKNOWN


def test_team_voter_is_stable():
    v = TeamVoter()
    for _ in range(5):
        v.update(np.array([7]), np.array([1]))
    assert v.update(np.array([7]), np.array([0]))[0] == 1


def test_assign_goalkeepers():
    gk = assign_goalkeepers(np.array([3.0, 102.0]), np.array([30, 35, 70, 75.0]), np.array([0, 0, 1, 1]))
    assert gk.tolist() == [0, 1]


def test_best_ball_keeps_one():
    d = Detections(np.zeros((3, 4), np.float32), np.float32([0.3, 0.8, 0.9]),
                   np.array([BALL, BALL, PLAYER]), np.array([-1, -1, 4]))
    b = best_ball(d)
    assert len(b) == 1 and b.confidence[0] == pytest.approx(0.8)


def test_pipeline_end_to_end(video, calib, tmp_path):
    out = run(video, tmp_path / "out", FakeDetector, calib,
              PipelineConfig(team_sample_frames=5))
    assert out["video"].stat().st_size > 0
    cap = cv2.VideoCapture(str(out["video"]))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 40
    cap.release()

    df = pd.read_csv(out["tracks"])
    assert set(df["clase"]) == {"jugador", "portero", "arbitro", "balon"}
    # Posiciones en metros correctas (jugador 1 en el frame 0 está en (20, 10)).
    row = df[(df["frame"] == 0) & (df["track_id"] == 1)].iloc[0]
    assert row["x_m"] == pytest.approx(20, abs=0.5) and row["y_m"] == pytest.approx(10, abs=0.5)
    # Rojos y azules en equipos distintos, consistentes en todo el vídeo.
    team_of = df[df["clase"] == "jugador"].groupby("track_id")["equipo"].agg(set)
    reds = set().union(*team_of.loc[1:5])
    blues = set().union(*team_of.loc[11:15])
    assert len(reds) == 1 and len(blues) == 1 and reds != blues and UNKNOWN not in reds | blues
    # Cada portero con el equipo de su lado.
    gk = df[df["clase"] == "portero"].groupby("track_id")["equipo"].first()
    assert gk[31] in reds and gk[32] in blues


def test_stats(video, calib, tmp_path):
    out = run(video, tmp_path / "out", FakeDetector, calib, PipelineConfig(team_sample_frames=5))
    df = pd.read_csv(out["tracks"]).dropna(subset=["x_m"])
    summary = player_summary(df).set_index("track_id")
    # 3 m/s durante 3,9 s -> ~11,7 m y ~10,8 km/h.
    assert summary.loc[1, "distancia_m"] == pytest.approx(11.7, abs=0.6)
    assert summary.loc[1, "vel_max_kmh"] == pytest.approx(10.8, abs=1.0)
    assert summary.loc[31, "distancia_m"] == pytest.approx(0, abs=0.1)
    pos = possession(df)
    red_team = int(df[df["track_id"] == 1]["equipo"].iloc[0])
    assert pos[red_team] == pytest.approx(100.0)
    img = heatmap(df[df["clase"] == "jugador"], calib.pitch, scale=4)
    assert img.ndim == 3 and img.mean() > 0


def test_click_calibrator(tmp_path):
    from soccerstats.calibrador import ClickCalibrator, build_html

    out = tmp_path / "calib.json"
    cal = ClickCalibrator(out, (W, H))
    refs = PITCH.keypoints()
    names = ["esquina_sup_izq", "esquina_sup_der", "centro"]
    for n in names:
        u, v = m2img([refs[n]])[0]
        state = cal.set_point(n, u, v)
    assert state["error"] is None and state["lineas"] == []   # con 3 puntos aún no hay campo
    u, v = m2img([refs["area_izq_sup"]])[0]
    state = cal.set_point("area_izq_sup", u, v)
    assert state["error"] < 0.1 and len(state["lineas"]) > 0
    # Lo guardado en disco vale directamente para analizar.
    assert Calibration.from_json(out).reprojection_error() < 0.1
    # Borrar un punto y retomar desde el fichero.
    cal.set_point("centro", None, None)
    assert set(ClickCalibrator(out, (W, H)).points) == {"esquina_sup_izq", "esquina_sup_der", "area_izq_sup"}
    assert "__IMG__" not in build_html(render(0), cal, "cb")
