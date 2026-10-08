"""Analiza un vídeo: detección, tracking, equipos, campo 2D y vídeo anotado.

Ejemplo:
    python scripts/analizar.py datos/partido.mp4 \
        --calibracion config/calibracion.json \
        --modelo modelos/football-player-detection.pt \
        --salida salidas/partido
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soccerstats.detection import YoloTracker  # noqa: E402
from soccerstats.pipeline import PipelineConfig, run  # noqa: E402
from soccerstats.pitch import Calibration  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--calibracion", help="JSON de calibración (sin él no hay minimapa ni metros)")
    ap.add_argument("--modelo", default="yolo11l.pt",
                    help="Modelo YOLO: uno afinado para fútbol o uno genérico COCO")
    ap.add_argument("--tracker", default="botsort.yaml", choices=["botsort.yaml", "bytetrack.yaml"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--dispositivo", default=None, help="'0' para GPU, 'cpu' para CPU")
    ap.add_argument("--salida", default="salidas")
    ap.add_argument("--stride", type=int, default=1, help="Procesa 1 de cada N frames")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    calib = Calibration.from_json(args.calibracion) if args.calibracion else None
    if calib:
        print(f"Calibración: {len(calib.names)} puntos, error medio {calib.reprojection_error():.2f} m")

    def factory():
        return YoloTracker(args.modelo, tracker=args.tracker, conf=args.conf,
                           imgsz=args.imgsz, device=args.dispositivo)

    start = time.time()

    def progress(i, total):
        if i % 50 == 0:
            rate = (i + 1) / max(time.time() - start, 1e-6)
            print(f"  frame {i}/{total}  ({rate:.1f} fps)", flush=True)

    cfg = PipelineConfig(stride=args.stride, max_frames=args.max_frames)
    out = run(args.video, args.salida, factory, calib, cfg, progress)
    print(f"Listo en {time.time() - start:.0f} s")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
