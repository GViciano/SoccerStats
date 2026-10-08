"""Herramientas para calibrar la cámara (relacionar píxeles con metros del campo).

Uso:
    python scripts/calibrar.py puntos
        Lista los nombres de puntos válidos y guarda una imagen de referencia del campo.

    python scripts/calibrar.py frame VIDEO [--segundo 10] [--salida salidas/frame.jpg]
        Extrae un frame con una cuadrícula de píxeles para leer las coordenadas a ojo.

    python scripts/calibrar.py marcar VIDEO --salida config/calibracion.json
        (Solo en local, con ventana; requiere `pip install opencv-python`, no la versión
        headless) Haz clic en los puntos y escribe su nombre.

    python scripts/calibrar.py verificar VIDEO config/calibracion.json
        Dibuja las líneas del campo proyectadas sobre el frame para comprobar la calibración.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soccerstats.pitch import Calibration, PitchCanvas, PitchSpec  # noqa: E402


def grab_frame(video: str, second: float):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"No se pudo leer el segundo {second} de {video}")
    return frame


def draw_grid(frame, step: int = 100):
    img = frame.copy()
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (255, 255, 0), 1)
        cv2.putText(img, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (255, 255, 0), 1)
        cv2.putText(img, str(y), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    return img


def reference_image(pitch: PitchSpec, path: Path):
    canvas = PitchCanvas(pitch, scale=10, padding=60)
    img = canvas.blank()
    for name, xy in pitch.keypoints().items():
        p = tuple(canvas.to_px(xy))
        cv2.circle(img, p, 5, (0, 0, 255), -1)
        cv2.putText(img, name, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def cmd_puntos(args):
    pitch = PitchSpec.futbol7() if args.futbol7 else PitchSpec()
    for name, (x, y) in pitch.keypoints().items():
        print(f"  {name:22s} ({x:6.2f} m, {y:6.2f} m)")
    out = Path(args.salida)
    out.parent.mkdir(parents=True, exist_ok=True)
    reference_image(pitch, out)
    print(f"\nImagen de referencia guardada en {out}")


def cmd_frame(args):
    frame = grab_frame(args.video, args.segundo)
    out = Path(args.salida)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    grid = out.with_name(out.stem + "_cuadricula" + out.suffix)
    cv2.imwrite(str(grid), draw_grid(frame))
    print(f"Frame: {out}\nCon cuadrícula: {grid}")


def cmd_marcar(args):
    frame = grab_frame(args.video, args.segundo)
    pitch = PitchSpec.futbol7() if args.futbol7 else PitchSpec()
    names = list(pitch.keypoints())
    points: dict[str, list[int]] = {}
    clicks: list[tuple[int, int]] = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.namedWindow("calibrar", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibrar", on_click)
    print("Haz clic en un punto del campo y escribe su nombre en la consola. "
          "Pulsa 'q' en la ventana para terminar.")
    while True:
        view = frame.copy()
        for n, (x, y) in points.items():
            cv2.circle(view, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(view, n, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("calibrar", view)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if clicks:
            x, y = clicks.pop(0)
            name = input(f"Punto ({x}, {y}). Nombre ({', '.join(names)}): ").strip()
            if name in names:
                points[name] = [x, y]
            else:
                print("Nombre no válido, se ignora.")
    cv2.destroyAllWindows()
    data = {"campo": {"tipo": "futbol7" if args.futbol7 else "futbol11"}, "puntos": points}
    Path(args.salida).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado en {args.salida}")


def cmd_verificar(args):
    calib = Calibration.from_json(args.calibracion)
    frame = grab_frame(args.video, args.segundo)
    out = Path(args.salida)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), calib.draw_overlay(frame))
    err = calib.reprojection_error()
    print(f"Puntos usados: {len(calib.names)}  |  error medio: {err:.2f} m")
    if err > 1.0:
        print("El error es alto (> 1 m): revisa algún punto mal marcado o mal nombrado.")
    print(f"Imagen de verificación: {out}  (las líneas amarillas deben coincidir con las del campo)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("puntos")
    p.add_argument("--futbol7", action="store_true")
    p.add_argument("--salida", default="salidas/referencia_campo.png")
    p.set_defaults(func=cmd_puntos)

    p = sub.add_parser("frame")
    p.add_argument("video")
    p.add_argument("--segundo", type=float, default=5.0)
    p.add_argument("--salida", default="salidas/frame.jpg")
    p.set_defaults(func=cmd_frame)

    p = sub.add_parser("marcar")
    p.add_argument("video")
    p.add_argument("--segundo", type=float, default=5.0)
    p.add_argument("--futbol7", action="store_true")
    p.add_argument("--salida", default="config/calibracion.json")
    p.set_defaults(func=cmd_marcar)

    p = sub.add_parser("verificar")
    p.add_argument("video")
    p.add_argument("calibracion")
    p.add_argument("--segundo", type=float, default=5.0)
    p.add_argument("--salida", default="salidas/verificacion.jpg")
    p.set_defaults(func=cmd_verificar)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
