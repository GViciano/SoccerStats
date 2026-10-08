"""Calcula estadísticas a partir de tracks.csv (generado por analizar.py).

Ejemplo:
    python scripts/estadisticas.py salidas/partido/tracks.csv --calibracion config/calibracion.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soccerstats.pitch import Calibration  # noqa: E402
from soccerstats.stats import heatmap, load_tracks, player_summary, possession  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks")
    ap.add_argument("--calibracion", required=True, help="El mismo JSON usado en analizar.py")
    ap.add_argument("--salida", default=None, help="Carpeta de salida (por defecto, la del CSV)")
    ap.add_argument("--min-segundos", type=float, default=5.0,
                    help="Ignora tracks más cortos (fragmentos de tracking)")
    args = ap.parse_args()

    out = Path(args.salida or Path(args.tracks).parent)
    out.mkdir(parents=True, exist_ok=True)
    pitch = Calibration.from_json(args.calibracion).pitch
    df = load_tracks(args.tracks)

    summary = player_summary(df)
    summary = summary[summary["segundos"] >= args.min_segundos]
    summary.to_csv(out / "resumen_jugadores.csv", index=False)
    print(summary.to_string(index=False))

    pos = possession(df)
    if len(pos):
        print("\nPosesión (%):")
        for team, pct in pos.items():
            print(f"  Equipo {team}: {pct}")

    people = df[df["clase"].isin(["jugador", "portero"])]
    for team in (0, 1):
        cv2.imwrite(str(out / f"calor_equipo_{team}.png"), heatmap(people[people["equipo"] == team], pitch))
    cv2.imwrite(str(out / "calor_balon.png"), heatmap(df[df["clase"] == "balon"], pitch, sigma_m=1.5))
    print(f"\nResultados en {out}/")


if __name__ == "__main__":
    main()
