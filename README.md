# ⚽ SoccerStats

Análisis de partidos de fútbol grabados con una **cámara fija**, usando visión por computador:

- **Detección** de jugadores, porteros, árbitros y balón (YOLO).
- **Tracking**: un ID por jugador a lo largo del vídeo (BoT-SORT / ByteTrack).
- **Separación de equipos** automática por el color de la camiseta (sin etiquetar nada).
- **Minimapa 2D**: posiciones reales en metros gracias a una calibración del campo (homografía).
- **Estadísticas**: distancia recorrida, velocidad media y máxima, sprints, posesión y mapas de calor.

Pensado para ejecutarse en **Google Colab** (GPU gratuita): abre
[`notebooks/SoccerStats_Colab.ipynb`](notebooks/SoccerStats_Colab.ipynb) y sigue los pasos.

## Cómo funciona

```
vídeo ──► YOLO + tracker ──► equipo por color ──► homografía (píxeles → metros) ──► salidas
           (cada frame)       (KMeans + votación     (calibración manual             ├─ anotado.mp4
                               por jugador)           de 4+ puntos)                   ├─ tracks.csv
                                                                                     └─ estadísticas
```

| Módulo | Qué hace |
|---|---|
| `soccerstats/detection.py` | Envoltorio de YOLO con tracking. Admite un modelo afinado para fútbol o uno genérico COCO. |
| `soccerstats/teams.py` | Color de camiseta (ignorando el césped), agrupación en 2 equipos, votación estable por ID y asignación de porteros. |
| `soccerstats/calibrador.py` | Calibración haciendo clic sobre la imagen en Colab, con las líneas del campo dibujadas al momento. |
| `soccerstats/pitch.py` | Medidas del campo (fútbol 11 y 7), puntos de referencia, homografía y dibujo del minimapa. |
| `soccerstats/pipeline.py` | Une todo: procesa el vídeo y escribe el vídeo anotado y `tracks.csv`. |
| `soccerstats/stats.py` | Distancias, velocidades, sprints, posesión y mapas de calor a partir de `tracks.csv`. |

## Uso por línea de comandos

```bash
pip install -r requirements.txt

# 1. Calibración (una vez por posición de cámara). En Colab es más cómodo con clics:
#    from soccerstats.calibrador import calibrar_con_clics
python scripts/calibrar.py puntos                       # nombres de puntos + imagen de referencia
python scripts/calibrar.py frame datos/partido.mp4      # frame con cuadrícula para leer píxeles
#    -> rellena config/calibracion.json (ver config/calibracion_ejemplo.json)
python scripts/calibrar.py verificar datos/partido.mp4 config/calibracion.json

# 2. Análisis
python scripts/analizar.py datos/partido.mp4 \
    --calibracion config/calibracion.json \
    --modelo modelos/football-player-detection.pt \
    --dispositivo 0 --salida salidas/partido

# 3. Estadísticas
python scripts/estadisticas.py salidas/partido/tracks.csv --calibracion config/calibracion.json
```

### Formato de `tracks.csv`

Una fila por objeto y frame:
`frame, t (s), track_id, clase (jugador/portero/arbitro/balon), equipo (0, 1 o -1 = desconocido), conf, x_px, y_px, x_m, y_m`.

Las coordenadas en metros tienen el origen en la esquina superior izquierda del campo (vista desde la cámara),
con el eje X a lo largo y el eje Y a lo ancho.

## Recomendaciones para grabar

- Cámara **elevada** (4–6 m o grada alta), en el **centro del lateral**, con **todo el campo** en imagen.
- **Fija**: trípode o soporte firme, sin zoom ni movimiento. Así la calibración vale para todo el partido.
- 1080p o más, 25–30 fps, buena luz y sin contraluz.

## Limitaciones conocidas

- Los IDs del tracking se cortan cuando un jugador queda tapado; un mismo jugador puede aparecer con varios IDs.
  Las estadísticas son por ID (`--min-segundos` filtra los fragmentos cortos).
- Con el modelo genérico COCO los árbitros y porteros no se distinguen de los jugadores.
- La zona más lejana a la cámara tiene menos precisión (los jugadores ocupan pocos píxeles).

## Tests

```bash
pytest -q
```

Los tests generan un vídeo sintético y usan un detector simulado, así que no necesitan GPU ni descargar modelos.

## Créditos

El modelo de detección afinado para fútbol que descarga el notebook procede del proyecto
[roboflow/sports](https://github.com/roboflow/sports). La detección usa
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (licencia AGPL-3.0).
