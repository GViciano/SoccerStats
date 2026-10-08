"""Calibración del campo haciendo clic sobre la imagen (pensado para Google Colab).

Uso en una celda de Colab:

    from soccerstats.calibrador import calibrar_con_clics
    calibrar_con_clics(VIDEO, CALIB_PATH, segundo=2)

Se elige un punto en la lista, se hace clic donde está en la imagen y, en cuanto
hay 4 puntos, se dibujan encima las líneas del campo calculadas para comprobar
que encajan. Cada clic se guarda automáticamente en el JSON de calibración.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from .pitch import Calibration, PitchSpec, _densify, pitch_from_config


class ClickCalibrator:
    """Lógica de la calibración por clics (independiente de Colab, para poder testearla)."""

    def __init__(self, output_json: str | Path, image_size: tuple[int, int],
                 campo: dict | None = None):
        self.output_json = Path(output_json)
        self.width, self.height = image_size
        self.campo = campo or {"tipo": "futbol11"}
        self.pitch: PitchSpec = pitch_from_config(self.campo)
        self.points: dict[str, list[float]] = {}
        if self.output_json.exists():   # retoma una calibración anterior
            try:
                prev = json.loads(self.output_json.read_text(encoding="utf-8"))
                if prev.get("campo", {}).get("tipo", "futbol11") == self.campo.get("tipo", "futbol11"):
                    self.points = {k: v for k, v in prev.get("puntos", {}).items()
                                   if k in self.pitch.keypoints() and list(v) != [0, 0]}
            except (json.JSONDecodeError, AttributeError):
                pass

    def set_point(self, name: str, x: float | None, y: float | None) -> dict:
        """Asigna (o borra, si x es None) un punto y devuelve el estado para dibujar."""
        if name not in self.pitch.keypoints():
            return {**self.state(), "aviso": f"Punto desconocido: {name}"}
        if x is None:
            self.points.pop(name, None)
        else:
            self.points[name] = [round(float(x), 1), round(float(y), 1)]
        self.save()
        return self.state()

    def save(self) -> None:
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        data = {"campo": self.campo, "puntos": self.points}
        self.output_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def state(self) -> dict:
        out = {"puntos": self.points, "lineas": [], "error": None, "aviso": ""}
        if len(self.points) < 4:
            out["aviso"] = f"Marca al menos {4 - len(self.points)} punto(s) más."
            return out
        try:
            calib = Calibration.from_dict({"campo": self.campo, "puntos": self.points})
        except ValueError as e:
            out["aviso"] = str(e)
            return out
        out["error"] = round(calib.reprojection_error(), 2)
        out["lineas"] = self._project_lines(calib)
        if out["error"] > 1.0:
            out["aviso"] = ("Error alto: algún punto está mal colocado o con el nombre "
                            "equivocado (¿'sup' e 'inf' cambiados?).")
        return out

    def _project_lines(self, calib: Calibration) -> list[list[list[float]]]:
        """Líneas del campo en píxeles, partidas donde quedan detrás de la cámara o muy fuera."""
        lim = 2.0 * max(self.width, self.height)
        polylines = []
        for line in self.pitch.lines():
            pts = _densify(line, step=0.5)
            hom = np.c_[pts, np.ones(len(pts))] @ calib.H_inv.T
            w = hom[:, 2]
            ok = w > 1e-9
            xy = np.full((len(pts), 2), np.nan)
            xy[ok] = hom[ok, :2] / w[ok, None]
            ok &= (np.abs(xy[:, 0]) < lim) & (np.abs(xy[:, 1]) < lim)
            current: list[list[float]] = []
            for valid, p in zip(ok, xy):
                if valid:
                    current.append([round(float(p[0]), 1), round(float(p[1]), 1)])
                elif current:
                    polylines.append(current)
                    current = []
            if len(current) > 1:
                polylines.append(current)
        return [p for p in polylines if len(p) > 1]


def read_frame(video: str | Path, segundo: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(segundo * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f"No se pudo leer el segundo {segundo} de {video}")
    return frame


def build_html(frame: np.ndarray, calibrator: ClickCalibrator, callback: str) -> str:
    """Página interactiva: imagen clicable + campo de referencia + lista de puntos."""
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(jpg.tobytes()).decode()
    pitch = calibrator.pitch
    config = {
        "callback": callback,
        "keypoints": pitch.keypoints(),
        "pitchLines": [l.round(2).tolist() for l in pitch.lines()],
        "pitchSize": [pitch.length, pitch.width],
        "state": calibrator.state(),
    }
    return (_HTML.replace("__IMG__", img_b64)
                 .replace("__CONFIG__", json.dumps(config, ensure_ascii=False)))


def calibrar_con_clics(video: str | Path, salida_json: str | Path, segundo: float = 2.0,
                       tipo: str = "futbol11", largo: float | None = None,
                       ancho: float | None = None) -> ClickCalibrator:
    """Muestra la herramienta de calibración por clics en Google Colab."""
    from google.colab import output  # solo disponible en Colab
    from IPython.display import HTML, JSON, display

    frame = read_frame(video, segundo)
    campo = {"tipo": tipo}
    if largo:
        campo["length"] = float(largo)
    if ancho:
        campo["width"] = float(ancho)
    calibrator = ClickCalibrator(salida_json, (frame.shape[1], frame.shape[0]), campo)

    callback = "soccerstats.calibrar_punto"
    output.register_callback(callback, lambda name, x, y: JSON(calibrator.set_point(name, x, y)))
    display(HTML(build_html(frame, calibrator, callback)))
    return calibrator


_HTML = r"""
<div id="ss-calib" style="font-family: system-ui, sans-serif; font-size: 14px; color: #222; background: #fff; padding: 8px;">
  <div style="display:flex; gap:12px; align-items:flex-start; flex-wrap:wrap;">
    <div style="min-width:260px; max-width:320px;">
      <p style="margin:0 0 6px"><b>1.</b> Elige un punto de la lista<br>
         <b>2.</b> Haz clic donde está en la imagen<br>
         <b>3.</b> Repite con 4 a 8 puntos repartidos por el campo</p>
      <select id="ss-name" size="12" style="width:100%; font-size:13px;"></select>
      <div style="margin:6px 0; display:flex; gap:6px;">
        <button id="ss-del" type="button">Borrar punto elegido</button>
      </div>
      <canvas id="ss-ref" style="width:100%; border:1px solid #ccc; border-radius:4px;"></canvas>
      <div id="ss-status" style="margin-top:6px; padding:6px; border-radius:4px; background:#f3f3f3;"></div>
    </div>
    <div style="flex:1; min-width:400px;">
      <canvas id="ss-img" style="width:100%; cursor:crosshair; border:1px solid #ccc; border-radius:4px;"></canvas>
      <div style="color:#666; font-size:12px;">Las líneas amarillas son el campo calculado: deben caer sobre las líneas reales. Puedes repetir un clic para corregir un punto.</div>
    </div>
  </div>
</div>
<script>
(function () {
  const cfg = __CONFIG__;
  let state = cfg.state;
  const names = Object.keys(cfg.keypoints);
  const sel = document.getElementById('ss-name');
  const imgCanvas = document.getElementById('ss-img');
  const refCanvas = document.getElementById('ss-ref');
  const status = document.getElementById('ss-status');
  const img = new Image();

  function fillList() {
    const current = sel.value;
    sel.innerHTML = '';
    for (const n of names) {
      const o = document.createElement('option');
      o.value = n;
      o.textContent = (state.puntos[n] ? '✔ ' : '   ') + n;
      sel.appendChild(o);
    }
    sel.value = current || names.find(n => !state.puntos[n]) || names[0];
  }

  function drawImage() {
    if (!img.complete) return;
    imgCanvas.width = img.naturalWidth; imgCanvas.height = img.naturalHeight;
    const ctx = imgCanvas.getContext('2d');
    const k = Math.max(1, img.naturalWidth / 1000);
    ctx.drawImage(img, 0, 0);
    ctx.strokeStyle = '#ffd400'; ctx.lineWidth = 2 * k;
    for (const line of state.lineas) {
      ctx.beginPath();
      line.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      ctx.stroke();
    }
    ctx.font = (14 * k) + 'px sans-serif';
    for (const [n, p] of Object.entries(state.puntos)) {
      const active = n === sel.value;
      ctx.fillStyle = active ? '#00e5ff' : '#ff2d2d';
      ctx.beginPath(); ctx.arc(p[0], p[1], (active ? 7 : 5) * k, 0, 2 * Math.PI); ctx.fill();
      ctx.lineWidth = 3 * k; ctx.strokeStyle = '#000';
      ctx.strokeText(n, p[0] + 8 * k, p[1] - 8 * k);
      ctx.fillText(n, p[0] + 8 * k, p[1] - 8 * k);
    }
  }

  function drawRef() {
    const [L, W] = cfg.pitchSize, pad = 12, s = 280 / L;
    refCanvas.width = L * s + 2 * pad; refCanvas.height = W * s + 2 * pad;
    const ctx = refCanvas.getContext('2d');
    const P = p => [p[0] * s + pad, p[1] * s + pad];
    ctx.fillStyle = '#3c8c3c'; ctx.fillRect(0, 0, refCanvas.width, refCanvas.height);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1;
    for (const line of cfg.pitchLines) {
      ctx.beginPath();
      line.forEach((p, i) => { const q = P(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
      ctx.stroke();
    }
    for (const [n, p] of Object.entries(cfg.keypoints)) {
      const q = P(p), active = n === sel.value;
      ctx.fillStyle = active ? '#00e5ff' : (state.puntos[n] ? '#ff2d2d' : 'rgba(255,255,255,0.6)');
      ctx.beginPath(); ctx.arc(q[0], q[1], active ? 6 : 3, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.fillStyle = '#fff'; ctx.font = '11px sans-serif';
    ctx.fillText('Punto elegido: ' + sel.value, pad, refCanvas.height - 3);
  }

  function drawStatus() {
    const n = Object.keys(state.puntos).length;
    let html = '<b>Puntos marcados:</b> ' + n;
    if (state.error !== null && state.error !== undefined) {
      const good = state.error <= 1.0;
      html += '<br><b>Error medio:</b> <span style="color:' + (good ? '#0a7d20' : '#c00') + '">' +
              state.error.toFixed(2) + ' m ' + (good ? '✅' : '⚠️') + '</span>';
    }
    if (state.aviso) html += '<br>' + state.aviso;
    if (n >= 4 && state.error !== null && state.error <= 1.0)
      html += '<br>Guardado. Si las líneas encajan, pasa a la siguiente celda.';
    status.innerHTML = html;
  }

  function redraw() { fillList(); drawImage(); drawRef(); drawStatus(); }

  async function send(name, x, y) {
    status.innerHTML = 'Guardando…';
    try {
      const r = await google.colab.kernel.invokeFunction(cfg.callback, [name, x, y], {});
      state = r.data['application/json'];
    } catch (e) {
      status.innerHTML = '❌ No se pudo comunicar con Python: ' + e +
        '<br>Vuelve a ejecutar la celda.';
      return false;
    }
    return true;
  }

  imgCanvas.addEventListener('click', async (ev) => {
    const rect = imgCanvas.getBoundingClientRect();
    const x = (ev.clientX - rect.left) * imgCanvas.width / rect.width;
    const y = (ev.clientY - rect.top) * imgCanvas.height / rect.height;
    const name = sel.value;
    if (await send(name, x, y)) {
      const next = names.find(n => !state.puntos[n] && n !== name);
      redraw();
      if (next) { sel.value = next; drawImage(); drawRef(); }
    }
  });
  document.getElementById('ss-del').addEventListener('click', async () => {
    if (await send(sel.value, null, null)) redraw();
  });
  sel.addEventListener('change', () => { drawImage(); drawRef(); });

  img.onload = redraw;
  img.src = 'data:image/jpeg;base64,__IMG__';
  fillList();
})();
</script>
"""
