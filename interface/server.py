"""interface.server — Serveur web local pour le robot RATIS.

Déploie une interface web où RATIS voit (caméra), sent (capteurs), pense (LCT),
ressent (ETH), parle (décodeur), et certifie (ZK) — en temps réel, dans le
navigateur. Souverain : 100% local, pas de cloud.

Endpoints :
  GET  /            — l'interface web (HTML + JS)
  GET  /api/video   — flux caméra (MJPEG)
  GET  /api/think   — une boucle cognitive (JSON : perception, émotion, décision, ZK)
  POST /api/chat    — poser une question à RATIS (dialogue engine)
  GET  /api/health  — santé du robot
"""
from __future__ import annotations

import io
import json
import math
import time
import threading
from pathlib import Path

from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel

import numpy as np

import sys
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ratis_robot.phone_robot import PhoneRobot
from ratis_robot.ratis_brain import RatisBrain

app = FastAPI(title="RATIS Robot Souverain", version="1.0")

# le robot (singleton global)
_robot: PhoneRobot | None = None
_robot_lock = threading.Lock()


def get_robot() -> PhoneRobot:
    global _robot
    if _robot is None:
        with _robot_lock:
            if _robot is None:
                _robot = PhoneRobot(RatisBrain())
                _robot.connect()
    return _robot


@app.get("/api/health")
async def health():
    r = get_robot()
    return {"status": "ok", "connected": r._connected,
            "camera": r._cap is not None, "phone": r._phone is not None}


@app.get("/api/think")
async def think():
    """Une boucle cognitive : voit, sent, pense, certifie."""
    r = get_robot()
    decision = r.think()
    return JSONResponse({
        "perception": {
            "p_sig": round(decision.perception.p_sig, 4),
            "n_cycles": decision.perception.n_cycles,
            "n_points": decision.perception.n_points,
            "has_structure": decision.perception.has_structure,
        },
        "emotion": {
            "C": round(decision.emotion.C, 4),
            "arousal": round(decision.emotion.arousal, 4),
            "emotion": decision.emotion.emotion,
        },
        "decision": {
            "action": decision.action,
            "confidence": round(decision.confidence, 3),
            "phrase": decision.phrase,
            "zk_hash": decision.zk_hash,
            "certified": decision.certified,
        },
        "timestamp": time.time(),
    })


class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Pose une question à RATIS (dialogue engine topologique)."""
    r = get_robot()
    response = r.brain.answer(req.question)
    return JSONResponse({"question": req.question, "response": response})


@app.get("/api/video")
async def video():
    """Flux caméra MJPEG (si webcam dispo)."""
    r = get_robot()
    if r._cap is None:
        return Response(status_code=404, content="Pas de caméra")

    def generate():
        while True:
            frame = r.get_frame()
            if frame is None:
                continue
            try:
                import cv2
                _, buf = cv2.imencode(".jpg", frame)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            except ImportError:
                break
            time.sleep(0.1)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
async def index():
    """L'interface web RATIS."""
    return INTERFACE_HTML


INTERFACE_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RATIS — Robot Souverain</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a1a; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
  .header { text-align: center; padding: 20px; background: linear-gradient(135deg, #0f3460, #1a1a2e); }
  .header h1 { font-size: 1.8em; color: #e94560; }
  .header p { color: #95a5a6; font-size: 0.9em; }
  .container { display: flex; flex-wrap: wrap; max-width: 1200px; margin: 20px auto; gap: 20px; }
  .panel { background: #16213e; border-radius: 12px; padding: 20px; flex: 1; min-width: 300px; }
  .panel h2 { color: #e94560; margin-bottom: 15px; font-size: 1.1em; }
  .metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1a1a3e; }
  .metric .label { color: #95a5a6; }
  .metric .value { font-weight: bold; }
  .emotion { font-size: 1.5em; text-align: center; padding: 15px; margin: 10px 0; border-radius: 8px; }
  .emotion.focus { background: #16a08522; color: #16a085; }
  .emotion.calme { background: #0f346022; color: #6c5ce7; }
  .emotion.anxiete { background: #f39c1222; color: #f39c12; }
  .emotion.danger { background: #e9456022; color: #e94560; }
  .emotion.neutre { background: #95a5a622; color: #95a5a6; }
  .zk { font-family: monospace; font-size: 0.85em; color: #16a085; text-align: center; padding: 8px; }
  .phrase { font-style: italic; text-align: center; padding: 12px; color: #e0e0e0; background: #0a0a2e; border-radius: 8px; margin: 10px 0; }
  .chat { display: flex; flex-direction: column; gap: 10px; }
  .chat-input { display: flex; gap: 10px; }
  .chat-input input { flex: 1; padding: 10px; background: #0a0a2e; border: 1px solid #1a1a3e; color: #e0e0e0; border-radius: 8px; }
  .chat-input button { padding: 10px 20px; background: #e94560; color: white; border: none; border-radius: 8px; cursor: pointer; }
  .chat-msg { padding: 10px; border-radius: 8px; max-width: 80%; }
  .chat-msg.user { background: #0f3460; align-self: flex-end; }
  .chat-msg.ratis { background: #16213e; align-self: flex-start; }
  .action-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: bold; }
  .action.agir { background: #16a08522; color: #16a085; }
  .action.attendre { background: #f39c1222; color: #f39c12; }
  .action.reculer { background: #e9456022; color: #e94560; }
  .action.observer { background: #6c5ce722; color: #6c5ce7; }
  .action.saisir { background: #16a08522; color: #16a085; }
  #video { width: 100%; border-radius: 8px; }
  .footer { text-align: center; padding: 20px; color: #95a5a6; font-size: 0.8em; }
</style>
</head>
<body>
<div class="header">
  <h1>🤖 RATIS — Robot Souverain</h1>
  <p>Voit · Sent · Pense (LCT) · Ressent (ETH) · Parle · Certifie (ZK) — 100% local</p>
</div>
<div class="container">
  <div class="panel">
    <h2>👁️ Vision (caméra)</h2>
    <img id="video" src="/api/video" alt="Caméra">
    <div class="metric"><span class="label">P_sig (persistance topo)</span><span class="value" id="p_sig">—</span></div>
    <div class="metric"><span class="label">Cycles H1</span><span class="value" id="n_cycles">—</span></div>
    <div class="metric"><span class="label">Structure détectée</span><span class="value" id="structure">—</span></div>
  </div>
  <div class="panel">
    <h2>🧠 Cognition (LCT + ETH)</h2>
    <div class="emotion neutre" id="emotion">neutre</div>
    <div class="metric"><span class="label">Cohérence C</span><span class="value" id="C">—</span></div>
    <div class="metric"><span class="label">Arousal (agitation)</span><span class="value" id="arousal">—</span></div>
    <div class="metric"><span class="label">Confiance</span><span class="value" id="confidence">—</span></div>
    <div class="metric"><span class="label">Action</span><span class="value"><span class="action-badge observer" id="action">observer</span></span></div>
    <div class="phrase" id="phrase">Initialisation...</div>
    <div class="zk" id="zk">ZK: —</div>
  </div>
  <div class="panel">
    <h2>🗣️ Dialogue (parler)</h2>
    <div class="chat" id="chat-box" style="max-height:300px;overflow-y:auto;margin-bottom:15px;"></div>
    <div class="chat-input">
      <input id="chat-input" placeholder="Pose une question à RATIS..." onkeypress="if(event.key==='Enter')sendChat()">
      <button onclick="sendChat()">Envoyer</button>
    </div>
  </div>
</div>
<div class="footer">
  © 2026 JOHNKING0 & Jonathan Evina · Loi LCT : R = P_sig, ΔW = η·φ·P_sig·C · Souverain, pas de cloud
</div>
<script>
async function think() {
  try {
    const r = await fetch('/api/think');
    const d = await r.json();
    document.getElementById('p_sig').textContent = d.perception.p_sig.toFixed(3);
    document.getElementById('n_cycles').textContent = d.perception.n_cycles;
    document.getElementById('structure').textContent = d.perception.has_structure ? '✓ cohérente' : '✗ bruit';
    const emo = d.emotion.emotion;
    document.getElementById('emotion').textContent = emo;
    document.getElementById('emotion').className = 'emotion ' + emo;
    document.getElementById('C').textContent = d.emotion.C.toFixed(3);
    document.getElementById('arousal').textContent = d.emotion.arousal.toFixed(3);
    document.getElementById('confidence').textContent = (d.decision.confidence*100).toFixed(0) + '%';
    const act = d.decision.action;
    document.getElementById('action').textContent = act;
    document.getElementById('action').className = 'action-badge ' + act;
    document.getElementById('phrase').textContent = '« ' + d.decision.phrase + ' »';
    document.getElementById('zk').textContent = 'ZK: ' + d.decision.zk_hash + (d.decision.certified ? ' ✓' : ' ✗');
  } catch(e) { console.error(e); }
}
async function sendChat() {
  const input = document.getElementById('chat-input');
  const q = input.value.trim();
  if (!q) return;
  input.value = '';
  const box = document.getElementById('chat-box');
  const u = document.createElement('div'); u.className='chat-msg user'; u.textContent=q; box.appendChild(u);
  const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:q})});
  const d = await r.json();
  const m = document.createElement('div'); m.className='chat-msg ratis'; m.textContent=d.response; box.appendChild(m);
  box.scrollTop = box.scrollHeight;
}
think();
setInterval(think, 1000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  RATIS Robot Souverain — Interface web")
    print("  Ouvre http://localhost:12000 dans ton navigateur")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=12000)
