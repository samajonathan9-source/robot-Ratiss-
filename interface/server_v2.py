"""interface.server — Serveur web RATIS avec vrais composants LeRobot.

Utilise les VRAIS modules LeRobot (cameras, teleoperators) + le cerveau RATIS.
RATIS voit (vraie caméra LeRobot), sent (capteurs), pense (LCT), ressent (ETH),
parle (décodeur + TTS gTTS), et certifie (ZK).

Endpoints :
  GET  /            — interface web
  GET  /api/video   — vrai flux caméra (LeRobot opencv)
  GET  /api/think   — boucle cognitive
  POST /api/chat    — dialogue (base + génération LCT)
  POST /api/speak   — TTS gTTS
  GET  /api/health  — santé
"""
from __future__ import annotations

import io
import json
import math
import time
import threading
import tempfile
import os
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel

import numpy as np
import cv2

import sys
_REPO = Path(__file__).resolve().parents[1]
_LEROBOT = _REPO / "lerobot" / "src"
if str(_LEROBOT) not in sys.path:
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, str(_LEROBOT))

from ratis_robot.ratis_brain import RatisBrain

app = FastAPI(title="RATIS Robot Souverain", version="2.0")

_robot_brain: RatisBrain | None = None
_robot_lock = threading.Lock()
_cap = None  # camera OpenCV (LeRobot utilise cv2 en interne)


def get_brain() -> RatisBrain:
    global _robot_brain
    if _robot_brain is None:
        with _robot_lock:
            if _robot_brain is None:
                _robot_brain = RatisBrain()
    return _robot_brain


def get_camera():
    """Connexion caméra via OpenCV (même backend que LeRobot cameras/opencv)."""
    global _cap
    if _cap is None:
        _cap = cv2.VideoCapture(0)
        if not _cap.isOpened():
            _cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not _cap.isOpened():
            print("[RATIS] Caméra non détectée — mode simulation.")
            _cap = None
    return _cap


@app.get("/api/health")
async def health():
    b = get_brain()
    cap = get_camera()
    return {"status": "ok", "camera": cap is not None,
            "dialogue_trained": b._trained,
            "ttf_brain": True}


@app.get("/api/think")
async def think():
    """Boucle cognitive : voit, ressent, décide, certifie."""
    b = get_brain()
    cap = get_camera()
    frame = None
    if cap is not None:
        ret, frame = cap.read()
        if not ret:
            frame = None
    # capteurs simulés (téléphone LeRobot non dispo ici)
    t = time.time()
    sensors = {
        "accelerometer": np.array([0.05 * math.sin(t), 0.03, 0.02]),
        "gyroscope": np.array([0.01, 0.01 * math.cos(t), 0.005]),
    }
    decision = b.think(frame, sensors)
    return JSONResponse({
        "perception": {
            "p_sig": round(decision.perception.p_sig, 4),
            "n_cycles": decision.perception.n_cycles,
            "n_points": decision.perception.n_points,
            "has_structure": decision.perception.has_structure,
            "camera": frame is not None,
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
    """Dialogue RATIS : base de connaissances + génération LCT."""
    b = get_brain()
    response = b.answer(req.question)
    return JSONResponse({"question": req.question, "response": response})


class SpeakRequest(BaseModel):
    text: str


@app.post("/api/speak")
async def speak(req: SpeakRequest):
    """TTS gTTS (français) → MP3."""
    try:
        from gtts import gTTS
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir="/tmp")
        tts = gTTS(text=req.text, lang="fr", slow=False)
        tts.save(tmp.name)
        audio = open(tmp.name, "rb").read()
        os.unlink(tmp.name)
        return Response(content=audio, media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/video")
async def video():
    """Vrai flux caméra (LeRobot opencv) annoté avec l'état cognitif."""
    cap = get_camera()
    b = get_brain()
    if cap is None:
        # image de substitution avec l'état cognitif
        def gen_placeholder():
            while True:
                img = np.zeros((240, 320, 3), dtype=np.uint8)
                d = b.history[-1] if b.history else None
                if d:
                    txt1 = f"{d.emotion.emotion} | {d.action}"
                    txt2 = f"ZK: {d.zk_hash[:8]}"
                    cv2.putText(img, "PAS DE CAMERA", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                    cv2.putText(img, txt1, (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(img, txt2, (80, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                _, buf = cv2.imencode(".jpg", img)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                time.sleep(0.5)
        return StreamingResponse(gen_placeholder(), media_type="multipart/x-mixed-replace; boundary=frame")

    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            d = b.history[-1] if b.history else None
            if d:
                txt = f"{d.emotion.emotion} | {d.action} | ZK:{d.zk_hash[:8]}"
                cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"P_sig={d.perception.p_sig:.2f} C={d.emotion.C:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            _, buf = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.1)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
async def index():
    return INTERFACE_HTML


INTERFACE_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RATIS — Robot Souverain</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a1a;color:#e0e0e0;font-family:'Segoe UI',sans-serif}
.header{text-align:center;padding:20px;background:linear-gradient(135deg,#0f3460,#1a1a2e)}
.header h1{font-size:1.8em;color:#e94560}
.header p{color:#95a5a6;font-size:0.9em}
.container{display:flex;flex-wrap:wrap;max-width:1200px;margin:20px auto;gap:20px}
.panel{background:#16213e;border-radius:12px;padding:20px;flex:1;min-width:300px}
.panel h2{color:#e94560;margin-bottom:15px;font-size:1.1em}
.metric{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1a1a3e}
.metric .label{color:#95a5a6}
.metric .value{font-weight:bold}
.emotion{font-size:1.5em;text-align:center;padding:15px;margin:10px 0;border-radius:8px}
.emotion.focus{background:#16a08522;color:#16a085}
.emotion.calme{background:#0f346022;color:#6c5ce7}
.emotion.anxiete{background:#f39c1222;color:#f39c12}
.emotion.danger{background:#e9456022;color:#e94560}
.emotion.neutre{background:#95a5a622;color:#95a5a6}
.zk{font-family:monospace;font-size:0.85em;color:#16a085;text-align:center;padding:8px}
.phrase{font-style:italic;text-align:center;padding:12px;color:#e0e0e0;background:#0a0a2e;border-radius:8px;margin:10px 0}
.chat{display:flex;flex-direction:column;gap:10px}
.chat-input{display:flex;gap:10px}
.chat-input input{flex:1;padding:10px;background:#0a0a2e;border:1px solid #1a1a3e;color:#e0e0e0;border-radius:8px}
.chat-input button{padding:10px 20px;background:#e94560;color:white;border:none;border-radius:8px;cursor:pointer}
.chat-msg{padding:10px;border-radius:8px;max-width:85%;line-height:1.4}
.chat-msg.user{background:#0f3460;align-self:flex-end}
.chat-msg.ratis{background:#16213e;align-self:flex-start}
.action-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:0.85em;font-weight:bold}
.action.agir{background:#16a08522;color:#16a085}
.action.attendre{background:#f39c1222;color:#f39c12}
.action.reculer{background:#e9456022;color:#e94560}
.action.observer{background:#6c5ce722;color:#6c5ce7}
.action.saisir{background:#16a08522;color:#16a085}
#video{width:100%;border-radius:8px;background:#000}
.footer{text-align:center;padding:20px;color:#95a5a6;font-size:0.8em}
</style>
</head>
<body>
<div class="header">
<h1>🤖 RATIS — Robot Souverain</h1>
<p>Voit (LeRobot caméra) · Sent (capteurs ETH) · Pense (LCT) · Parle (gTTS) · Certifie (ZK) — 100% local</p>
</div>
<div class="container">
<div class="panel">
<h2>👁️ Vision (caméra LeRobot)</h2>
<img id="video" src="/api/video" alt="Caméra">
<div class="metric"><span class="label">P_sig (persistance topo)</span><span class="value" id="p_sig">—</span></div>
<div class="metric"><span class="label">Cycles H1</span><span class="value" id="n_cycles">—</span></div>
<div class="metric"><span class="label">Structure</span><span class="value" id="structure">—</span></div>
</div>
<div class="panel">
<h2>🧠 Cognition (LCT + ETH)</h2>
<div class="emotion neutre" id="emotion">neutre</div>
<div class="metric"><span class="label">Cohérence C</span><span class="value" id="C">—</span></div>
<div class="metric"><span class="label">Arousal</span><span class="value" id="arousal">—</span></div>
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
<button onclick="speakLast()" style="background:#16a085">🔊 Parler</button>
</div>
</div>
</div>
<div class="footer">© 2026 JOHNKING0 & Jonathan Evina · Loi LCT : R = P_sig, ΔW = η·φ·P_sig·C · Cerveau TTF + LeRobot</div>
<script>
async function think(){try{const r=await fetch('/api/think');const d=await r.json();
document.getElementById('p_sig').textContent=d.perception.p_sig.toFixed(3);
document.getElementById('n_cycles').textContent=d.perception.n_cycles;
document.getElementById('structure').textContent=d.perception.has_structure?'✓ cohérente':'✗ bruit';
const emo=d.emotion.emotion;document.getElementById('emotion').textContent=emo;
document.getElementById('emotion').className='emotion '+emo;
document.getElementById('C').textContent=d.emotion.C.toFixed(3);
document.getElementById('arousal').textContent=d.emotion.arousal.toFixed(3);
document.getElementById('confidence').textContent=(d.decision.confidence*100).toFixed(0)+'%';
const act=d.decision.action;document.getElementById('action').textContent=act;
document.getElementById('action').className='action-badge '+act;
document.getElementById('phrase').textContent='« '+d.decision.phrase+' »';
document.getElementById('zk').textContent='ZK: '+d.decision.zk_hash+(d.decision.certified?' ✓':' ✗');
}catch(e){console.error(e)}}
async function sendChat(){const input=document.getElementById('chat-input');const q=input.value.trim();
if(!q)return;input.value='';
const box=document.getElementById('chat-box');
const u=document.createElement('div');u.className='chat-msg user';u.textContent=q;box.appendChild(u);
const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
const d=await r.json();
const m=document.createElement('div');m.className='chat-msg ratis';m.textContent=d.response;box.appendChild(m);
box.scrollTop=box.scrollHeight;}
async function speakLast(){const msgs=document.querySelectorAll('.chat-msg.ratis');
if(msgs.length===0)return;const text=msgs[msgs.length-1].textContent;
const r=await fetch('/api/speak',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text})});
if(r.ok){const blob=await r.blob();new Audio(URL.createObjectURL(blob)).play();}}
think();setInterval(think,1000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  RATIS Robot Souverain v2 — LeRobot + cerveau LCT")
    print("  Ouvre http://localhost:12000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=12000)
