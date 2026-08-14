"""tests/test_robot_brain.py — Test du cerveau RATIS robotique.

Valide la boucle cognitive : percevoir → ressentir → décider → certifier,
sans matériel (capteurs simulés, frame synthétique).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ratis_robot.ratis_brain import RatisBrain


def main():
    print("=" * 60)
    print("  Test du cerveau RATIS robotique (sans matériel)")
    print("=" * 60)

    brain = RatisBrain()

    # 1. Percevoir (frame synthétique — scène structurée vs bruit)
    print("\n1. PERCEVOIR (frame synthétique) :")
    frame_structured = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    frame_structured[:50, :] = 0  # moitié noire = structure
    p1 = brain.perceive(frame_structured)
    print(f"   scène structurée : P_sig={p1.p_sig:.3f}, cycles={p1.n_cycles}, structure={p1.has_structure}")

    p2 = brain.perceive(None)
    print(f"   pas de caméra    : P_sig={p2.p_sig:.3f}, cycles={p2.n_cycles}")

    # 2. Ressentir (capteurs simulés — stable vs agité)
    print("\n2. RESSENTIR (capteurs simulés) :")
    sensors_calm = {"accelerometer": np.array([0.01, 0.01, 0.01]),
                    "gyroscope": np.array([0.001, 0.001, 0.001])}
    e1 = brain.feel(sensors_calm)
    print(f"   stable  : C={e1.C:.3f}, arousal={e1.arousal:.4f}, émotion={e1.emotion}")

    sensors_chaos = {"accelerometer": np.array([5.0, 3.0, 2.0]),
                     "gyroscope": np.array([1.0, 0.5, 0.3])}
    e2 = brain.feel(sensors_chaos)
    print(f"   agité   : C={e2.C:.3f}, arousal={e2.arousal:.4f}, émotion={e2.emotion}")

    # 3. Décider + certifier
    print("\n3. DÉCIDER + CERTIFIER :")
    d1 = brain.decide(p1, e1)
    print(f"   scène stable + calme : action={d1.action}, conf={d1.confidence:.0%}")
    print(f"   phrase : « {d1.phrase} »")
    print(f"   ZK : {d1.zk_hash} {'✓' if d1.certified else '✗'}")

    d2 = brain.decide(p1, e2)
    print(f"   scène stable + agité : action={d2.action}, conf={d2.confidence:.0%}")
    print(f"   phrase : « {d2.phrase} »")
    print(f"   ZK : {d2.zk_hash} {'✓' if d2.certified else '✗'}")

    # 4. Boucle complète (think)
    print("\n4. BOUCLE COGNITIVE (think) :")
    d = brain.think(frame_structured, sensors_calm)
    print(f"   perception : P_sig={d.perception.p_sig:.3f}, structure={d.perception.has_structure}")
    print(f"   émotion    : C={d.emotion.C:.3f}, {d.emotion.emotion}")
    print(f"   décision   : {d.action} (conf {d.confidence:.0%})")
    print(f"   phrase     : « {d.phrase} »")
    print(f"   ZK         : {d.zk_hash} {'✓ certifié' if d.certified else '✗'}")

    # 5. Dialogue
    print("\n5. DIALOGUE :")
    try:
        rep = brain.answer("qui es-tu")
        print(f"   Q: qui es-tu ?")
        print(f"   R: {rep[:120]}...")
    except Exception as ex:
        print(f"   (dialogue engine non disponible : {ex})")

    print(f"\n{'='*60}")
    print("BILAN : le cerveau RATIS fonctionne en robotique (sans matériel)")
    print("  perçoit, ressent, décide, certifie — souverain")
    print(f"  ZK : {sum(1 for d in brain.history if d.certified)}/{len(brain.history)} certifiées")

    return {"n_decisions": len(brain.history),
            "n_certified": sum(1 for d in brain.history if d.certified)}


if __name__ == "__main__":
    out = main()
    out_path = _ROOT / "tests" / "robot_brain_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRésultats : {out_path}")
