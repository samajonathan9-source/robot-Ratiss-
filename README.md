# Robot RATIS — Robot souverain (cerveau LCT greffé sur LeRobot)

> **Architect** : Jonathan Evina · ORCID 0009-0000-4092-5313
> **Propriété intellectuelle** : JOHNKING0 & Jonathan Evina
> **Loi fondamentale** : LCT (R = P_sig, ΔW = η·φ·P_sig·C) — figée

Un robot qui **voit** (caméra), **sent** (capteurs), **pense** (LCT),
**ressent** (ETH), **parle** (décodeur), et **certifie** (ZK) — souverain,
100% local, pas de cloud, pas de LLM externe.

## Pourquoi c'est différent

La robotique classique (LeRobot, π0, ACT) : le robot **imite** (gradient sur
millions de démonstrations). Il ne comprend pas, ne ressent pas, ne certifie pas.

RATIS : le robot **décide par physique topologique**. La loi LCT gouverne la
décision — P_sig (persistance de la scène) × C (cohérence des capteurs). Le robot
certifie chaque décision par hash ZK. Son émotion **émerge** des capteurs (ETH).

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              ROBOT RATIS SOUVERAIN                   │
│                                                       │
│  Caméra ──▶ PERCEVOIR (cycles H1 → P_sig)            │
│  Capteurs ─▶ RESSENTIR (ETH : accel/gyro → C, émotion)│
│               ↓                                       │
│  Cerveau TTF ─▶ PENSER (MCB, sans mots)               │
│  Réseau LCT  ─▶ COMPRENDRE (ΔW = η·φ·P_sig·C)        │
│  Décodeur    ─▶ PARLER (langage conditionné)          │
│  Hash topo   ─▶ CERTIFIER (ZK invariant)              │
│               ↓                                       │
│  Action (LeRobot MotorsBus ou affichage)              │
└─────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install numpy scipy fastapi uvicorn opencv-python gudhi
# optionnel : pip install lerobot (pour le téléphone + moteurs)
```

## Lancer l'interface web

```bash
python interface/server.py
# Ouvre http://localhost:12000
```

L'interface affiche en temps réel :
- 👁️ **Vision** : flux caméra + P_sig + cycles H1
- 🧠 **Cognition** : émotion (ETH), cohérence C, action, confiance
- 🗣️ **Dialogue** : pose des questions à RATIS, il répond

## Lancer le test (sans matériel)

```bash
python tests/test_robot_brain.py
```

## Modules

| Module | Rôle |
|---|---|
| `ratis_robot/ratis_brain.py` | Le cerveau RATIS (percevoir, ressentir, décider, certifier) |
| `ratis_robot/phone_robot.py` | Le robot téléphone (caméra + capteurs + cerveau) |
| `interface/server.py` | Serveur web FastAPI (interface temps réel) |

## La loi LCT en robotique

```
R = P_sig  (persistance topo de la scène observée)
C = 1/(1+arousal)  (cohérence des capteurs)
ΔW = η · φ · P_sig · C  (apprentissage, pas de gradient)
```

Le robot agit quand la topologie de la scène est cohérente (P_sig haut) ET son
état interne est stable (C haut). Sinon il attend ou recule. Chaque décision est
certifiée par un hash topologique invariant (ZK).

## Souveraineté

- Aucune donnée vers le cloud
- Aucun LLM externe
- Le téléphone seul est déjà un robot cognitif souverain
- La certification ZK prouve que la décision respecte les invariants physiques

---

*© 2026 JOHNKING0 & Jonathan Evina. Loi LCT figée, robotique souveraine.*
