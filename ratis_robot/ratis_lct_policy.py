"""ratis_robot.ratis_lct_policy — La politique RATIS LCT pour LeRobot.

Remplace ACTPolicy / DiffusionPolicy par le cerveau RATIS. Au lieu d'un réseau
entraîné par gradient, la décision vient de la loi LCT (ΔW = η·φ·P_sig·C) :
  - L'image caméra → topologie (cycles H1 → P_sig)
  - Les capteurs → ETH (cohérence C → émotion)
  - La décision = cohérence topologique (pas de gradient)
  - Certification ZK (hash topo invariant)

Cette classe respecte l'interface de LeRobot (select_action) mais pense par LCT.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np

from ratis_robot.ratis_brain import RatisBrain, RobotPerception, RobotEmotion


class RatisLCTPolicy:
    """Politique RATIS LCT — remplace les politiques gradient de LeRobot.

    Au lieu de select_action(obs) → action par réseau de neurones (gradient),
    on fait : select_action(obs) → action par cohérence topologique (LCT).

    L'observation LeRobot contient :
      - observation.images : frame caméra (H, W, C)
      - observation.state : état du robot (positions moteurs, etc.)

    On transforme l'état en environnement thermodynamique (ETH) :
      - l'agitation des moteurs = arousal → C (cohérence)
      - la position = phase φ
    """

    def __init__(self, eta: float = 0.2, n_joints: int = 6):
        self.brain = RatisBrain(eta=eta)
        self.n_joints = n_joints
        self.last_action = np.zeros(n_joints)

    def select_action(self, batch: dict) -> np.ndarray:
        """Décide l'action par LCT (pas de gradient).

        Args:
            batch : dictionnaire LeRobot avec observation.images, observation.state.

        Returns:
            action : np.ndarray (n_joints,) — deltas de position moteurs.
        """
        # 1. extraire l'observation
        images = batch.get("observation.images", None)
        state = batch.get("observation.state", None)

        # image → numpy (si tensor PyTorch)
        frame = None
        if images is not None:
            if hasattr(images, "cpu"):
                frame = images.cpu().numpy()
            else:
                frame = np.array(images)
            # (H, W, C) ou (B, H, W, C) → on prend la première
            if frame.ndim == 4:
                frame = frame[0]

        # état → capteurs ETH
        sensors = None
        if state is not None:
            state_np = state.cpu().numpy() if hasattr(state, "cpu") else np.array(state)
            # l'agitation = norme de la dérivée de l'état (= vitesse des moteurs)
            delta = state_np - self.last_action[:len(state_np)] if len(self.last_action) >= len(state_np) else state_np
            arousal = float(np.linalg.norm(delta))
            sensors = {
                "accelerometer": np.array([arousal, 0, 0]),
                "gyroscope": np.array([delta[0] if len(delta) > 0 else 0, 0, 0]),
                "motor_state": state_np,
            }
            self.last_action = state_np

        # 2. boucle cognitive RATIS
        decision = self.brain.think(frame, sensors)

        # 3. transformer la décision en action motrice
        action = self._decision_to_action(decision)

        return action

    def _decision_to_action(self, decision) -> np.ndarray:
        """Transforme une décision cognitive en action motrice (n_joints).

        L'action dépend de la décision RATIS :
          - saisir : les moteurs avancent (grip)
          - reculer : les moteurs reculent
          - attendre/observer : immobilisation
        """
        action = np.zeros(self.n_joints)
        if decision.action == "saisir":
            action[:3] = 0.05    # avancer
            action[-1] = 0.3     # fermer la pince
        elif decision.action == "reculer":
            action[:3] = -0.05   # reculer
        elif decision.action == "agir":
            action[:3] = 0.03
        # attendre/observer → action = 0 (immobile)
        return action

    def perceive_topologically(self, frame: np.ndarray) -> RobotPerception:
        """Perception topologique pure (exposé pour le monitoring)."""
        return self.brain.perceive(frame)

    def feel_thermodynamically(self, sensors: dict) -> RobotEmotion:
        """Ressenti ETH pur (exposé pour le monitoring)."""
        return self.brain.feel(sensors)

    @property
    def last_decision(self):
        """La dernière décision cognitive (pour l'affichage)."""
        return self.brain.history[-1] if self.brain.history else None
