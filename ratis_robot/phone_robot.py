"""ratis_robot.phone_robot — Le robot téléphone RATIS souverain.

Ton téléphone (ou webcam + clavier) devient un robot RATIS :
  - Il VOIT (webcam → topologie de la scène)
  - Il SENT (capteurs simulés ou accel/gyro du téléphone)
  - Il PENSE (cerveau RATIS : LCT + TTF + ETH)
  - Il PARLE (décodeur LCT + synthèse vocale)
  - Il CERTIFIE (ZK hash topo invariant)

Sans bras physique : le téléphone seul est déjà un robot cognitif souverain.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ratis_robot.ratis_brain import RatisBrain, RobotDecision


class PhoneRobot:
    """Le robot téléphone RATIS.

    Args:
        brain : le cerveau RATIS (RatisBrain).
        camera_index : index de la webcam (0 = caméra par défaut).
    """

    def __init__(self, brain: RatisBrain | None = None, camera_index: int = 0):
        self.brain = brain or RatisBrain()
        self.camera_index = camera_index
        self._cap = None
        self._phone = None
        self._connected = False

    def connect(self) -> bool:
        """Connecte la caméra (OpenCV) et les capteurs (si téléphone dispo)."""
        try:
            import cv2
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                print("[PhoneRobot] Caméra non détectée — mode sans caméra (fallback).")
                self._cap = None
            else:
                print("[PhoneRobot] Caméra connectée.")
        except ImportError:
            print("[PhoneRobot] OpenCV non installé — mode sans caméra.")
            self._cap = None

        # tentative de connexion au téléphone (LeRobot teleop_phone)
        try:
            from lerobot.teleoperators.phone import PhoneTeleoperator
            self._phone = PhoneTeleoperator()
            self._phone.connect()
            print("[PhoneRobot] Téléphone connecté (capteurs ETH).")
        except Exception:
            print("[PhoneRobot] Pas de téléphone LeRobot — capteurs simulés.")
            self._phone = None

        self._connected = True
        return True

    def get_frame(self) -> np.ndarray | None:
        """Capture une frame de la webcam."""
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def get_sensors(self) -> dict:
        """Lit les capteurs (téléphone ou simulés)."""
        if self._phone is not None:
            try:
                obs = self._phone.get_observation()
                return obs
            except Exception:
                pass
        # capteurs simulés (pour test sans téléphone)
        t = time.time()
        return {
            "accelerometer": np.array([0.1 * math.sin(t), 0.05 * math.cos(t * 1.3), 0.02]),
            "gyroscope": np.array([0.01 * math.sin(t * 0.7), 0.02 * math.cos(t), 0.01]),
            "orientation": np.array([0.0, 0.0, 0.0]),
        }

    def think(self) -> RobotDecision:
        """Une boucle cognitive complète : voit, sent, pense, certifie."""
        frame = self.get_frame()
        sensors = self.get_sensors()
        return self.brain.think(frame, sensors)

    def speak(self, text: str) -> str:
        """Synthèse vocale (si dispo) + retour texte."""
        try:
            import subprocess
            subprocess.Popen(["espeak", "-v", "fr", text], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except Exception:
            pass  # pas de espeak — le texte s'affiche à l'écran
        return text

    def disconnect(self):
        if self._cap is not None:
            self._cap.release()
        if self._phone is not None:
            try:
                self._phone.disconnect()
            except Exception:
                pass
        self._connected = False
