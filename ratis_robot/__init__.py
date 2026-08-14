"""ratis_robot — Robot RATIS souverain (cerveau LCT greffé sur LeRobot).

Le robot qui voit, sent, pense, ressent, parle et certifie — souverain.
"""
from ratis_robot.ratis_brain import RatisBrain, RobotDecision, RobotPerception, RobotEmotion
from ratis_robot.phone_robot import PhoneRobot

__all__ = ["RatisBrain", "PhoneRobot", "RobotDecision", "RobotPerception", "RobotEmotion"]
