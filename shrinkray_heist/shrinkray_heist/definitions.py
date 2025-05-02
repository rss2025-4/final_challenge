# Enum where 
from enum import Enum

class States(Enum):
    """
    Enum for the states of the robot.
    """
    ON = 0
    OFF = 1
class Target(Enum):
    """
    Enum that defines which node the state machine node is controlling.
    """
    DETECTOR = 0
    PLANNER = 1
    BASEMENT = 2
    TRAFFIC_LIGHT = 3
    SHRINK_RAY = 4