# Enum where 
from enum import Enum

class TripSegment(Enum):
    """
    Enum for the current 
    """
    # NOTE doesnt make sense for the way back , could be better organized
    START = -1 
    RAY_LOC1 = 0
    RAY_OBJ1 = 1
    RAY_LOC2 = 2
    RAY_OBJ2 = 3
    
    # this is for the way back 
    
class Target(Enum):
    """
    Enum that defines which node the state machine node is controlling.
    """
    DETECTOR = 0
    PLANNER = 1
    BASEMENT = 2
    TRAFFIC_LIGHT = 3
    SHRINK_RAY = 4