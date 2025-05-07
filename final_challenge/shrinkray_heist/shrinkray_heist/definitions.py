# Enum where 
from enum import Enum

class TripSegment(Enum):
    """
    Enum for the current 
    """
    # NOTE doesnt make sense for the way back , could be better organized
    START = 0
    RAY_LOC1 = 1
    RAY_LOC2 = 2
    END = 3
    
    # this is for the way back 
    

class ObjectDetected(Enum):
    """
    Enum for the objects that can be detected by the detector.
    """
    NONE = 0
    TRAFFIC_LIGHT_RED = 1
    TRAFFIC_LIGHT_GREEN = 2
    TRAFFIC_LIGHT_YELLOW = 3
    SHRINK_RAY = 4

class State(Enum):
    """
    Enum that defines the states of the robot
    """
    IDLE = 0
    DETECTING = 1
    PLANNING = 2
    FOLLOWING = 3
    WAITING = 4  
class Target(Enum):
    """
    Enum that defines which node the state machine node is controlling.
    """
    DETECTOR_TRAFFIC_LIGHT = 0
    DETECTOR_SHRINK_RAY = 1
    PLANNER = 2
    FOLLOWER = 3
    
class TrafficSimulation(Enum):
    """
    Enum that defines the traffic simulation state.
    """
    NO_TRAFFIC = 0       # when the car is not near a traffic light
    INCOMING_TRAFFIC = 1 # when the car is approaching a red light 
    ONGOING_TRAFFIC = 2  # when the car is waiting for the traffic light to turn green
    HANDLED_TRAFFIC = 3  # when the car has passed the traffic light

class Drive(Enum):
    """
    Enum that defines which node the state machine node is controlling.
    """
    IN_PROGRESS = 0
    GOAL_REACHED = 1