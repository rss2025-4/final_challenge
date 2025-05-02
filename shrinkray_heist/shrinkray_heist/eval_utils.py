from dataclasses import dataclass
from typing import Any

from rclpy import Context
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)


@dataclass
class InjectedConfig:
    _context: Context | None = None
    _use_strict_qos: bool = False

    def extra_node_kwargs(self) -> dict[str, Any]:
        return {"context": self._context}

    def goal_and_pose_qos(self):
        if self._use_strict_qos:
            return QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=20,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            )
        else:
            return QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
