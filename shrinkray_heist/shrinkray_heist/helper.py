from math import atan2, cos, sin


def world_to_grid(x, y, map_info, logger, debug=False):
    """Convert world coordinates to grid coordinates correctly handling transforms"""
    try:
        # Get origin pose
        origin_x = map_info.origin.position.x
        origin_y = map_info.origin.position.y
        # Get orientation
        qz = map_info.origin.orientation.z
        qw = map_info.origin.orientation.w
        # Calculate rotation angle
        angle = 2 * atan2(qz, qw)

        # Transform point to map frame
        dx = x - origin_x
        dy = y - origin_y

        # Rotate point
        x_rot = dx * cos(-angle) - dy * sin(-angle)
        y_rot = dx * sin(-angle) + dy * cos(-angle)

        # Convert to grid coordinates
        grid_x = int(x_rot / map_info.resolution)
        grid_y = int(y_rot / map_info.resolution)

        # Log transformation for debugging
        logger.debug(
            f"World ({x:.2f}, {y:.2f}) -> Grid ({grid_x}, {grid_y})\n"
            f"Origin: ({origin_x:.2f}, {origin_y:.2f}), Angle: {angle:.2f}"
        )

        return grid_x, grid_y

    except Exception as e:
        if debug:
            logger.error(f"World to grid transform error: {str(e)}")
        return None


"""
Convert world coordinates (x,y) in meters from the center of the map 
To the grid coordinates (u,v) in cells from the origin of the map
"""


def grid_to_world(grid_x, grid_y, map_info, logger, debug=False):
    """Convert grid coordinates back to world coordinates"""
    try:
        # Get origin pose
        origin_x = map_info.origin.position.x
        origin_y = map_info.origin.position.y
        # Get orientation
        qz = map_info.origin.orientation.z
        qw = map_info.origin.orientation.w
        # Calculate rotation angle
        angle = 2 * atan2(qz, qw)

        # Convert grid to local coordinates
        x_local = grid_x * map_info.resolution
        y_local = grid_y * map_info.resolution

        # Rotate back
        x_rot = x_local * cos(angle) - y_local * sin(angle)
        y_rot = x_local * sin(angle) + y_local * cos(angle)

        # Transform to world frame
        x = x_rot + origin_x
        y = y_rot + origin_y

        return x, y

    except Exception as e:
        if debug:
            logger.error(f"Grid to world transform error: {str(e)}")
        return None
