
import math
from collections import defaultdict, deque # Added deque
from queue import PriorityQueue
import numpy as np 


class AStar:
    count = 0
    def __init__(self, occupancy_grid, logger, debug=False,
                 wall_penalty_weight=10.0,
                 wall_heuristic_search_radius=5): # Added new parameters
        self.map = occupancy_grid  # Assuming occupancy_grid is a 2D numpy array
        self.R = occupancy_grid.shape[0]
        self.C = occupancy_grid.shape[1]
        self.logger = logger
        self.debug = True
        self.wall_penalty_weight = wall_penalty_weight
        self.wall_heuristic_search_radius = wall_heuristic_search_radius
        AStar.count += 1
        self.wall_dist_table = None # Pre-compute wall distance field

    def set_dist_table(self, dist_table):
        self.wall_dist_table = dist_table

    def _is_valid(self, x, y):
        """Checks if coordinates are within grid boundaries."""
        return 0 <= x < self.C and 0 <= y < self.R
    
    def _precompute_nearest_wall(self):
        """
        Pre-computes a grid where each cell stores the number of grid steps
        to the nearest wall/obstacle. Uses a multi-source BFS.
        """
        if self.debug:
            self.logger.info("Pre-computing wall distance field...")

        # Initialize distance field with infinity, same shape and type as map (or float for inf)
        distance_field = np.full_like(self.map, float('inf'), dtype=float)
        q_bfs = deque()

        # Initialize queue with all wall/obstacle cells.
        # Their distance to the nearest wall is 0.
        # We iterate using (row, col) for direct numpy indexing.
        # (row index corresponds to y, col index corresponds to x)
        for r in range(self.R):  # y
            for c in range(self.C):  # x
                if self.map[r, c] != 0:  # Obstacle cell
                    distance_field[r, c] = 0
                    q_bfs.append(((c, r), 0))  # Enqueue as ((x_pos, y_pos), dist)

        if not q_bfs and self.debug:
             self.logger.info("No obstacles found on map for wall distance field pre-computation.")
        
        processed_bfs_nodes = 0
        while q_bfs:
            (curr_x, curr_y), dist = q_bfs.popleft()
            processed_bfs_nodes += 1

            # Explore 8-connected neighbors
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, 1)]:
                nx, ny = curr_x + dx, curr_y + dy # (x,y) for neighbor position

                if self._is_valid(nx, ny):
                    new_dist_to_wall = dist + 1 # Cost of one step

                    # If we found a shorter path (in terms of steps) to an obstacle for this neighbor
                    # Access distance_field with [row_idx, col_idx] which is [ny, nx]
                    if new_dist_to_wall < distance_field[ny, nx]:
                        distance_field[ny, nx] = new_dist_to_wall
                        q_bfs.append(((nx, ny), new_dist_to_wall))
        
        if self.debug:
            self.logger.info(f"Wall distance field pre-computation complete. Processed {processed_bfs_nodes} BFS nodes.")
            # For very detailed debugging, you might log a sample of the field:
            # if self.R > 0 and self.C > 0:
            #    self.logger.debug(f"Sample of wall_distance_field (top-left 5x5 or less):\n{distance_field[:min(5, self.R), :min(5, self.C)]}")
        return distance_field
    
    def heuristic(self, a, b): # a: current node (x,y), b: goal node (x,y)
        # Standard heuristic (Euclidean distance to goal)
        dx_goal = abs(a[0] - b[0])
        dy_goal = abs(a[1] - b[1])
        h_goal_dist = math.sqrt(dx_goal**2 + dy_goal**2)

        # Wall avoidance penalty calculation
        dist_to_wall = self.wall_dist_table[a[1], a[0]]  # Access distance field with [row_idx, col_idx] which is [y, x]
        
        wall_penalty = 0.0
        # Apply penalty only if a wall is found within the effective search radius
        # (dist_to_wall will be < self.wall_heuristic_search_radius or == self.wall_heuristic_search_radius if found exactly at radius)
        if dist_to_wall > 0 and dist_to_wall < float('inf'): # Effectively dist_to_wall <= self.wall_heuristic_search_radius
            # Inverse distance penalty: closer to wall -> higher penalty
            wall_penalty = self.wall_penalty_weight * (1.0 / dist_to_wall)
        
        
            # self.logger.info(f"Heuristic for node {a} to goal {b}: GoalDist={h_goal_dist:.2f}, DistToWall={dist_to_wall}, WallPenalty={wall_penalty:.2f}, TotalH={(h_goal_dist + wall_penalty):.2f}")
            
        return h_goal_dist + wall_penalty

    def get_movement_cost(self, current_pos, next_pos):
        x, y = next_pos
        # Check if next_pos is valid and not an obstacle
        if not self._is_valid(x, y) or self.map[y, x] != 0:
            return float('inf')  # Strictly impassable

        # Base cost: 1 for N/E/S/W, sqrt(2) for diagonals
        dx_move = abs(current_pos[0] - x)
        dy_move = abs(current_pos[1] - y)
        return math.sqrt(2) if (dx_move == 1 and dy_move == 1) else 1.0
    
    
    def get_neighbors(self, pos):
        """Get valid neighbors of a grid position (8-connected)"""
        x, y = pos
        neighbors = []
        # 8-connected grid
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.C and 0 <= ny < self.R:
                neighbors.append((nx, ny))
        return neighbors

    # a* is djisktra's algorithm with a cost function c(x,y) = g(x,y) + h(x,y)
    # g(x,y) = cost to get to (x,y) from start (minimized by the priority queue)
    # h(x,y) = heuristic cost from (x,y) to goal
    # h(x,y) must never overestimate the cost to get to the goal
    def a_star(self, start, goal):
        q = PriorityQueue()
        q.put((0, start))

        backtrack = {}
        costs = defaultdict(lambda: float("inf"))
        
        # dummy = (float("inf"), float("inf"))
        # backtrack[dummy] = None
        # backtrack[start] = dummy
        costs[start] = 0.0
        # prev = dummy
        visited = set()
        

        while not q.empty():
            
            current_cost, current = q.get()
            if current in visited:
                if self.debug:
                    self.logger.info(f"Already visited: {current}\n")
                continue
            visited.add(current)
            if self.debug:
                self.logger.info(f"current: {current}, cost: {current_cost}")
            
            # backtrack[current] = prev
            # prev = current
            

            if current == goal:
                break

            for next in self.get_neighbors(current):
                move_cost = self.get_movement_cost(current, next)
                new_cost = costs[current] + move_cost

                # this means that we have found a better path to next
                if new_cost < costs[next]:
                    backtrack[next] = current
                    costs[next] = new_cost
                    priority = new_cost + self.heuristic(next, goal)
                    q.put((priority, next))
                    
                
        if self.debug: self.logger.info("Path reconstruction phase.")
        path = []
        curr_path_node = goal

        if costs[goal] == float('inf'):
            if self.debug: self.logger.warning(f"Goal {goal} not reached (g_cost is infinity).")
            print("No path found to goal.")
            return None
        
        if start == goal: # Handle case where start is the goal
            if self.debug: self.logger.info("Start node is the goal node.")
            self.logger.warn("Path found (start is goal).")
            return [start]

        while curr_path_node != start:
            path.append(curr_path_node)
            prev_node = backtrack.get(curr_path_node, None)
            
            if prev_node is None:
                if self.debug: self.logger.error(f"Path reconstruction failed: No backtrack entry for {curr_path_node} (and it's not start).")
                self.logger.warn("Path reconstruction error: Incomplete path.")
                return None 
            curr_path_node = prev_node        
        # path = []
        # current = goal
        # while current != start:
        #     path.append(current)
        #     current = backtrack.get(current, None)
        #     if current is None:
        #         print("No path found")
        #         return None  # No path found
        path.append(start)
        path.reverse()
        print("Path found")
        return path










# from collections import defaultdict
# from queue import PriorityQueue
# import math


# class AStar:
#     def __init__(self, occupancy_grid, logger, debug=False):
#         self.map = occupancy_grid  # Assuming occupancy_grid is a 2D numpy array
#         self.R = occupancy_grid.shape[0]
#         self.C = occupancy_grid.shape[1]
#         self.logger = logger
#         self.debug = debug

#     def heuristic(self, a, b):
#         dx = abs(a[0] - b[0])
#         dy = abs(a[1] - b[1])
#         return (dx**2 + dy**2) ** 0.5

#     def get_movement_cost(self, current_pos, next_pos):
#         x, y = next_pos
#         if self.map[y, x] != 0:
#             return 100.0  # obstacle penalty

#         # base cost = 1 for N/E/S/W, sqrt(2) for diagonals
#         dx = abs(current_pos[0] - x)
#         dy = abs(current_pos[1] - y)
#         return math.sqrt(2) if (dx == 1 and dy == 1) else 1.0
    
#     def get_neighbors(self, pos):
#         """Get valid neighbors of a grid position (8-connected)"""
#         x, y = pos
#         neighbors = []
#         # 8-connected grid
#         for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < self.C and 0 <= ny < self.R:
#                 neighbors.append((nx, ny))
#         return neighbors

#     # a* is djisktra's algorithm with a cost function c(x,y) = g(x,y) + h(x,y)
#     # g(x,y) = cost to get to (x,y) from start (minimized by the priority queue)
#     # h(x,y) = heuristic cost from (x,y) to goal
#     # h(x,y) must never overestimate the cost to get to the goal
#     def a_star(self, start, goal):
#         q = PriorityQueue()
#         q.put((0, start))

#         backtrack = {}
#         costs = defaultdict(lambda: float("inf"))
        
#         # dummy = (float("inf"), float("inf"))
#         # backtrack[dummy] = None
#         # backtrack[start] = dummy
#         costs[start] = 0.0
#         # prev = dummy
#         visited = set()
        

#         while not q.empty():
            
#             current_cost, current = q.get()
#             if current in visited:
#                 if self.debug:
#                     self.logger.info(f"Already visited: {current}\n")
#                 continue
#             visited.add(current)
#             if self.debug:
#                 self.logger.info(f"current: {current}, cost: {current_cost}")
            
#             # backtrack[current] = prev
#             # prev = current
            

#             if current == goal:
#                 break

#             for next in self.get_neighbors(current):
#                 move_cost = self.get_movement_cost(current, next)
#                 new_cost = costs[current] + move_cost

#                 # this means that we have found a better path to next
#                 if new_cost < costs[next]:
#                     backtrack[next] = current
#                     costs[next] = new_cost
#                     priority = new_cost + self.heuristic(next, goal)
#                     q.put((priority, next))
                    
                
#         path = []
#         current = goal
#         while current != start:
#             path.append(current)
#             current = backtrack.get(current, None)
#             if current is None:
#                 print("No path found")
#                 return None  # No path found
#         path.append(start)
#         path.reverse()
#         print("Path found")
#         return path


# # idea: do some convolution to get the cost of each cell , factoring in the cost of the cells around it (so we avoid being close to obstacles) == dilation lol
