from collections import defaultdict
from queue import PriorityQueue
import math


class AStar:
    def __init__(self, occupancy_grid, logger, debug=False):
        self.map = occupancy_grid  # Assuming occupancy_grid is a 2D numpy array
        self.R = occupancy_grid.shape[0]
        self.C = occupancy_grid.shape[1]
        self.logger = logger
        self.debug = debug

    def heuristic(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx**2 + dy**2) ** 0.5

    def get_movement_cost(self, current_pos, next_pos):
        x, y = next_pos
        if self.map[y, x] != 0:
            return 100.0  # obstacle penalty

        # base cost = 1 for N/E/S/W, sqrt(2) for diagonals
        dx = abs(current_pos[0] - x)
        dy = abs(current_pos[1] - y)
        return math.sqrt(2) if (dx == 1 and dy == 1) else 1.0
    
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
                    
                
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = backtrack.get(current, None)
            if current is None:
                print("No path found")
                return None  # No path found
        path.append(start)
        path.reverse()
        print("Path found")
        return path


# idea: do some convolution to get the cost of each cell , factoring in the cost of the cells around it (so we avoid being close to obstacles) == dilation lol
