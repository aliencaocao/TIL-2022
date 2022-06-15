from typing import List, Tuple, TypeVar, Dict
import numpy as np
from tilsdk.localization import *

T = TypeVar('T')


class MyPlanner:
    def __init__(self, map_: SignedDistanceGrid = None, waypoint_sparsity=0.5, optimize_threshold=3, consider=4):
        '''
        Parameters
        ----------
        map : SignedDistanceGrid
            Distance grid map
        sdf_weight: float
            Relative weight of distance in cost function.
        waypoint_sparsity: float
            0.5 results in every 50th waypoint being taken at scale=0.01 and 10th at scale=0.05
        consider: float
            For the get_explore function only. See there for more details.
        '''
        self.optimize_threshold = optimize_threshold
        self.map = map_
        self.bgrid = self.transform_add_border(map_.grid.copy())  # Grid which takes the borders into account
        self.astar_grid = self.transform_for_astar(self.bgrid.copy())
        self.waypoint_sparsity = waypoint_sparsity
        self.big_grid = [[0 for j in range(14)] for i in range(10)]  # Big_grid stores whether each 0.5*0.5m tile of the arena has been visited
        self.big_grid_centre = [[0 for j in range(14)] for i in range(10)]
        self.consider = consider
        for i in range(10):
            for j in range(14):
                self.big_grid_centre[i][j] = RealLocation(j + 0.25, i + 0.25)

    def transform_add_border(self, og_grid):
        grid = og_grid.copy()
        a, b = grid.shape
        for i in range(a):
            for j in range(b):
                grid[i][j] = min(grid[i][j], i + 1, a - i, j + 1, b - j)
        return grid

    def transform_for_astar(self, grid):
        # !-- Possible to edit this transform if u want
        grid2 = grid.copy()
        grid2[grid2 > 0] = 1 + 100 / (grid2[grid2 > 0])
        grid2[grid2 <= 0] = np.inf
        return grid2.astype("float32")

    @staticmethod
    def big_grid_of(l: RealLocation):  # Returns the big grid array indices of a real location
        return int(l[0] // 0.5), int(l[1] // 0.5)

    def visit(self, l: RealLocation):
        indices = self.big_grid_of(l)
        self.big_grid[indices[0]][indices[1]] = max(1, self.big_grid[indices[0]][indices[1]])

    def get_explore(self, l: RealLocation, debug: bool = False):  # Call this to get a location to go to if there are no locations of interest left
        # debug: Whether to plot maps and show info
        # consider (in __init__): Consider the astar paths of this number of closest unvisited cells by euclidean distance
        # Larger number gives better performance but slower
        m = 100
        for i in range(10):
            for j in range(14):
                m = min(m, self.big_grid[i][j])
        if m == 1:  # Can comment out this part if u want the robot to vroom around infinitely
            return None

        closeness = []
        for i in range(10):
            for j in range(14):
                if self.big_grid[i][j] == m:
                    closeness.append((self.heuristic(self.big_grid_centre[i][j], l), (i, j)))
        closeness.sort()

        if len(closeness) == 0:
            return None

        closeness = closeness[:min(self.consider, len(closeness))]
        for i in range(len(closeness)):
            loc = self.big_grid_centre[closeness[i][1][0]][closeness[i][1][1]]
            if debug:
                print("l, loc:", l, loc)
            path = self.plan(l, loc, whole_path=True, display=debug)
            closeness[i] = (1e18 if type(path) == type(None) else len(path), closeness[i][1])
            if debug:
                print("Path length:", closeness[i][0])

        closeness.sort()
        if debug:
            print("Closest guys", closeness)

        closest = closeness[0]
        self.big_grid[closest[1][0]][closest[1][1]] += 1

        if debug:
            plt.imshow(self.big_grid)
            plt.title("Big grid now")
            plt.show()
        return self.big_grid_centre[closest[1][0]][closest[1][1]]

    def heuristic(self, a: GridLocation, b: GridLocation) -> float:
        '''Planning heuristic function.
        Parameters
        ----------
        a: GridLocation
            Starting location.
        b: GridLocation
            Goal location.
        '''
        return euclidean_distance(a, b)

    def nearest_clear(self, loc, passable):
        '''Utility function to find the nearest clear cell to a blocked cell'''
        loc = loc[::-1]
        if not passable[loc]:
            best = (1e18, (-1, -1))
            for i in range(map_.height):  # y
                for j in range(map_.width):  # x
                    if map_.grid[(i, j)] > 0:
                        best = min(best, (self.heuristic(GridLocation(i, j), loc), (i, j)))
            loc = best[1]
        return loc

    def plan(self, start: RealLocation, goal: RealLocation, whole_path: bool = False, display: bool = True) -> List[RealLocation]:
        '''Plan in real coordinates.

        Raises NoPathFileException path is not found.

        Parameters
        ----------
        start: RealLocation
            Starting location.
        goal: RealLocation
            Goal location.
        whole_path:
            Whether to return the whole path instead of version with select waypoints
        display:
            Whether to visualise the path

        Returns
        -------
        path
            List of RealLocation from start to goal.
        '''

        path = self.plan_grid(self.map.real_to_grid(start), self.map.real_to_grid(goal), whole_path)
        if path is None:
            return path
        if display:
            visualise_path(path, self.map)
        path = [self.map.grid_to_real(wp) for wp in path]
        return path

    def plan_grid(self, start: GridLocation, goal: GridLocation, whole_path: bool = False, debug=False) -> List[GridLocation]:
        '''Plan in grid coordinates.

        Raises NoPathFileException path is not found.

        Parameters
        ----------
        start: GridLocation
            Starting location.
        goal: GridLocation
            Goal location.
        whole_path:
            Whether to return the whole path instead of version with select waypoints
        debug:
            Whether to print start and end locations
        Returns
        -------
        path
            List of GridLocation from start to goal.
        '''

        if not self.map:
            raise RuntimeError('Planner map is not initialized.')

        start = start[1], start[0]
        goal = goal[1], goal[0]  # Use i=x,j=y convention for convenience
        passable = self.map.grid > 0

        if debug:
            print("original start", start)
            print("original goal", goal)
        start = self.nearest_clear(start, passable)
        goal = self.nearest_clear(goal, passable)
        if debug:
            print("start", start)
            print("goal", goal)

        # astar
        path = pyastar2d.astar_path(self.astar_grid, start, goal, allow_diagonal=True)
        if path is None:
            return None
        coeff = int(self.waypoint_sparsity / self.map.scale)  # default sparsity 0.5 --> 50 for 0.01, 10 for 0.05
        path = list(path)
        path = [(x[1], x[0]) for x in path]
        if whole_path:
            return path
        coeff = max(coeff, 1)
        path = path[:1] + path[:-1:coeff] + path[-1:]  # Take the 1st, last, and every 20th waypoint in the middle
        path.append(path[-1])  # Duplicate last waypoint to avoid bug in main loop
        return self.optimize_path(path)

    def optimize_path(self, path: List[GridLocation]) -> List[GridLocation]:
        new_path = [path[0]]  # starting point always in path
        for i in range(1, len(path) - 1, 1):
            if not ((abs(path[i - 1][0] - path[i][0]) < self.optimize_threshold and abs(path[i][0] - path[i + 1][0]) < self.optimize_threshold) or (
                    abs(path[i - 1][1] - path[i][1]) < self.optimize_threshold and abs(
                    path[i][1] - path[i + 1][1]) < self.optimize_threshold)):  # 3 consecutive points are on a straight line in either x or y direction
                new_path.append(path[i])
        new_path.append(path[-1])  # add last point
        return new_path