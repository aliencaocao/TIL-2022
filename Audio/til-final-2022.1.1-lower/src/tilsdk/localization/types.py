import numpy as np
from typing import Any, Optional, Tuple, List, Union, NamedTuple, overload
from scipy.ndimage import distance_transform_edt

### Consts and Types ####

_SQRT2 = 1.4142135623730951

class GridLocation(NamedTuple):
    '''Pixel coordinates (x, y)'''

    x: int
    '''X-coordinate.'''

    y: int
    '''Y-coordinate.'''

    def __add__(self, other):
        return type(self)(*[self[i]+other[i] for i in range(len(self))])

    def __sub__(self, other):
        return type(self)(*[self[i]-other[i] for i in range(len(self))])

    def __mul__(self, other):
        return type(self)(*[e*other for e in self])

    def __truediv__(self, other:Union[float, int]):
        return type(self)(*[e/other for e in self])


class RealLocation(NamedTuple):
    '''Pixel coordinates (x, y)'''

    x: float
    '''X-coordinate.'''

    y: float
    '''Y-coordinate.'''

    def __add__(self, other):
        return type(self)(*[self[i]+other[i] for i in range(len(self))])

    def __sub__(self, other):
        return type(self)(*[self[i]-other[i] for i in range(len(self))])

    def __mul__(self, other):
        return type(self)(*[e*other for e in self])

    def __truediv__(self, other:Union[float, int]):
        return type(self)(*[e/other for e in self])

class GridPose(NamedTuple):
    '''Pixel coordinates (x, y, z) where z is angle from x-axis in deg.'''

    x: int
    '''X-coordinate.'''

    y: int
    '''Y-coordinate.'''

    z: float
    '''Heading angle (rel. x-axis) in degrees.'''

    def __add__(self, other):
        return type(self)(*[self[i]+other[i] for i in range(len(self))])

    def __sub__(self, other):
        return type(self)(*[self[i]-other[i] for i in range(len(self))])

    def __mul__(self, other):
        return type(self)(*[e*other for e in self])

    def __truediv__(self, other:Union[float, int]):
        return type(self)(*[e/other for e in self])

class RealPose(NamedTuple):
    '''Real coordinates (x, y, z) where z is angle from x-axis in deg.'''
    
    x: float
    '''X-coordinate.'''

    y: float
    '''Y-coordinate.'''

    z: float
    '''Heading angle (rel. x-axis) in degrees.'''

    def __add__(self, other):
        return type(self)(*[self[i]+other[i] for i in range(len(self))])

    def __sub__(self, other):
        return type(self)(*[self[i]-other[i] for i in range(len(self))])

    def __mul__(self, other):
        return type(self)(*[e*other for e in self])

    def __truediv__(self, other:Union[float, int]):
        return type(self)(*[e/other for e in self])

class Clue(NamedTuple):
    '''Clue'''

    clue_id: int
    '''Unique clue id.'''

    location: RealLocation
    '''Associated location.'''

    audio: bytes
    '''Clue audio data.'''

class SignedDistanceGrid:
    '''Grid map representation.

    Grid elements are square and represented by a float.
    Value indicates distance from nearest obstacle.
    Value <= 0 indicates occupied, > 0 indicates passable.

    Grid is centered-aligned, i.e. real-world postion
    corresponds to center of grid square.
    '''

    def __init__(self, width:int=0, height:int=0, grid:Optional[Any]=None, scale:float=1.0):
        '''
        Parameters
        ----------
        width : int
            Width of map in number of grid elements, corresponding to real-world x-axis. Ignored if grid parameter is specified.
        height : int
            Height of map in number of grid elements, corresponding to real-world y-axis. Ignored if grid parameter is specified.
        grid : nxm ArrayLike
            Numpy array of grid data, corresponding to a grid of width m and heigh n.
        scale : float
            Ratio of real-world unit to grid unit.
        '''

        self.scale = scale

        if grid is not None:
            self.grid = grid
            self.width = grid.shape[1]
            self.height = grid.shape[0]
        else:
            self.grid = np.inf((height, width), dtype=float)

    @staticmethod
    def from_image(img:Any, scale:float=1.0):
        '''Factory method to create map from image.
        
        Only the first channel is used. Channel value should be 0 where passable.

        Parameters
        ----------
        img : Any
            Input image.
        scale : float
            Ratio of real-world unit to grid unit.

        Returns
        -------
        map : SignedDistanceGrid
        '''
        bin_img = img[:,:,0] > 0
        grid = distance_transform_edt(1-bin_img) - distance_transform_edt(bin_img) 

        return SignedDistanceGrid(grid=grid, scale=scale)

    def in_bounds(self, id:GridLocation) -> bool:
        '''Check if grid location is in bounds.
        
        Parameters
        ----------
        id : GridLocation
            Input location.
        
        Returns
        -------
        bool
            True if location is in bounds.
        '''
        x, y, = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id:GridLocation) -> bool:
        '''Check if grid location is passable.
        
        Parameters
        ----------
        id : GridLocation
            Input location.
        
        Returns
        -------
        bool
            True if location is in passable.
        '''
        return self.grid[id[1], id[0]] > 0

    def neighbours(self, id:GridLocation) -> List[Tuple[GridLocation, float]]:
        '''Get valid neighbours and cost of grid location.
        
        Parameters
        ----------
        id : GridLocation
            Input location.
        
        Returns
        -------
        neighbours : List[Tuple[GridLocation, float]]
            List of tuples of neighbouring locations and the costs to those locations.
        '''
        x, y = id
        neighbours = [
            (GridLocation(x-1, y-1), _SQRT2), # NW
            (GridLocation(x  , y-1), 1    ), # N
            (GridLocation(x+1, y-1), _SQRT2), # NE
            (GridLocation(x-1, y  ), 1    ), # W
            (GridLocation(x+1, y  ), 1    ), # E
            (GridLocation(x-1, y+1), _SQRT2), # SW
            (GridLocation(x  , y+1), 1    ), # S
            (GridLocation(x+1, y+1), _SQRT2), # SE
        ]

        results = filter(lambda n: self.in_bounds(n[0]), neighbours)
        results = filter(lambda n: self.passable(n[0]), results)
        results = [(*r, self.grid[r[0][1], r[0][0]]) for r in results]
        return results

    def real_to_grid(self, id:RealLocation) -> GridLocation:
        '''Convert real coordinates to grid coordinates.
        
        Parameters
        ----------
        id : RealLocation
            Input location.
        
        Returns
        -------
        GridLocation
            Corresponding GridLocation.
        '''
        return real_to_grid(id, self.scale)

    def grid_to_real(self, id:GridLocation) -> RealLocation:
        '''Convert grid coordinates to real coordinates.
        
        Parameters
        ----------
        id : GridLocation
            Input location.
        
        Returns
        -------
        RealLocation
            Corresponding RealLocation.
        '''
        return grid_to_real(id, self.scale)

    def dilated(self, distance:float):
        '''Dilate obstacles in grid.
        
        Parameters
        ----------
        distance : float
            Size of dilation.

        Returns
        -------
        SignedDistanceGrid
            Grid with dilated obstacles.
        '''
        grid = self.grid - distance
        return SignedDistanceGrid(grid=grid, scale=self.scale)

#### Helper functions ####

def euclidean_distance(a:Union[RealLocation, RealPose], b:Union[RealLocation, RealPose]) -> float:
    '''Compute the Euclidean distance between points.
    
    Parameters
    ----------
    a
        First point.
    b
        Second point.
    
    Returns
    -------
    float
        Euclidean distance between points.
    '''
    return np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

@overload
def real_to_grid(id:RealLocation, scale:float) -> GridLocation: ...

@overload
def real_to_grid(id:RealPose, scale:float) -> GridPose: ...

def real_to_grid(id:Union[RealLocation, RealPose], scale:float) -> Union[GridLocation, GridPose]:
        '''Convert real coordinates to grid coordinates.
        
        .. note::
            Grid coordinates are discretized. To get non discretized grid coordinates, see :meth:`real_to_grid_exact`.
        
        Parameters
        ----------
        id
            Input location/pose.

        Returns
        -------
        output
            Corresponding gird location/pose.
        '''
        if len(id) == 3:
            return GridPose(int(np.round(id[0]/scale)), int(np.round(id[1])/scale), id[2])
        return GridLocation(int(np.round(id[0]/scale)), int(np.round(id[1]/scale)))

def real_to_grid_exact(id:RealLocation, scale:float) -> Tuple[float, float]:
    '''Convert real coordinates to grid coordinates without discretization.
    
    Parameters
    ----------
    id
        Input location.
    scale
        Ratio of real-world unit to grid unit.

    Returns
    -------
    Tuple[float, float]
        Grid location without discretization.
    '''
    return (id[0]/scale, id[1]/scale)

@overload
def grid_to_real(id:GridLocation, scale:float) -> RealLocation: ...

@overload
def grid_to_real(id:GridPose, scale:float) -> RealPose: ...

def grid_to_real(id:Union[GridLocation, GridPose], scale:float) -> Union[RealLocation, RealPose]:
    '''Convert grid coordinates to real coordinates.

    Parameters
    ----------
    id
        Input location/pose.

    Returns
    -------
    output
        Corresponding real location/pose.
    '''
    if len(id) == 3:
        return RealPose(id[0]*scale, id[1]*scale, id[2])
    return RealLocation(id[0]*scale, id[1]*scale)

