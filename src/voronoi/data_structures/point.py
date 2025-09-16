"""Point class for geometric operations."""

from typing import Union, Tuple
import math
from numba import jit


class Point:
    """A 2D point with high-precision coordinates and optimized operations."""

    __slots__ = ('x', 'y', '_hash')

    def __init__(self, x: Union[float, int], y: Union[float, int]):
        self.x = float(x)
        self.y = float(y)
        self._hash = None

    def __eq__(self, other: 'Point') -> bool:
        if not isinstance(other, Point):
            return False
        return abs(self.x - other.x) < 1e-10 and abs(self.y - other.y) < 1e-10

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((round(self.x, 10), round(self.y, 10)))
        return self._hash

    def __repr__(self) -> str:
        return f"Point({self.x:.6f}, {self.y:.6f})"

    def __str__(self) -> str:
        return f"({self.x:.3f}, {self.y:.3f})"

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: Union[float, int]) -> 'Point':
        return Point(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: Union[float, int]) -> 'Point':
        return self * scalar

    def __truediv__(self, scalar: Union[float, int]) -> 'Point':
        return Point(self.x / scalar, self.y / scalar)

    @property
    def magnitude(self) -> float:
        """Calculate the magnitude (distance from origin) of the point."""
        return math.sqrt(self.x * self.x + self.y * self.y)

    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def distance_squared_to(self, other: 'Point') -> float:
        """Calculate squared Euclidean distance (faster when comparison is needed)."""
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy

    def dot(self, other: 'Point') -> float:
        """Calculate dot product with another point (treated as vector)."""
        return self.x * other.x + self.y * other.y

    def cross(self, other: 'Point') -> float:
        """Calculate cross product (z-component) with another point."""
        return self.x * other.y - self.y * other.x

    def normalize(self) -> 'Point':
        """Return normalized point (unit vector)."""
        mag = self.magnitude
        if mag < 1e-10:
            return Point(0, 0)
        return Point(self.x / mag, self.y / mag)

    def rotate(self, angle: float) -> 'Point':
        """Rotate point by angle (in radians) around origin."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Point(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )

    def perpendicular(self) -> 'Point':
        """Return perpendicular vector (rotated 90 degrees counter-clockwise)."""
        return Point(-self.y, self.x)

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple for use with other libraries."""
        return (self.x, self.y)

    @classmethod
    def from_tuple(cls, coords: Tuple[Union[float, int], Union[float, int]]) -> 'Point':
        """Create point from tuple."""
        return cls(coords[0], coords[1])


@jit(nopython=True)
def point_distance_squared(x1: float, y1: float, x2: float, y2: float) -> float:
    """Optimized squared distance calculation using Numba."""
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy


@jit(nopython=True)
def point_cross_product(x1: float, y1: float, x2: float, y2: float) -> float:
    """Optimized cross product calculation using Numba."""
    return x1 * y2 - y1 * x2


@jit(nopython=True)
def point_orientation(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> int:
    """
    Find orientation of ordered triplet (p, q, r).
    Returns:
        0 -> p, q and r are colinear
        1 -> Clockwise
        2 -> Counterclockwise
    """
    val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
    if abs(val) < 1e-10:
        return 0  # collinear
    return 1 if val > 0 else 2  # clockwise or counterclockwise