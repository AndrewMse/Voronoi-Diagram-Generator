"""Face class for DCEL representation of Voronoi diagram."""

from typing import List, Optional, Iterator
from .point import Point


class Face:
    """
    A face in the DCEL representation.
    In Voronoi diagrams, faces correspond to Voronoi cells (regions around sites).
    """

    def __init__(self, site: Optional[Point] = None):
        # The site that this face represents (generator point)
        self.site = site

        # One of the half-edges on the boundary of this face
        self.outer_component: Optional['HalfEdge'] = None

        # List of inner components (holes) - typically empty for Voronoi diagrams
        self.inner_components: List['HalfEdge'] = []

        # For unbounded faces
        self.is_unbounded: bool = False

        # Unique identifier
        self.id: int = Face._next_id
        Face._next_id += 1

    _next_id: int = 0

    def __str__(self) -> str:
        site_str = str(self.site) if self.site else "unbounded"
        return f"Face({site_str})"

    def __repr__(self) -> str:
        return f"Face(id={self.id}, site={self.site}, bounded={not self.is_unbounded})"

    @property
    def is_outer_face(self) -> bool:
        """Check if this is the outer (unbounded) face."""
        return self.site is None

    def get_boundary_edges(self) -> List['HalfEdge']:
        """Get all half-edges on the boundary of this face."""
        if not self.outer_component:
            return []

        edges = []
        current = self.outer_component

        # Traverse the boundary
        while True:
            edges.append(current)
            current = current.next

            # Stop when we complete the cycle
            if current == self.outer_component or current is None:
                break

            # Safety check to avoid infinite loops
            if len(edges) > 1000:  # Reasonable upper bound
                break

        return edges

    def get_boundary_vertices(self) -> List['Vertex']:
        """Get all vertices on the boundary of this face."""
        from .vertex import Vertex

        vertices = []
        for edge in self.get_boundary_edges():
            if edge.origin:
                vertices.append(edge.origin)

        return vertices

    def get_boundary_points(self) -> List[Point]:
        """Get all boundary points of this face."""
        return [vertex.point for vertex in self.get_boundary_vertices()]

    def area(self) -> float:
        """
        Calculate the area of this face using the shoelace formula.
        Returns 0 for unbounded faces.
        """
        if self.is_unbounded or not self.outer_component:
            return 0.0

        points = self.get_boundary_points()
        if len(points) < 3:
            return 0.0

        # Shoelace formula
        area = 0.0
        n = len(points)

        for i in range(n):
            j = (i + 1) % n
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y

        return abs(area) / 2.0

    def centroid(self) -> Optional[Point]:
        """
        Calculate the centroid of this face.
        Returns None for unbounded faces.
        """
        if self.is_unbounded or not self.outer_component:
            return None

        points = self.get_boundary_points()
        if not points:
            return None

        # Simple centroid calculation (arithmetic mean of vertices)
        # For more accuracy with non-convex polygons, could use area-weighted centroid
        sum_x = sum(p.x for p in points)
        sum_y = sum(p.y for p in points)
        return Point(sum_x / len(points), sum_y / len(points))

    def contains_point(self, point: Point) -> bool:
        """
        Check if a point lies inside this face using the ray casting algorithm.
        Returns False for unbounded faces.
        """
        if self.is_unbounded or not self.outer_component:
            return False

        points = self.get_boundary_points()
        if len(points) < 3:
            return False

        # Ray casting algorithm
        x, y = point.x, point.y
        n = len(points)
        inside = False

        p1x, p1y = points[0].x, points[0].y
        for i in range(1, n + 1):
            p2x, p2y = points[i % n].x, points[i % n].y

            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def distance_to_point(self, point: Point) -> float:
        """
        Calculate the minimum distance from a point to this face.
        For bounded faces, returns the distance to the boundary.
        """
        if self.is_unbounded:
            return float('inf')

        points = self.get_boundary_points()
        if not points:
            return float('inf')

        # If point is inside, distance is 0
        if self.contains_point(point):
            return 0.0

        # Find minimum distance to boundary edges
        min_distance = float('inf')
        n = len(points)

        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]

            # Distance from point to line segment
            distance = self._point_to_segment_distance(point, p1, p2)
            min_distance = min(min_distance, distance)

        return min_distance

    def _point_to_segment_distance(self, point: Point, seg_start: Point, seg_end: Point) -> float:
        """Calculate the shortest distance from a point to a line segment."""
        # Vector from seg_start to seg_end
        seg_vec = seg_end - seg_start
        seg_length_sq = seg_vec.dot(seg_vec)

        if seg_length_sq < 1e-12:  # Degenerate segment
            return point.distance_to(seg_start)

        # Vector from seg_start to point
        point_vec = point - seg_start

        # Project point onto the line segment
        t = max(0, min(1, point_vec.dot(seg_vec) / seg_length_sq))

        # Find the closest point on the segment
        closest_point = seg_start + seg_vec * t

        return point.distance_to(closest_point)

    def set_boundary_chain(self, start_edge: 'HalfEdge') -> None:
        """
        Set the outer boundary of this face and update all edges' face references.
        """
        self.outer_component = start_edge

        # Update face reference for all boundary edges
        current = start_edge
        while True:
            current.face = self
            current = current.next

            if current == start_edge or current is None:
                break