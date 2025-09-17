"""Edge and HalfEdge classes for DCEL representation."""

from typing import Optional, List
from .point import Point
from .vertex import Vertex


class HalfEdge:
    """
    Half-edge data structure for Doubly Connected Edge List (DCEL).
    Essential for efficient Voronoi diagram representation.
    """

    def __init__(self):
        # Topological relationships
        self.origin: Optional[Vertex] = None
        self.destination: Optional[Vertex] = None
        self.twin: Optional[HalfEdge] = None
        self.next: Optional[HalfEdge] = None
        self.prev: Optional[HalfEdge] = None
        self.face: Optional['Face'] = None  # Forward reference

        # Geometric properties
        self.direction: Optional[Point] = None

        # For Fortune's algorithm
        self.left_site: Optional[Point] = None
        self.right_site: Optional[Point] = None

        # Unique identifier
        self.id: int = HalfEdge._next_id
        HalfEdge._next_id += 1

    _next_id: int = 0

    def __str__(self) -> str:
        origin_str = str(self.origin.point) if self.origin else "None"
        dest_str = str(self.destination.point) if self.destination else "None"
        return f"HalfEdge({origin_str} -> {dest_str})"

    def __repr__(self) -> str:
        return f"HalfEdge(id={self.id}, origin={self.origin}, dest={self.destination})"

    @property
    def is_complete(self) -> bool:
        """Check if half-edge has both origin and destination vertices."""
        return self.origin is not None and self.destination is not None

    @property
    def is_infinite(self) -> bool:
        """Check if half-edge is infinite (missing an endpoint)."""
        return self.origin is None or self.destination is None

    @property
    def vector(self) -> Optional[Point]:
        """Get the vector from origin to destination."""
        if not self.is_complete:
            return None
        return self.destination.point - self.origin.point

    @property
    def length(self) -> Optional[float]:
        """Get the length of the edge."""
        if not self.is_complete:
            return None
        return self.origin.point.distance_to(self.destination.point)

    @property
    def midpoint(self) -> Optional[Point]:
        """Get the midpoint of the edge."""
        if not self.is_complete:
            return None
        return (self.origin.point + self.destination.point) / 2

    def set_twin(self, twin: 'HalfEdge') -> None:
        """Set the twin relationship bidirectionally."""
        self.twin = twin
        twin.twin = self

        # Ensure consistent site assignments
        if self.left_site and self.right_site:
            twin.left_site = self.right_site
            twin.right_site = self.left_site

    def set_endpoints(self, origin: Vertex, destination: Vertex) -> None:
        """Set both endpoints of the half-edge."""
        self.origin = origin
        self.destination = destination

        # Add this half-edge to the origin vertex's incident edges
        if origin:
            origin.add_incident_edge(self)

    def bisects(self, site1: Point, site2: Point) -> bool:
        """Check if this edge bisects the given two sites."""
        return ((self.left_site == site1 and self.right_site == site2) or
                (self.left_site == site2 and self.right_site == site1))


class Edge:
    """
    Full edge representation consisting of two half-edges.
    Represents a bisector between two Voronoi sites.
    """

    def __init__(self, site1: Point, site2: Point):
        self.site1 = site1
        self.site2 = site2

        # Create twin half-edges
        self.half_edge1 = HalfEdge()
        self.half_edge2 = HalfEdge()

        # Set up twin relationship
        self.half_edge1.set_twin(self.half_edge2)

        # Set site information
        self.half_edge1.left_site = site1
        self.half_edge1.right_site = site2
        self.half_edge2.left_site = site2
        self.half_edge2.right_site = site1

        # Unique identifier
        self.id: int = Edge._next_id
        Edge._next_id += 1

    _next_id: int = 0

    def __str__(self) -> str:
        return f"Edge(bisector of {self.site1} and {self.site2})"

    def __repr__(self) -> str:
        return f"Edge(id={self.id}, site1={self.site1}, site2={self.site2})"

    @property
    def is_complete(self) -> bool:
        """Check if both half-edges are complete."""
        return self.half_edge1.is_complete and self.half_edge2.is_complete

    @property
    def vertices(self) -> List[Optional[Vertex]]:
        """Get the vertices of this edge."""
        vertices = []
        if self.half_edge1.origin:
            vertices.append(self.half_edge1.origin)
        if self.half_edge1.destination:
            vertices.append(self.half_edge1.destination)
        return list(set(vertices))  # Remove duplicates

    def get_perpendicular_bisector_line(self) -> tuple[Point, Point]:
        """
        Get the perpendicular bisector line parameters.
        Returns (midpoint, direction_vector).
        """
        midpoint = (self.site1 + self.site2) / 2

        # Direction perpendicular to the line connecting the sites
        site_vector = self.site2 - self.site1
        direction = site_vector.perpendicular().normalize()

        return midpoint, direction

    def contains_vertex(self, vertex: Vertex) -> bool:
        """Check if the edge contains the given vertex."""
        return (self.half_edge1.origin == vertex or
                self.half_edge1.destination == vertex or
                self.half_edge2.origin == vertex or
                self.half_edge2.destination == vertex)

    def set_vertex(self, vertex: Vertex, is_start: bool = True) -> None:
        """
        Set a vertex as either the start or end of the edge.

        Args:
            vertex: The vertex to set
            is_start: If True, sets as start vertex; if False, sets as end vertex
        """
        if is_start:
            self.half_edge1.origin = vertex
            self.half_edge2.destination = vertex
        else:
            self.half_edge1.destination = vertex
            self.half_edge2.origin = vertex

        # Add to vertex's incident edges
        vertex.add_incident_edge(self.half_edge1)
        vertex.add_incident_edge(self.half_edge2)