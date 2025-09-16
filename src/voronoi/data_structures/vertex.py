"""Vertex class for Voronoi diagram representation."""

from typing import List, Optional, TYPE_CHECKING
from .point import Point

if TYPE_CHECKING:
    from .edge import HalfEdge


class Vertex:
    """
    A vertex in the Voronoi diagram.
    Represents intersection points of Voronoi edges (circumcenters of Delaunay triangles).
    """

    def __init__(self, point: Point):
        self.point = point
        self.incident_edges: List['HalfEdge'] = []

        # For Fortune's algorithm
        self.is_at_infinity: bool = False

        # Unique identifier
        self.id: int = Vertex._next_id
        Vertex._next_id += 1

    _next_id: int = 0

    def __str__(self) -> str:
        return f"Vertex{self.point}"

    def __repr__(self) -> str:
        return f"Vertex(id={self.id}, point={self.point}, degree={self.degree})"

    def __eq__(self, other: 'Vertex') -> bool:
        if not isinstance(other, Vertex):
            return False
        return self.point == other.point

    def __hash__(self) -> int:
        return hash(self.point)

    @property
    def degree(self) -> int:
        """Get the degree of the vertex (number of incident edges)."""
        return len(self.incident_edges)

    @property
    def x(self) -> float:
        """Get x-coordinate."""
        return self.point.x

    @property
    def y(self) -> float:
        """Get y-coordinate."""
        return self.point.y

    def add_incident_edge(self, edge: 'HalfEdge') -> None:
        """Add an incident half-edge to this vertex."""
        if edge not in self.incident_edges:
            self.incident_edges.append(edge)

    def remove_incident_edge(self, edge: 'HalfEdge') -> None:
        """Remove an incident half-edge from this vertex."""
        if edge in self.incident_edges:
            self.incident_edges.remove(edge)

    def get_neighboring_vertices(self) -> List['Vertex']:
        """Get all vertices connected to this vertex by an edge."""
        neighbors = []
        for edge in self.incident_edges:
            if edge.origin == self and edge.destination:
                neighbors.append(edge.destination)
            elif edge.destination == self and edge.origin:
                neighbors.append(edge.origin)
        return neighbors

    def is_connected_to(self, other: 'Vertex') -> bool:
        """Check if this vertex is connected to another vertex."""
        return other in self.get_neighboring_vertices()

    def get_incident_faces(self) -> List['Face']:
        """Get all faces incident to this vertex."""
        from .face import Face
        faces = []
        for edge in self.incident_edges:
            if edge.face and edge.face not in faces:
                faces.append(edge.face)
        return faces

    @classmethod
    def at_infinity(cls, direction: Point) -> 'Vertex':
        """
        Create a vertex at infinity with given direction.
        Used for unbounded Voronoi edges.
        """
        # Use a very large coordinate in the given direction
        large_value = 1e6
        infinite_point = Point(
            direction.x * large_value,
            direction.y * large_value
        )
        vertex = cls(infinite_point)
        vertex.is_at_infinity = True
        return vertex