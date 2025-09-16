"""Voronoi diagram data structure with DCEL representation."""

from typing import List, Optional, Dict, Set, Tuple
import math
from .point import Point
from .vertex import Vertex
from .edge import Edge, HalfEdge
from .face import Face


class VoronoiDiagram:
    """
    Complete Voronoi diagram representation using Doubly Connected Edge List (DCEL).
    Provides efficient access to all geometric and topological information.
    """

    def __init__(self, bounding_box: Optional[Tuple[float, float, float, float]] = None):
        # DCEL components
        self.vertices: List[Vertex] = []
        self.edges: List[Edge] = []
        self.faces: List[Face] = []

        # Site points that generated this diagram
        self.sites: List[Point] = []

        # Spatial indexing for fast queries
        self._site_to_face: Dict[Point, Face] = {}
        self._vertex_dict: Dict[Point, Vertex] = {}

        # Bounding box for finite representation
        self.bounding_box = bounding_box or (-1000, -1000, 1000, 1000)
        self.min_x, self.min_y, self.max_x, self.max_y = self.bounding_box

        # Outer face (unbounded region)
        self.outer_face = Face(site=None)
        self.outer_face.is_unbounded = True
        self.faces.append(self.outer_face)

        # Statistics and metadata
        self.is_finalized = False

    def add_site(self, site: Point) -> Face:
        """
        Add a site to the diagram and create its corresponding face.

        Args:
            site: The site point to add

        Returns:
            The face corresponding to this site
        """
        if site in self.sites:
            return self._site_to_face[site]

        self.sites.append(site)

        # Create a new face for this site
        face = Face(site=site)
        self.faces.append(face)
        self._site_to_face[site] = face

        return face

    def add_vertex(self, point: Point) -> Vertex:
        """
        Add a vertex to the diagram.

        Args:
            point: The location of the vertex

        Returns:
            The vertex object (new or existing)
        """
        # Check if vertex already exists at this location
        if point in self._vertex_dict:
            return self._vertex_dict[point]

        vertex = Vertex(point)
        self.vertices.append(vertex)
        self._vertex_dict[point] = vertex

        return vertex

    def add_edge(self, site1: Point, site2: Point) -> Edge:
        """
        Add an edge (bisector) between two sites.

        Args:
            site1: First site
            site2: Second site

        Returns:
            The edge object
        """
        edge = Edge(site1, site2)
        self.edges.append(edge)
        return edge

    def get_face_for_site(self, site: Point) -> Optional[Face]:
        """Get the face corresponding to a site."""
        return self._site_to_face.get(site)

    def get_vertex_at_point(self, point: Point) -> Optional[Vertex]:
        """Get the vertex at a specific point."""
        return self._vertex_dict.get(point)

    def finalize_diagram(self) -> None:
        """
        Finalize the diagram by clipping infinite edges and computing final topology.
        """
        if self.is_finalized:
            return

        self._clip_infinite_edges()
        self._compute_face_boundaries()
        self._validate_topology()

        self.is_finalized = True

    def _clip_infinite_edges(self) -> None:
        """Clip infinite edges to the bounding box."""
        for edge in self.edges:
            if edge.half_edge1.is_infinite or edge.half_edge2.is_infinite:
                self._clip_edge_to_bounds(edge)

    def _clip_edge_to_bounds(self, edge: Edge) -> None:
        """Clip a single edge to the bounding box."""
        midpoint, direction = edge.get_perpendicular_bisector_line()

        # Find intersections with bounding box
        intersections = []

        # Check intersection with each boundary
        boundaries = [
            (self.min_x, self.min_y, self.min_x, self.max_y),  # Left
            (self.max_x, self.min_y, self.max_x, self.max_y),  # Right
            (self.min_x, self.min_y, self.max_x, self.min_y),  # Bottom
            (self.min_x, self.max_y, self.max_x, self.max_y),  # Top
        ]

        for x1, y1, x2, y2 in boundaries:
            intersection = self._line_segment_intersection(
                midpoint, direction, Point(x1, y1), Point(x2, y2)
            )
            if intersection:
                intersections.append(intersection)

        # Create vertices at intersection points
        if len(intersections) >= 2:
            v1 = self.add_vertex(intersections[0])
            v2 = self.add_vertex(intersections[1])

            # Set edge endpoints
            edge.set_vertex(v1, is_start=True)
            edge.set_vertex(v2, is_start=False)

    def _line_segment_intersection(self, point: Point, direction: Point,
                                   seg_start: Point, seg_end: Point) -> Optional[Point]:
        """Find intersection between a line and a line segment."""
        # Line equation: point + t * direction
        # Segment equation: seg_start + s * (seg_end - seg_start)

        seg_dir = seg_end - seg_start
        denominator = direction.cross(seg_dir)

        if abs(denominator) < 1e-10:
            return None  # Parallel lines

        t = (seg_start - point).cross(seg_dir) / denominator
        s = (seg_start - point).cross(direction) / denominator

        # Check if intersection is within the segment
        if 0 <= s <= 1:
            return point + direction * t

        return None

    def _compute_face_boundaries(self) -> None:
        """Compute the boundary cycles for each face."""
        for face in self.faces:
            if face.is_unbounded:
                continue

            # Find all half-edges that belong to this face
            face_edges = []
            for edge in self.edges:
                if edge.half_edge1.left_site == face.site:
                    face_edges.append(edge.half_edge1)
                if edge.half_edge2.left_site == face.site:
                    face_edges.append(edge.half_edge2)

            if face_edges:
                # Order the edges to form a cycle
                boundary_cycle = self._order_face_edges(face_edges)
                if boundary_cycle:
                    face.set_boundary_chain(boundary_cycle[0])

    def _order_face_edges(self, edges: List[HalfEdge]) -> List[HalfEdge]:
        """Order half-edges to form a proper boundary cycle."""
        if not edges:
            return []

        # Simple implementation: could be improved with better connectivity analysis
        ordered = [edges[0]]
        remaining = edges[1:]

        while remaining:
            found_next = False
            current_end = ordered[-1].destination

            if not current_end:
                break

            for i, edge in enumerate(remaining):
                if edge.origin == current_end:
                    ordered.append(edge)
                    remaining.pop(i)
                    found_next = True
                    break

            if not found_next:
                break

        # Set up next/prev pointers
        for i in range(len(ordered)):
            current = ordered[i]
            next_edge = ordered[(i + 1) % len(ordered)]
            prev_edge = ordered[(i - 1) % len(ordered)]

            current.next = next_edge
            current.prev = prev_edge

        return ordered

    def _validate_topology(self) -> bool:
        """Validate the topological consistency of the DCEL."""
        # Check vertex-edge consistency
        for vertex in self.vertices:
            for edge in vertex.incident_edges:
                if edge.origin != vertex and edge.destination != vertex:
                    return False

        # Check edge-face consistency
        for edge in self.edges:
            if edge.half_edge1.twin != edge.half_edge2:
                return False
            if edge.half_edge2.twin != edge.half_edge1:
                return False

        return True

    def get_nearest_site(self, query_point: Point) -> Optional[Tuple[Point, float]]:
        """
        Find the nearest site to a query point.

        Args:
            query_point: The point to query

        Returns:
            Tuple of (nearest_site, distance) or None if no sites
        """
        if not self.sites:
            return None

        nearest_site = self.sites[0]
        min_distance = query_point.distance_to(nearest_site)

        for site in self.sites[1:]:
            distance = query_point.distance_to(site)
            if distance < min_distance:
                min_distance = distance
                nearest_site = site

        return nearest_site, min_distance

    def get_voronoi_cell(self, site: Point) -> Optional[Face]:
        """Get the Voronoi cell (face) for a given site."""
        return self._site_to_face.get(site)

    def get_cell_neighbors(self, site: Point) -> List[Point]:
        """Get all sites that are neighbors to the given site."""
        face = self.get_voronoi_cell(site)
        if not face:
            return []

        neighbors = []
        for edge in face.get_boundary_edges():
            # The neighboring site is on the other side of the edge
            if edge.left_site == site:
                if edge.right_site:
                    neighbors.append(edge.right_site)
            elif edge.right_site == site:
                if edge.left_site:
                    neighbors.append(edge.left_site)

        return list(set(neighbors))  # Remove duplicates

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the diagram."""
        return {
            'num_sites': len(self.sites),
            'num_vertices': len(self.vertices),
            'num_edges': len(self.edges),
            'num_faces': len(self.faces),
            'num_bounded_faces': len([f for f in self.faces if not f.is_unbounded]),
        }

    def bounds(self) -> Tuple[float, float, float, float]:
        """Get the actual bounds of all diagram elements."""
        if not self.vertices:
            return self.bounding_box

        min_x = min(v.x for v in self.vertices)
        max_x = max(v.x for v in self.vertices)
        min_y = min(v.y for v in self.vertices)
        max_y = max(v.y for v in self.vertices)

        return min_x, min_y, max_x, max_y

    def __str__(self) -> str:
        stats = self.get_statistics()
        return (f"VoronoiDiagram({stats['num_sites']} sites, "
                f"{stats['num_vertices']} vertices, {stats['num_edges']} edges)")

    def __repr__(self) -> str:
        return f"VoronoiDiagram(sites={len(self.sites)}, finalized={self.is_finalized})"