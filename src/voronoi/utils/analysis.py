"""Advanced analysis tools for Voronoi diagrams."""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from ..data_structures.voronoi_diagram import VoronoiDiagram
from ..data_structures.point import Point
from ..data_structures.face import Face


@dataclass
class DiagramStatistics:
    """Statistical analysis of a Voronoi diagram."""
    num_sites: int
    num_vertices: int
    num_edges: int
    num_faces: int

    # Cell statistics
    avg_cell_area: float
    cell_area_variance: float
    min_cell_area: float
    max_cell_area: float

    # Vertex degree statistics
    avg_vertex_degree: float
    vertex_degree_variance: float
    min_vertex_degree: int
    max_vertex_degree: int

    # Edge length statistics
    avg_edge_length: float
    edge_length_variance: float
    min_edge_length: float
    max_edge_length: float

    # Geometric properties
    diagram_area: float
    diagram_perimeter: float
    convexity_deficiency: float

    # Regularity measures
    regularity_index: float
    uniformity_index: float


class VoronoiAnalyzer:
    """
    Advanced analyzer for Voronoi diagrams providing geometric and topological insights.
    """

    def __init__(self):
        self.epsilon = 1e-10

    def analyze_diagram(self, diagram: VoronoiDiagram) -> DiagramStatistics:
        """
        Perform comprehensive analysis of a Voronoi diagram.

        Args:
            diagram: The Voronoi diagram to analyze

        Returns:
            Complete statistical analysis
        """
        # Basic counts
        num_sites = len(diagram.sites)
        num_vertices = len(diagram.vertices)
        num_edges = len(diagram.edges)
        num_faces = len([f for f in diagram.faces if not f.is_unbounded])

        # Cell area analysis
        cell_areas = self._calculate_cell_areas(diagram)
        avg_cell_area = np.mean(cell_areas) if cell_areas else 0
        cell_area_variance = np.var(cell_areas) if len(cell_areas) > 1 else 0
        min_cell_area = min(cell_areas) if cell_areas else 0
        max_cell_area = max(cell_areas) if cell_areas else 0

        # Vertex degree analysis
        vertex_degrees = [vertex.degree for vertex in diagram.vertices]
        avg_vertex_degree = np.mean(vertex_degrees) if vertex_degrees else 0
        vertex_degree_variance = np.var(vertex_degrees) if len(vertex_degrees) > 1 else 0
        min_vertex_degree = min(vertex_degrees) if vertex_degrees else 0
        max_vertex_degree = max(vertex_degrees) if vertex_degrees else 0

        # Edge length analysis
        edge_lengths = self._calculate_edge_lengths(diagram)
        avg_edge_length = np.mean(edge_lengths) if edge_lengths else 0
        edge_length_variance = np.var(edge_lengths) if len(edge_lengths) > 1 else 0
        min_edge_length = min(edge_lengths) if edge_lengths else 0
        max_edge_length = max(edge_lengths) if edge_lengths else 0

        # Geometric properties
        diagram_area = self._calculate_diagram_area(diagram)
        diagram_perimeter = self._calculate_diagram_perimeter(diagram)
        convexity_deficiency = self._calculate_convexity_deficiency(diagram)

        # Regularity measures
        regularity_index = self._calculate_regularity_index(diagram)
        uniformity_index = self._calculate_uniformity_index(diagram)

        return DiagramStatistics(
            num_sites=num_sites,
            num_vertices=num_vertices,
            num_edges=num_edges,
            num_faces=num_faces,
            avg_cell_area=avg_cell_area,
            cell_area_variance=cell_area_variance,
            min_cell_area=min_cell_area,
            max_cell_area=max_cell_area,
            avg_vertex_degree=avg_vertex_degree,
            vertex_degree_variance=vertex_degree_variance,
            min_vertex_degree=min_vertex_degree,
            max_vertex_degree=max_vertex_degree,
            avg_edge_length=avg_edge_length,
            edge_length_variance=edge_length_variance,
            min_edge_length=min_edge_length,
            max_edge_length=max_edge_length,
            diagram_area=diagram_area,
            diagram_perimeter=diagram_perimeter,
            convexity_deficiency=convexity_deficiency,
            regularity_index=regularity_index,
            uniformity_index=uniformity_index
        )

    def _calculate_cell_areas(self, diagram: VoronoiDiagram) -> List[float]:
        """Calculate areas of all bounded Voronoi cells."""
        areas = []
        for face in diagram.faces:
            if not face.is_unbounded and face.site:
                area = face.area()
                if area > self.epsilon:
                    areas.append(area)
        return areas

    def _calculate_edge_lengths(self, diagram: VoronoiDiagram) -> List[float]:
        """Calculate lengths of all diagram edges."""
        lengths = []
        for edge in diagram.edges:
            length = edge.length
            if length and length > self.epsilon:
                lengths.append(length)
        return lengths

    def _calculate_diagram_area(self, diagram: VoronoiDiagram) -> float:
        """Calculate total area of the diagram bounds."""
        min_x, min_y, max_x, max_y = diagram.bounding_box
        return (max_x - min_x) * (max_y - min_y)

    def _calculate_diagram_perimeter(self, diagram: VoronoiDiagram) -> float:
        """Calculate perimeter of the diagram bounds."""
        min_x, min_y, max_x, max_y = diagram.bounding_box
        return 2 * ((max_x - min_x) + (max_y - min_y))

    def _calculate_convexity_deficiency(self, diagram: VoronoiDiagram) -> float:
        """
        Calculate convexity deficiency of the site set.
        This measures how much the convex hull differs from a circle.
        """
        if len(diagram.sites) < 3:
            return 0.0

        # Find convex hull
        hull_points = self._convex_hull(diagram.sites)
        if len(hull_points) < 3:
            return 0.0

        # Calculate hull area and perimeter
        hull_area = self._polygon_area(hull_points)
        hull_perimeter = self._polygon_perimeter(hull_points)

        if hull_perimeter < self.epsilon:
            return 0.0

        # Compare to circle with same perimeter
        circle_radius = hull_perimeter / (2 * math.pi)
        circle_area = math.pi * circle_radius * circle_radius

        # Convexity deficiency is the ratio of areas
        return 1.0 - (hull_area / circle_area) if circle_area > 0 else 0.0

    def _calculate_regularity_index(self, diagram: VoronoiDiagram) -> float:
        """
        Calculate regularity index based on cell shape uniformity.
        Higher values indicate more regular (hexagonal-like) cells.
        """
        if not diagram.faces:
            return 0.0

        regularity_scores = []

        for face in diagram.faces:
            if face.is_unbounded or not face.site:
                continue

            boundary_points = face.get_boundary_points()
            if len(boundary_points) < 3:
                continue

            # Calculate shape regularity
            area = face.area()
            perimeter = self._polygon_perimeter(boundary_points)

            if perimeter < self.epsilon:
                continue

            # Isoperimetric ratio (circle has ratio of 1, regular polygons have ratios < 1)
            shape_ratio = 4 * math.pi * area / (perimeter * perimeter)

            # Number of sides regularity (hexagon = 6 is optimal for Voronoi)
            num_sides = len(boundary_points)
            side_regularity = 1.0 - abs(num_sides - 6) / 6.0
            side_regularity = max(0, side_regularity)

            # Combined regularity score
            regularity_score = shape_ratio * side_regularity
            regularity_scores.append(regularity_score)

        return np.mean(regularity_scores) if regularity_scores else 0.0

    def _calculate_uniformity_index(self, diagram: VoronoiDiagram) -> float:
        """
        Calculate uniformity index based on area and neighbor distance variance.
        Higher values indicate more uniform distribution.
        """
        if len(diagram.sites) < 2:
            return 1.0

        # Calculate area uniformity
        cell_areas = self._calculate_cell_areas(diagram)
        if not cell_areas:
            area_uniformity = 0.0
        else:
            area_cv = np.std(cell_areas) / np.mean(cell_areas) if np.mean(cell_areas) > 0 else 0
            area_uniformity = 1.0 / (1.0 + area_cv)

        # Calculate neighbor distance uniformity
        neighbor_distances = []
        for site in diagram.sites:
            face = diagram.get_voronoi_cell(site)
            if face:
                neighbors = diagram.get_cell_neighbors(site)
                if neighbors:
                    distances = [site.distance_to(neighbor) for neighbor in neighbors]
                    neighbor_distances.extend(distances)

        if not neighbor_distances:
            distance_uniformity = 0.0
        else:
            distance_cv = np.std(neighbor_distances) / np.mean(neighbor_distances) if np.mean(neighbor_distances) > 0 else 0
            distance_uniformity = 1.0 / (1.0 + distance_cv)

        # Combined uniformity
        return (area_uniformity + distance_uniformity) / 2.0

    def _convex_hull(self, points: List[Point]) -> List[Point]:
        """Calculate convex hull using Graham scan algorithm."""
        if len(points) < 3:
            return points

        # Find bottom-most point (and leftmost if tie)
        start = min(points, key=lambda p: (p.y, p.x))

        # Sort points by polar angle with respect to start point
        def polar_angle(point: Point) -> float:
            dx = point.x - start.x
            dy = point.y - start.y
            return math.atan2(dy, dx)

        sorted_points = sorted([p for p in points if p != start], key=polar_angle)
        hull_points = [start] + sorted_points

        # Graham scan
        hull = []
        for point in hull_points:
            while len(hull) > 1:
                # Check if we make a right turn (clockwise)
                cross = ((hull[-1].x - hull[-2].x) * (point.y - hull[-1].y) -
                        (hull[-1].y - hull[-2].y) * (point.x - hull[-1].x))
                if cross <= 0:  # Right turn or collinear, remove last point
                    hull.pop()
                else:
                    break
            hull.append(point)

        return hull

    def _polygon_area(self, points: List[Point]) -> float:
        """Calculate area of a polygon using shoelace formula."""
        if len(points) < 3:
            return 0.0

        area = 0.0
        n = len(points)

        for i in range(n):
            j = (i + 1) % n
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y

        return abs(area) / 2.0

    def _polygon_perimeter(self, points: List[Point]) -> float:
        """Calculate perimeter of a polygon."""
        if len(points) < 2:
            return 0.0

        perimeter = 0.0
        n = len(points)

        for i in range(n):
            j = (i + 1) % n
            perimeter += points[i].distance_to(points[j])

        return perimeter

    def detect_patterns(self, diagram: VoronoiDiagram) -> Dict[str, Any]:
        """
        Detect patterns and anomalies in the Voronoi diagram.

        Returns:
            Dictionary containing detected patterns and their characteristics
        """
        patterns = {
            'clustering': self._detect_clustering(diagram),
            'regular_structure': self._detect_regular_structure(diagram),
            'boundary_effects': self._detect_boundary_effects(diagram),
            'degeneracies': self._detect_degeneracies(diagram)
        }

        return patterns

    def _detect_clustering(self, diagram: VoronoiDiagram) -> Dict[str, Any]:
        """Detect clustering patterns in site distribution."""
        if len(diagram.sites) < 3:
            return {'detected': False, 'reason': 'insufficient_sites'}

        # Calculate nearest neighbor distances
        nn_distances = []
        for site in diagram.sites:
            min_dist = float('inf')
            for other_site in diagram.sites:
                if site != other_site:
                    dist = site.distance_to(other_site)
                    if dist < min_dist:
                        min_dist = dist
            nn_distances.append(min_dist)

        # Analyze distribution of nearest neighbor distances
        mean_nn = np.mean(nn_distances)
        std_nn = np.std(nn_distances)

        # Clark-Evans aggregation index
        bounds = diagram.bounds()
        if bounds:
            min_x, min_y, max_x, max_y = bounds
            area = (max_x - min_x) * (max_y - min_y)
            density = len(diagram.sites) / area
            expected_nn = 0.5 / math.sqrt(density)

            clark_evans_index = mean_nn / expected_nn if expected_nn > 0 else 1.0

            # Interpretation
            if clark_evans_index < 0.7:
                clustering_type = "highly_clustered"
            elif clark_evans_index < 1.0:
                clustering_type = "moderately_clustered"
            elif clark_evans_index < 1.3:
                clustering_type = "random"
            else:
                clustering_type = "regular"

            return {
                'detected': True,
                'clark_evans_index': clark_evans_index,
                'clustering_type': clustering_type,
                'mean_nearest_neighbor': mean_nn,
                'std_nearest_neighbor': std_nn
            }

        return {'detected': False, 'reason': 'no_bounds'}

    def _detect_regular_structure(self, diagram: VoronoiDiagram) -> Dict[str, Any]:
        """Detect regular grid-like structures."""
        if len(diagram.sites) < 9:  # Need at least 3x3 grid
            return {'detected': False, 'reason': 'insufficient_sites'}

        # Analyze vertex degree distribution
        vertex_degrees = [v.degree for v in diagram.vertices]
        degree_hist = {}
        for degree in vertex_degrees:
            degree_hist[degree] = degree_hist.get(degree, 0) + 1

        # Regular Voronoi diagrams have mostly degree-3 vertices
        degree_3_ratio = degree_hist.get(3, 0) / len(vertex_degrees) if vertex_degrees else 0

        # Analyze cell side count distribution
        cell_sides = []
        for face in diagram.faces:
            if not face.is_unbounded and face.site:
                boundary = face.get_boundary_points()
                cell_sides.append(len(boundary))

        if cell_sides:
            hexagon_ratio = cell_sides.count(6) / len(cell_sides)
            avg_sides = np.mean(cell_sides)
            std_sides = np.std(cell_sides)
        else:
            hexagon_ratio = 0
            avg_sides = 0
            std_sides = 0

        # Regular structure detection
        regularity_score = (degree_3_ratio + hexagon_ratio) / 2

        structure_type = "irregular"
        if regularity_score > 0.8:
            structure_type = "highly_regular"
        elif regularity_score > 0.6:
            structure_type = "moderately_regular"
        elif regularity_score > 0.4:
            structure_type = "somewhat_regular"

        return {
            'detected': regularity_score > 0.4,
            'structure_type': structure_type,
            'regularity_score': regularity_score,
            'degree_3_ratio': degree_3_ratio,
            'hexagon_ratio': hexagon_ratio,
            'avg_cell_sides': avg_sides,
            'std_cell_sides': std_sides
        }

    def _detect_boundary_effects(self, diagram: VoronoiDiagram) -> Dict[str, Any]:
        """Detect boundary effects in the diagram."""
        if not diagram.sites:
            return {'detected': False, 'reason': 'no_sites'}

        bounds = diagram.bounds()
        if not bounds:
            return {'detected': False, 'reason': 'no_bounds'}

        min_x, min_y, max_x, max_y = bounds
        width = max_x - min_x
        height = max_y - min_y
        boundary_margin = min(width, height) * 0.1

        # Count sites near boundaries
        boundary_sites = 0
        total_sites = len(diagram.sites)

        for site in diagram.sites:
            if (site.x - min_x < boundary_margin or
                max_x - site.x < boundary_margin or
                site.y - min_y < boundary_margin or
                max_y - site.y < boundary_margin):
                boundary_sites += 1

        boundary_ratio = boundary_sites / total_sites if total_sites > 0 else 0

        # Count unbounded faces
        unbounded_faces = sum(1 for face in diagram.faces if face.is_unbounded)

        return {
            'detected': boundary_ratio > 0.2 or unbounded_faces > 0,
            'boundary_site_ratio': boundary_ratio,
            'unbounded_faces': unbounded_faces,
            'boundary_margin': boundary_margin,
            'effect_strength': 'high' if boundary_ratio > 0.5 else 'moderate' if boundary_ratio > 0.3 else 'low'
        }

    def _detect_degeneracies(self, diagram: VoronoiDiagram) -> Dict[str, Any]:
        """Detect degenerate cases and numerical issues."""
        degeneracies = {
            'collinear_sites': 0,
            'duplicate_sites': 0,
            'high_degree_vertices': 0,
            'very_small_cells': 0,
            'very_large_cells': 0
        }

        # Check for duplicate sites
        site_positions = set()
        for site in diagram.sites:
            pos = (round(site.x, 10), round(site.y, 10))
            if pos in site_positions:
                degeneracies['duplicate_sites'] += 1
            site_positions.add(pos)

        # Check for high-degree vertices (>3 is unusual)
        for vertex in diagram.vertices:
            if vertex.degree > 3:
                degeneracies['high_degree_vertices'] += 1

        # Check cell areas
        cell_areas = self._calculate_cell_areas(diagram)
        if cell_areas:
            mean_area = np.mean(cell_areas)
            for area in cell_areas:
                if area < mean_area * 0.01:
                    degeneracies['very_small_cells'] += 1
                elif area > mean_area * 100:
                    degeneracies['very_large_cells'] += 1

        total_degeneracies = sum(degeneracies.values())

        return {
            'detected': total_degeneracies > 0,
            'total_count': total_degeneracies,
            'details': degeneracies,
            'severity': 'high' if total_degeneracies > len(diagram.sites) * 0.1 else 'low'
        }