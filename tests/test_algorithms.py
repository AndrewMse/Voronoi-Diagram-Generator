"""Tests for algorithm implementations."""

import unittest
import math
from src.voronoi.data_structures.point import Point
from src.voronoi.algorithms.fortune import FortunesAlgorithm
from src.voronoi.algorithms.geometry_utils import GeometryUtils


class TestGeometryUtils(unittest.TestCase):
    """Test geometric utility functions."""

    def test_circumcenter_calculation(self):
        """Test circumcenter calculation for three points."""
        # Equilateral triangle with known circumcenter
        p1 = Point(0, 0)
        p2 = Point(2, 0)
        p3 = Point(1, math.sqrt(3))

        center = GeometryUtils.circumcenter(p1, p2, p3)
        self.assertIsNotNone(center)
        self.assertAlmostEqual(center.x, 1, places=10)
        self.assertAlmostEqual(center.y, math.sqrt(3) / 3, places=10)

    def test_circumcenter_collinear_points(self):
        """Test circumcenter with collinear points."""
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(2, 0)

        center = GeometryUtils.circumcenter(p1, p2, p3)
        self.assertIsNone(center)  # Collinear points have no circumcenter

    def test_circumradius_calculation(self):
        """Test circumradius calculation."""
        # Right triangle with hypotenuse as diameter
        p1 = Point(0, 0)
        p2 = Point(4, 0)
        p3 = Point(0, 3)

        radius = GeometryUtils.circumradius(p1, p2, p3)
        self.assertIsNotNone(radius)
        self.assertAlmostEqual(radius, 2.5, places=10)  # Half of hypotenuse (5)

    def test_collinearity_detection(self):
        """Test collinearity detection."""
        # Collinear points
        p1 = Point(0, 0)
        p2 = Point(1, 1)
        p3 = Point(2, 2)

        self.assertTrue(GeometryUtils.are_collinear(p1, p2, p3))

        # Non-collinear points
        p4 = Point(0, 1)
        self.assertFalse(GeometryUtils.are_collinear(p1, p2, p4))

    def test_point_in_circle(self):
        """Test point-in-circle test."""
        center = Point(0, 0)
        radius = 5

        # Point inside
        self.assertTrue(GeometryUtils.point_in_circle(Point(3, 4), center, radius))

        # Point outside
        self.assertFalse(GeometryUtils.point_in_circle(Point(6, 0), center, radius))

        # Point on circle (approximately)
        self.assertFalse(GeometryUtils.point_in_circle(Point(5, 0), center, radius))

    def test_perpendicular_bisector(self):
        """Test perpendicular bisector calculation."""
        p1 = Point(0, 0)
        p2 = Point(4, 0)

        midpoint, direction = GeometryUtils.perpendicular_bisector(p1, p2)

        self.assertEqual(midpoint.x, 2)
        self.assertEqual(midpoint.y, 0)
        self.assertAlmostEqual(abs(direction.x), 0, places=10)
        self.assertAlmostEqual(abs(direction.y), 1, places=10)

    def test_line_intersection(self):
        """Test line intersection calculation."""
        # Two perpendicular lines
        p1 = Point(0, 0)
        d1 = Point(1, 0)  # Horizontal line
        p2 = Point(1, -1)
        d2 = Point(0, 1)  # Vertical line

        intersection = GeometryUtils.line_intersection(p1, d1, p2, d2)
        self.assertIsNotNone(intersection)
        self.assertAlmostEqual(intersection.x, 1, places=10)
        self.assertAlmostEqual(intersection.y, 0, places=10)

    def test_parallel_line_intersection(self):
        """Test intersection of parallel lines."""
        p1 = Point(0, 0)
        d1 = Point(1, 0)  # Horizontal line
        p2 = Point(0, 1)
        d2 = Point(1, 0)  # Parallel horizontal line

        intersection = GeometryUtils.line_intersection(p1, d1, p2, d2)
        self.assertIsNone(intersection)  # Parallel lines don't intersect

    def test_orientation_test(self):
        """Test 2D orientation test."""
        # Counterclockwise orientation
        pa = Point(0, 0)
        pb = Point(1, 0)
        pc = Point(0, 1)

        orientation = GeometryUtils.orient2d(pa, pb, pc)
        self.assertGreater(orientation, 0)  # Positive for CCW

        # Clockwise orientation
        pc_cw = Point(0, -1)
        orientation_cw = GeometryUtils.orient2d(pa, pb, pc_cw)
        self.assertLess(orientation_cw, 0)  # Negative for CW

        # Collinear
        pc_col = Point(2, 0)
        orientation_col = GeometryUtils.orient2d(pa, pb, pc_col)
        self.assertAlmostEqual(orientation_col, 0, places=10)


class TestFortunesAlgorithm(unittest.TestCase):
    """Test Fortune's sweep line algorithm."""

    def setUp(self):
        """Set up test fixtures."""
        self.algorithm = FortunesAlgorithm(bounding_box=(-10, -10, 10, 10))

    def test_simple_two_sites(self):
        """Test diagram generation with two sites."""
        sites = [Point(-1, 0), Point(1, 0)]
        diagram = self.algorithm.generate_voronoi_diagram(sites)

        self.assertEqual(len(diagram.sites), 2)
        # Two sites should create one edge (bisector)
        self.assertGreaterEqual(len(diagram.edges), 1)

    def test_three_sites_triangle(self):
        """Test diagram generation with three sites forming a triangle."""
        sites = [
            Point(0, 2),
            Point(-1, -1),
            Point(1, -1)
        ]
        diagram = self.algorithm.generate_voronoi_diagram(sites)

        self.assertEqual(len(diagram.sites), 3)
        # Three sites in general position should create one vertex and three edges
        self.assertGreaterEqual(len(diagram.vertices), 1)
        self.assertGreaterEqual(len(diagram.edges), 3)

    def test_collinear_sites(self):
        """Test diagram generation with collinear sites."""
        sites = [
            Point(0, 0),
            Point(1, 0),
            Point(2, 0)
        ]
        diagram = self.algorithm.generate_voronoi_diagram(sites)

        self.assertEqual(len(diagram.sites), 3)
        # Collinear sites should create parallel bisectors
        self.assertGreaterEqual(len(diagram.edges), 2)

    def test_square_grid_sites(self):
        """Test diagram generation with sites in a square grid."""
        sites = [
            Point(0, 0), Point(1, 0),
            Point(0, 1), Point(1, 1)
        ]
        diagram = self.algorithm.generate_voronoi_diagram(sites)

        self.assertEqual(len(diagram.sites), 4)
        # Four sites in a square should create specific topology
        self.assertGreater(len(diagram.vertices), 0)
        self.assertGreater(len(diagram.edges), 0)

    def test_random_sites_properties(self):
        """Test that generated diagrams satisfy basic properties."""
        import random
        random.seed(42)  # For reproducible tests

        sites = []
        for _ in range(10):
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            sites.append(Point(x, y))

        diagram = self.algorithm.generate_voronoi_diagram(sites)

        # Basic sanity checks
        self.assertEqual(len(diagram.sites), 10)
        self.assertGreater(len(diagram.edges), 0)

        # Euler's formula check (for planar graphs)
        # V - E + F = 2 (including outer face)
        V = len(diagram.vertices)
        E = len(diagram.edges)
        F = len(diagram.faces)  # Includes unbounded faces

        if V > 0 and E > 0:
            # Allow some tolerance for boundary effects
            euler_characteristic = V - E + F
            self.assertGreaterEqual(euler_characteristic, 1)
            self.assertLessEqual(euler_characteristic, 3)

    def test_duplicate_sites(self):
        """Test handling of duplicate sites."""
        sites = [
            Point(0, 0),
            Point(0, 0),  # Duplicate
            Point(1, 0)
        ]

        # Should handle gracefully without crashing
        try:
            diagram = self.algorithm.generate_voronoi_diagram(sites)
            # If no exception, check that duplicates were handled
            self.assertLessEqual(len(diagram.sites), 3)
        except ValueError:
            # Or it might raise ValueError for duplicates, which is also acceptable
            pass

    def test_empty_sites_list(self):
        """Test handling of empty sites list."""
        with self.assertRaises(ValueError):
            self.algorithm.generate_voronoi_diagram([])

    def test_single_site(self):
        """Test handling of single site."""
        sites = [Point(0, 0)]

        with self.assertRaises(ValueError):
            self.algorithm.generate_voronoi_diagram(sites)

    def test_algorithm_state_tracking(self):
        """Test algorithm state tracking during execution."""
        self.algorithm.enable_debug_mode()

        sites = [Point(0, 0), Point(1, 0), Point(0.5, 1)]
        diagram = self.algorithm.generate_voronoi_diagram(sites)

        # Check that debug information was recorded
        state = self.algorithm.get_algorithm_state()
        history = self.algorithm.get_event_history()

        self.assertGreater(state['step_count'], 0)
        self.assertGreater(len(history), 0)

    def test_vertex_degrees(self):
        """Test that Voronoi vertices have valid degrees."""
        sites = [
            Point(0, 0), Point(2, 0), Point(1, 2), Point(3, 1)
        ]
        diagram = self.algorithm.generate_voronoi_diagram(sites)

        # In a valid Voronoi diagram, most vertices should have degree 3
        for vertex in diagram.vertices:
            self.assertGreaterEqual(vertex.degree, 2)
            self.assertLessEqual(vertex.degree, 6)  # Reasonable upper bound

    def test_edge_completeness(self):
        """Test that edges are properly completed."""
        sites = [Point(-1, 0), Point(1, 0), Point(0, 1)]
        diagram = self.algorithm.generate_voronoi_diagram(sites)

        finite_edges = 0
        for edge in diagram.edges:
            if edge.is_complete:
                finite_edges += 1
                # Complete edges should have both endpoints
                self.assertIsNotNone(edge.half_edge1.origin)
                self.assertIsNotNone(edge.half_edge1.destination)

        # Should have at least some finite edges
        self.assertGreater(finite_edges, 0)


if __name__ == '__main__':
    unittest.main()