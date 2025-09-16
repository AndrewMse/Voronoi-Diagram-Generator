"""Tests for core data structures."""

import unittest
import math
from src.voronoi.data_structures.point import Point
from src.voronoi.data_structures.vertex import Vertex
from src.voronoi.data_structures.edge import Edge, HalfEdge
from src.voronoi.data_structures.face import Face
from src.voronoi.data_structures.beach_line import BeachLine, Arc
from src.voronoi.data_structures.event_queue import EventQueue, SiteEvent, CircleEvent


class TestPoint(unittest.TestCase):
    """Test Point class functionality."""

    def test_point_creation(self):
        """Test point creation and basic properties."""
        p = Point(3.14, 2.71)
        self.assertEqual(p.x, 3.14)
        self.assertEqual(p.y, 2.71)

    def test_point_equality(self):
        """Test point equality comparison."""
        p1 = Point(1, 2)
        p2 = Point(1, 2)
        p3 = Point(1, 3)

        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)

    def test_point_arithmetic(self):
        """Test point arithmetic operations."""
        p1 = Point(1, 2)
        p2 = Point(3, 4)

        # Addition
        p3 = p1 + p2
        self.assertEqual(p3.x, 4)
        self.assertEqual(p3.y, 6)

        # Subtraction
        p4 = p2 - p1
        self.assertEqual(p4.x, 2)
        self.assertEqual(p4.y, 2)

        # Scalar multiplication
        p5 = p1 * 2
        self.assertEqual(p5.x, 2)
        self.assertEqual(p5.y, 4)

    def test_distance_calculation(self):
        """Test distance calculations."""
        p1 = Point(0, 0)
        p2 = Point(3, 4)

        # Euclidean distance
        distance = p1.distance_to(p2)
        self.assertAlmostEqual(distance, 5.0, places=10)

        # Squared distance
        distance_sq = p1.distance_squared_to(p2)
        self.assertAlmostEqual(distance_sq, 25.0, places=10)

    def test_point_magnitude(self):
        """Test point magnitude calculation."""
        p = Point(3, 4)
        self.assertAlmostEqual(p.magnitude, 5.0, places=10)

    def test_point_normalization(self):
        """Test point normalization."""
        p = Point(3, 4)
        normalized = p.normalize()
        self.assertAlmostEqual(normalized.magnitude, 1.0, places=10)
        self.assertAlmostEqual(normalized.x, 0.6, places=10)
        self.assertAlmostEqual(normalized.y, 0.8, places=10)


class TestVertex(unittest.TestCase):
    """Test Vertex class functionality."""

    def test_vertex_creation(self):
        """Test vertex creation."""
        p = Point(1, 2)
        v = Vertex(p)

        self.assertEqual(v.point, p)
        self.assertEqual(v.x, 1)
        self.assertEqual(v.y, 2)
        self.assertEqual(v.degree, 0)

    def test_vertex_equality(self):
        """Test vertex equality."""
        p1 = Point(1, 2)
        p2 = Point(1, 2)
        v1 = Vertex(p1)
        v2 = Vertex(p2)

        self.assertEqual(v1, v2)


class TestEdge(unittest.TestCase):
    """Test Edge and HalfEdge classes."""

    def test_edge_creation(self):
        """Test edge creation."""
        site1 = Point(0, 0)
        site2 = Point(2, 0)
        edge = Edge(site1, site2)

        self.assertEqual(edge.site1, site1)
        self.assertEqual(edge.site2, site2)
        self.assertIsNotNone(edge.half_edge1)
        self.assertIsNotNone(edge.half_edge2)
        self.assertEqual(edge.half_edge1.twin, edge.half_edge2)
        self.assertEqual(edge.half_edge2.twin, edge.half_edge1)

    def test_perpendicular_bisector(self):
        """Test perpendicular bisector calculation."""
        site1 = Point(0, 0)
        site2 = Point(2, 0)
        edge = Edge(site1, site2)

        midpoint, direction = edge.get_perpendicular_bisector_line()
        self.assertEqual(midpoint.x, 1)
        self.assertEqual(midpoint.y, 0)
        # Direction should be perpendicular to (2, 0), so (0, 1) or (0, -1)
        self.assertAlmostEqual(abs(direction.x), 0, places=10)
        self.assertAlmostEqual(abs(direction.y), 1, places=10)


class TestBeachLine(unittest.TestCase):
    """Test BeachLine and Arc classes."""

    def test_beach_line_creation(self):
        """Test beach line creation."""
        beach_line = BeachLine()
        self.assertTrue(beach_line.is_empty())
        self.assertIsNone(beach_line.root)

    def test_first_arc_insertion(self):
        """Test inserting the first arc."""
        beach_line = BeachLine()
        site = Point(5, 10)

        arc = beach_line.insert_first_arc(site)

        self.assertFalse(beach_line.is_empty())
        self.assertEqual(arc.site, site)
        self.assertEqual(beach_line.root, arc)
        self.assertEqual(beach_line.leftmost, arc)
        self.assertEqual(beach_line.rightmost, arc)

    def test_arc_parabola_calculation(self):
        """Test arc parabola y-coordinate calculation."""
        site = Point(0, 5)
        arc = Arc(site)
        sweep_y = 0

        # At x = 0, parabola should pass through midpoint
        y = arc.get_y_at_x(0, sweep_y)
        self.assertAlmostEqual(y, 2.5, places=10)

        # At x = 5, parabola should be higher
        y = arc.get_y_at_x(5, sweep_y)
        self.assertAlmostEqual(y, 5.0, places=10)

    def test_arc_splitting(self):
        """Test arc splitting."""
        beach_line = BeachLine()
        site1 = Point(0, 10)
        site2 = Point(5, 5)

        # Insert first arc
        arc1 = beach_line.insert_first_arc(site1)

        # Split the arc
        left_arc, middle_arc, right_arc = beach_line.split_arc(arc1, site2)

        self.assertEqual(left_arc.site, site1)
        self.assertEqual(middle_arc.site, site2)
        self.assertEqual(right_arc.site, site1)

        # Check linkage
        self.assertEqual(left_arc.right_arc, middle_arc)
        self.assertEqual(middle_arc.left_arc, left_arc)
        self.assertEqual(middle_arc.right_arc, right_arc)
        self.assertEqual(right_arc.left_arc, middle_arc)


class TestEventQueue(unittest.TestCase):
    """Test EventQueue and Event classes."""

    def test_event_queue_creation(self):
        """Test event queue creation."""
        queue = EventQueue()
        self.assertTrue(queue.is_empty())
        self.assertEqual(len(queue), 0)

    def test_site_event_ordering(self):
        """Test site event ordering in queue."""
        queue = EventQueue()

        # Add events with different y-coordinates
        site1 = Point(0, 5)  # Lower y-coordinate
        site2 = Point(0, 10) # Higher y-coordinate

        queue.add_site_event(site1)
        queue.add_site_event(site2)

        # Higher y-coordinate should come first
        event1 = queue.pop_event()
        event2 = queue.pop_event()

        self.assertIsInstance(event1, SiteEvent)
        self.assertIsInstance(event2, SiteEvent)
        self.assertEqual(event1.site.y, 10)
        self.assertEqual(event2.site.y, 5)

    def test_event_invalidation(self):
        """Test event invalidation."""
        queue = EventQueue()
        site = Point(0, 0)
        arc = Arc(site)

        # Add a circle event
        center = Point(0, 0)
        radius = 5
        event = queue.add_circle_event(center, radius, arc)

        # Invalidate the event
        event.invalidate()

        # Should return None when popped (invalid events are skipped)
        popped_event = queue.pop_event()
        self.assertIsNone(popped_event)


class TestFace(unittest.TestCase):
    """Test Face class functionality."""

    def test_face_creation(self):
        """Test face creation."""
        site = Point(0, 0)
        face = Face(site)

        self.assertEqual(face.site, site)
        self.assertFalse(face.is_unbounded)
        self.assertFalse(face.is_outer_face)

    def test_outer_face(self):
        """Test outer face creation."""
        face = Face()  # No site = outer face

        self.assertIsNone(face.site)
        self.assertTrue(face.is_outer_face)

    def test_polygon_area_calculation(self):
        """Test polygon area calculation using shoelace formula."""
        # Create a simple square
        points = [
            Point(0, 0),
            Point(1, 0),
            Point(1, 1),
            Point(0, 1)
        ]

        face = Face(Point(0.5, 0.5))

        # Mock the boundary points method
        face.get_boundary_points = lambda: points

        area = face.area()
        self.assertAlmostEqual(area, 1.0, places=10)

    def test_point_containment(self):
        """Test point-in-polygon test."""
        # Create a simple square
        points = [
            Point(0, 0),
            Point(2, 0),
            Point(2, 2),
            Point(0, 2)
        ]

        face = Face(Point(1, 1))
        face.get_boundary_points = lambda: points

        # Point inside
        self.assertTrue(face.contains_point(Point(1, 1)))

        # Point outside
        self.assertFalse(face.contains_point(Point(3, 3)))

        # Point on boundary (may be implementation dependent)
        # self.assertTrue(face.contains_point(Point(0, 1)))


if __name__ == '__main__':
    unittest.main()