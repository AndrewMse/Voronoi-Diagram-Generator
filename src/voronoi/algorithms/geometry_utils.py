"""Geometric utility functions for Voronoi diagram computations."""

import math
from typing import Optional, Tuple
from ..data_structures.point import Point


class GeometryUtils:
    """Collection of geometric utility functions optimized for Voronoi computations."""

    @staticmethod
    def circumcenter(p1: Point, p2: Point, p3: Point) -> Optional[Point]:
        """
        Calculate the circumcenter of three points.
        This is where a circle event occurs in Fortune's algorithm.

        Args:
            p1, p2, p3: The three points

        Returns:
            The circumcenter point, or None if points are collinear
        """
        # Check for collinearity using cross product
        if GeometryUtils.are_collinear(p1, p2, p3):
            return None

        # Calculate circumcenter using determinant formula
        ax, ay = p1.x, p1.y
        bx, by = p2.x, p2.y
        cx, cy = p3.x, p3.y

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

        if abs(d) < 1e-10:
            return None  # Points are effectively collinear

        # Circumcenter coordinates
        ux = ((ax * ax + ay * ay) * (by - cy) +
              (bx * bx + by * by) * (cy - ay) +
              (cx * cx + cy * cy) * (ay - by)) / d

        uy = ((ax * ax + ay * ay) * (cx - bx) +
              (bx * bx + by * by) * (ax - cx) +
              (cx * cx + cy * cy) * (bx - ax)) / d

        return Point(ux, uy)

    @staticmethod
    def circumradius(p1: Point, p2: Point, p3: Point) -> Optional[float]:
        """
        Calculate the circumradius of three points.

        Args:
            p1, p2, p3: The three points

        Returns:
            The circumradius, or None if points are collinear
        """
        center = GeometryUtils.circumcenter(p1, p2, p3)
        if center is None:
            return None

        return center.distance_to(p1)

    @staticmethod
    def are_collinear(p1: Point, p2: Point, p3: Point, tolerance: float = 1e-10) -> bool:
        """
        Check if three points are collinear.

        Args:
            p1, p2, p3: The three points
            tolerance: Tolerance for floating-point comparison

        Returns:
            True if points are collinear
        """
        # Use cross product to check collinearity
        cross_product = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
        return abs(cross_product) < tolerance

    @staticmethod
    def point_in_circle(point: Point, center: Point, radius: float) -> bool:
        """
        Check if a point lies inside a circle.

        Args:
            point: The point to test
            center: Center of the circle
            radius: Radius of the circle

        Returns:
            True if point is inside the circle
        """
        distance_squared = point.distance_squared_to(center)
        return distance_squared < (radius * radius - 1e-10)

    @staticmethod
    def point_on_circle(point: Point, center: Point, radius: float, tolerance: float = 1e-10) -> bool:
        """
        Check if a point lies on a circle.

        Args:
            point: The point to test
            center: Center of the circle
            radius: Radius of the circle
            tolerance: Tolerance for floating-point comparison

        Returns:
            True if point is on the circle
        """
        distance = point.distance_to(center)
        return abs(distance - radius) < tolerance

    @staticmethod
    def line_intersection(p1: Point, d1: Point, p2: Point, d2: Point) -> Optional[Point]:
        """
        Find intersection of two lines defined by point + direction.

        Args:
            p1: Point on first line
            d1: Direction of first line
            p2: Point on second line
            d2: Direction of second line

        Returns:
            Intersection point, or None if lines are parallel
        """
        denominator = d1.cross(d2)

        if abs(denominator) < 1e-10:
            return None  # Lines are parallel

        t = (p2 - p1).cross(d2) / denominator
        return p1 + d1 * t

    @staticmethod
    def perpendicular_bisector(p1: Point, p2: Point) -> Tuple[Point, Point]:
        """
        Get the perpendicular bisector of two points.

        Args:
            p1: First point
            p2: Second point

        Returns:
            Tuple of (midpoint, direction_vector)
        """
        midpoint = (p1 + p2) / 2
        direction = (p2 - p1).perpendicular().normalize()
        return midpoint, direction

    @staticmethod
    def distance_point_to_line(point: Point, line_point: Point, line_direction: Point) -> float:
        """
        Calculate the distance from a point to a line.

        Args:
            point: The point
            line_point: A point on the line
            line_direction: Direction vector of the line

        Returns:
            The perpendicular distance
        """
        if line_direction.magnitude < 1e-10:
            return point.distance_to(line_point)

        # Vector from line_point to point
        to_point = point - line_point

        # Project onto line direction
        projection_length = to_point.dot(line_direction) / line_direction.magnitude
        projection = line_direction.normalize() * projection_length

        # Perpendicular component
        perpendicular = to_point - projection

        return perpendicular.magnitude

    @staticmethod
    def is_point_left_of_line(point: Point, line_start: Point, line_end: Point) -> bool:
        """
        Check if a point is to the left of a directed line.

        Args:
            point: The point to test
            line_start: Start point of the line
            line_end: End point of the line

        Returns:
            True if point is to the left of the line
        """
        return ((line_end.x - line_start.x) * (point.y - line_start.y) -
                (line_end.y - line_start.y) * (point.x - line_start.x)) > 0

    @staticmethod
    def angle_between_vectors(v1: Point, v2: Point) -> float:
        """
        Calculate the angle between two vectors in radians.

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Angle in radians [0, π]
        """
        dot_product = v1.dot(v2)
        magnitude_product = v1.magnitude * v2.magnitude

        if magnitude_product < 1e-10:
            return 0.0

        # Clamp to avoid numerical errors
        cos_angle = max(-1.0, min(1.0, dot_product / magnitude_product))
        return math.acos(cos_angle)

    @staticmethod
    def orient2d(pa: Point, pb: Point, pc: Point) -> float:
        """
        Robust 2D orientation test.
        Returns positive if abc is counterclockwise, negative if clockwise, zero if collinear.

        Args:
            pa, pb, pc: The three points

        Returns:
            Orientation value
        """
        return (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x)

    @staticmethod
    def in_circle(pa: Point, pb: Point, pc: Point, pd: Point) -> bool:
        """
        Test if point pd lies inside the circle defined by points pa, pb, pc.
        Assumes pa, pb, pc are in counterclockwise order.

        Args:
            pa, pb, pc: Points defining the circle (counterclockwise)
            pd: Point to test

        Returns:
            True if pd is inside the circle
        """
        center = GeometryUtils.circumcenter(pa, pb, pc)
        if center is None:
            return False

        radius = center.distance_to(pa)
        return GeometryUtils.point_in_circle(pd, center, radius)

    @staticmethod
    def convex_hull_area(points: list[Point]) -> float:
        """
        Calculate the area of a convex polygon defined by points.

        Args:
            points: List of points in counterclockwise order

        Returns:
            Area of the polygon
        """
        if len(points) < 3:
            return 0.0

        area = 0.0
        n = len(points)

        for i in range(n):
            j = (i + 1) % n
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y

        return abs(area) / 2.0