"""Advanced site generators for creating interesting Voronoi patterns."""

import math
import random
import numpy as np
from typing import List, Tuple, Optional, Callable
from enum import Enum

from ..data_structures.point import Point


class DistributionType(Enum):
    """Types of point distributions."""
    UNIFORM_RANDOM = "uniform_random"
    GAUSSIAN = "gaussian"
    POISSON = "poisson"
    GRID_REGULAR = "grid_regular"
    GRID_JITTERED = "grid_jittered"
    CIRCULAR = "circular"
    SPIRAL = "spiral"
    FRACTAL = "fractal"
    LLOYD_RELAXED = "lloyd_relaxed"
    BLUE_NOISE = "blue_noise"


class SiteGenerator:
    """Advanced generator for creating various types of site distributions."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the site generator.

        Args:
            seed: Random seed for reproducible results
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(self, distribution: DistributionType, count: int,
                bounds: Tuple[float, float, float, float],
                **kwargs) -> List[Point]:
        """
        Generate sites according to the specified distribution.

        Args:
            distribution: Type of distribution to generate
            count: Number of sites to generate
            bounds: (min_x, min_y, max_x, max_y) bounding rectangle
            **kwargs: Additional parameters specific to each distribution

        Returns:
            List of generated site points
        """
        if distribution == DistributionType.UNIFORM_RANDOM:
            return self._uniform_random(count, bounds)
        elif distribution == DistributionType.GAUSSIAN:
            return self._gaussian(count, bounds, **kwargs)
        elif distribution == DistributionType.POISSON:
            return self._poisson_disk(count, bounds, **kwargs)
        elif distribution == DistributionType.GRID_REGULAR:
            return self._grid_regular(count, bounds)
        elif distribution == DistributionType.GRID_JITTERED:
            return self._grid_jittered(count, bounds, **kwargs)
        elif distribution == DistributionType.CIRCULAR:
            return self._circular(count, bounds, **kwargs)
        elif distribution == DistributionType.SPIRAL:
            return self._spiral(count, bounds, **kwargs)
        elif distribution == DistributionType.FRACTAL:
            return self._fractal(count, bounds, **kwargs)
        elif distribution == DistributionType.LLOYD_RELAXED:
            return self._lloyd_relaxation(count, bounds, **kwargs)
        elif distribution == DistributionType.BLUE_NOISE:
            return self._blue_noise(count, bounds, **kwargs)
        else:
            raise ValueError(f"Unknown distribution type: {distribution}")

    def _uniform_random(self, count: int, bounds: Tuple[float, float, float, float]) -> List[Point]:
        """Generate uniformly random points."""
        min_x, min_y, max_x, max_y = bounds
        points = []

        for _ in range(count):
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            points.append(Point(x, y))

        return points

    def _gaussian(self, count: int, bounds: Tuple[float, float, float, float],
                 center: Optional[Tuple[float, float]] = None,
                 std_dev: float = 0.3) -> List[Point]:
        """Generate points with Gaussian distribution."""
        min_x, min_y, max_x, max_y = bounds
        width = max_x - min_x
        height = max_y - min_y

        if center is None:
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
        else:
            center_x, center_y = center

        points = []
        attempts = 0
        max_attempts = count * 10

        while len(points) < count and attempts < max_attempts:
            x = np.random.normal(center_x, std_dev * width)
            y = np.random.normal(center_y, std_dev * height)

            if min_x <= x <= max_x and min_y <= y <= max_y:
                points.append(Point(x, y))

            attempts += 1

        # Fill remaining with uniform random if needed
        while len(points) < count:
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            points.append(Point(x, y))

        return points

    def _poisson_disk(self, count: int, bounds: Tuple[float, float, float, float],
                     min_distance: Optional[float] = None) -> List[Point]:
        """
        Generate points using Poisson disk sampling for uniform spacing.
        Based on Bridson's algorithm.
        """
        min_x, min_y, max_x, max_y = bounds
        width = max_x - min_x
        height = max_y - min_y

        if min_distance is None:
            # Estimate minimum distance based on area and desired count
            area = width * height
            min_distance = math.sqrt(area / count) * 0.7

        cell_size = min_distance / math.sqrt(2)
        grid_width = int(width / cell_size) + 1
        grid_height = int(height / cell_size) + 1

        grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
        points = []
        active_list = []

        # Helper functions
        def get_grid_coords(point: Point) -> Tuple[int, int]:
            gx = int((point.x - min_x) / cell_size)
            gy = int((point.y - min_y) / cell_size)
            return max(0, min(grid_width - 1, gx)), max(0, min(grid_height - 1, gy))

        def is_valid_point(point: Point) -> bool:
            if not (min_x <= point.x <= max_x and min_y <= point.y <= max_y):
                return False

            gx, gy = get_grid_coords(point)

            # Check surrounding cells
            for i in range(max(0, gx - 2), min(grid_width, gx + 3)):
                for j in range(max(0, gy - 2), min(grid_height, gy + 3)):
                    neighbor = grid[i][j]
                    if neighbor and point.distance_to(neighbor) < min_distance:
                        return False
            return True

        # Start with random point
        initial_point = Point(
            random.uniform(min_x, max_x),
            random.uniform(min_y, max_y)
        )
        points.append(initial_point)
        active_list.append(initial_point)
        gx, gy = get_grid_coords(initial_point)
        grid[gx][gy] = initial_point

        # Generate points
        while active_list and len(points) < count:
            random_index = random.randint(0, len(active_list) - 1)
            current_point = active_list[random_index]
            found = False

            # Try to generate new point around current point
            for _ in range(30):  # k attempts
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(min_distance, 2 * min_distance)

                new_x = current_point.x + radius * math.cos(angle)
                new_y = current_point.y + radius * math.sin(angle)
                new_point = Point(new_x, new_y)

                if is_valid_point(new_point):
                    points.append(new_point)
                    active_list.append(new_point)
                    gx, gy = get_grid_coords(new_point)
                    grid[gx][gy] = new_point
                    found = True
                    break

            if not found:
                active_list.pop(random_index)

        return points[:count]

    def _grid_regular(self, count: int, bounds: Tuple[float, float, float, float]) -> List[Point]:
        """Generate points in a regular grid pattern."""
        min_x, min_y, max_x, max_y = bounds
        width = max_x - min_x
        height = max_y - min_y

        # Calculate grid dimensions
        aspect_ratio = width / height
        cols = int(math.sqrt(count * aspect_ratio))
        rows = int(count / cols)

        points = []
        for i in range(rows):
            for j in range(cols):
                if len(points) >= count:
                    break
                x = min_x + (j + 0.5) * width / cols
                y = min_y + (i + 0.5) * height / rows
                points.append(Point(x, y))

        return points

    def _grid_jittered(self, count: int, bounds: Tuple[float, float, float, float],
                      jitter: float = 0.4) -> List[Point]:
        """Generate grid points with random jitter."""
        regular_points = self._grid_regular(count, bounds)
        min_x, min_y, max_x, max_y = bounds
        width = max_x - min_x
        height = max_y - min_y

        # Calculate cell size for jitter bounds
        aspect_ratio = width / height
        cols = int(math.sqrt(count * aspect_ratio))
        rows = int(count / cols)
        cell_width = width / cols
        cell_height = height / rows

        jittered_points = []
        for point in regular_points:
            jitter_x = random.uniform(-jitter * cell_width / 2, jitter * cell_width / 2)
            jitter_y = random.uniform(-jitter * cell_height / 2, jitter * cell_height / 2)

            new_x = max(min_x, min(max_x, point.x + jitter_x))
            new_y = max(min_y, min(max_y, point.y + jitter_y))

            jittered_points.append(Point(new_x, new_y))

        return jittered_points

    def _circular(self, count: int, bounds: Tuple[float, float, float, float],
                 rings: int = 5, angular_jitter: float = 0.1) -> List[Point]:
        """Generate points in concentric circles."""
        min_x, min_y, max_x, max_y = bounds
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        max_radius = min(max_x - min_x, max_y - min_y) / 2 * 0.9

        points = []
        points_per_ring = count // rings
        remainder = count % rings

        for ring in range(rings):
            ring_points = points_per_ring + (1 if ring < remainder else 0)
            if ring_points == 0:
                continue

            radius = (ring + 1) * max_radius / rings

            for i in range(ring_points):
                angle = 2 * math.pi * i / ring_points
                angle += random.uniform(-angular_jitter, angular_jitter)

                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)

                # Ensure point is within bounds
                x = max(min_x, min(max_x, x))
                y = max(min_y, min(max_y, y))

                points.append(Point(x, y))

        return points

    def _spiral(self, count: int, bounds: Tuple[float, float, float, float],
               turns: float = 3.0, noise: float = 0.1) -> List[Point]:
        """Generate points along a spiral pattern."""
        min_x, min_y, max_x, max_y = bounds
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        max_radius = min(max_x - min_x, max_y - min_y) / 2 * 0.9

        points = []

        for i in range(count):
            t = i / count
            angle = turns * 2 * math.pi * t
            radius = t * max_radius

            # Add some noise
            angle += random.uniform(-noise, noise)
            radius += random.uniform(-noise * max_radius / 10, noise * max_radius / 10)

            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)

            # Ensure point is within bounds
            x = max(min_x, min(max_x, x))
            y = max(min_y, min(max_y, y))

            points.append(Point(x, y))

        return points

    def _fractal(self, count: int, bounds: Tuple[float, float, float, float],
                iterations: int = 4) -> List[Point]:
        """Generate points using a simple fractal pattern."""
        min_x, min_y, max_x, max_y = bounds
        points = []

        # Start with corners and center
        initial_points = [
            Point(min_x, min_y),
            Point(max_x, min_y),
            Point(max_x, max_y),
            Point(min_x, max_y),
            Point((min_x + max_x) / 2, (min_y + max_y) / 2)
        ]

        current_points = initial_points[:]

        for iteration in range(iterations):
            new_points = []
            for i in range(len(current_points)):
                for j in range(i + 1, len(current_points)):
                    # Create point between two existing points with some randomness
                    p1, p2 = current_points[i], current_points[j]
                    mid_x = (p1.x + p2.x) / 2
                    mid_y = (p1.y + p2.y) / 2

                    # Add some fractal-like displacement
                    offset = random.uniform(-0.2, 0.2) * (max_x - min_x) / (2 ** iteration)
                    angle = random.uniform(0, 2 * math.pi)

                    new_x = mid_x + offset * math.cos(angle)
                    new_y = mid_y + offset * math.sin(angle)

                    # Ensure within bounds
                    new_x = max(min_x, min(max_x, new_x))
                    new_y = max(min_y, min(max_y, new_y))

                    new_points.append(Point(new_x, new_y))

            current_points.extend(new_points)

            if len(current_points) >= count:
                break

        # Fill with random points if needed
        while len(current_points) < count:
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            current_points.append(Point(x, y))

        return current_points[:count]

    def _lloyd_relaxation(self, count: int, bounds: Tuple[float, float, float, float],
                         iterations: int = 3) -> List[Point]:
        """
        Generate points using Lloyd relaxation for more uniform distribution.
        This is a placeholder - full implementation would require Voronoi computation.
        """
        # Start with random points
        points = self._uniform_random(count, bounds)

        # For now, just apply some simple relaxation
        min_x, min_y, max_x, max_y = bounds

        for _ in range(iterations):
            new_points = []
            for point in points:
                # Simple relaxation: move towards center of nearby points
                nearby = [p for p in points if p != point and point.distance_to(p) < 100]

                if nearby:
                    avg_x = sum(p.x for p in nearby) / len(nearby)
                    avg_y = sum(p.y for p in nearby) / len(nearby)

                    # Move towards average position
                    new_x = point.x + (avg_x - point.x) * 0.1
                    new_y = point.y + (avg_y - point.y) * 0.1

                    # Keep within bounds
                    new_x = max(min_x, min(max_x, new_x))
                    new_y = max(min_y, min(max_y, new_y))

                    new_points.append(Point(new_x, new_y))
                else:
                    new_points.append(point)

            points = new_points

        return points

    def _blue_noise(self, count: int, bounds: Tuple[float, float, float, float],
                   min_distance: Optional[float] = None) -> List[Point]:
        """
        Generate blue noise distribution (similar to Poisson but with different characteristics).
        """
        # For now, use Poisson disk sampling as approximation
        return self._poisson_disk(count, bounds, min_distance)

    def create_pattern_sites(self, pattern_name: str, count: int,
                           bounds: Tuple[float, float, float, float]) -> List[Point]:
        """
        Create sites based on named patterns.

        Args:
            pattern_name: Name of the pattern ('random', 'grid', 'circle', etc.)
            count: Number of sites
            bounds: Bounding rectangle

        Returns:
            List of generated points
        """
        pattern_mapping = {
            'random': DistributionType.UNIFORM_RANDOM,
            'grid': DistributionType.GRID_REGULAR,
            'jittered_grid': DistributionType.GRID_JITTERED,
            'circle': DistributionType.CIRCULAR,
            'spiral': DistributionType.SPIRAL,
            'gaussian': DistributionType.GAUSSIAN,
            'poisson': DistributionType.POISSON,
            'fractal': DistributionType.FRACTAL,
            'relaxed': DistributionType.LLOYD_RELAXED,
            'blue_noise': DistributionType.BLUE_NOISE
        }

        if pattern_name not in pattern_mapping:
            raise ValueError(f"Unknown pattern: {pattern_name}")

        return self.generate(pattern_mapping[pattern_name], count, bounds)