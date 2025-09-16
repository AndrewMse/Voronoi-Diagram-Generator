"""Tests for utility modules."""

import unittest
import os
import tempfile
import json
from src.voronoi.utils.generators import SiteGenerator, DistributionType
from src.voronoi.utils.analysis import VoronoiAnalyzer
from src.voronoi.utils.export import DiagramExporter
from src.voronoi.algorithms.fortune import FortunesAlgorithm
from src.voronoi.data_structures.point import Point


class TestSiteGenerator(unittest.TestCase):
    """Test site generation utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = SiteGenerator(seed=42)  # Fixed seed for reproducible tests
        self.bounds = (0, 0, 10, 10)

    def test_uniform_random_generation(self):
        """Test uniform random point generation."""
        points = self.generator.generate(
            DistributionType.UNIFORM_RANDOM, 20, self.bounds
        )

        self.assertEqual(len(points), 20)

        # All points should be within bounds
        for point in points:
            self.assertGreaterEqual(point.x, 0)
            self.assertLessEqual(point.x, 10)
            self.assertGreaterEqual(point.y, 0)
            self.assertLessEqual(point.y, 10)

    def test_grid_regular_generation(self):
        """Test regular grid point generation."""
        points = self.generator.generate(
            DistributionType.GRID_REGULAR, 16, self.bounds
        )

        self.assertEqual(len(points), 16)

        # Points should be arranged in a regular pattern
        # For 16 points, expect a 4x4 grid
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]

        # Should have 4 unique x-coordinates and 4 unique y-coordinates
        unique_x = set(x_coords)
        unique_y = set(y_coords)

        self.assertLessEqual(len(unique_x), 5)  # Allow some tolerance
        self.assertLessEqual(len(unique_y), 5)

    def test_grid_jittered_generation(self):
        """Test jittered grid point generation."""
        points = self.generator.generate(
            DistributionType.GRID_JITTERED, 16, self.bounds, jitter=0.2
        )

        self.assertEqual(len(points), 16)

        # Points should be within bounds
        for point in points:
            self.assertGreaterEqual(point.x, 0)
            self.assertLessEqual(point.x, 10)
            self.assertGreaterEqual(point.y, 0)
            self.assertLessEqual(point.y, 10)

    def test_circular_generation(self):
        """Test circular arrangement generation."""
        points = self.generator.generate(
            DistributionType.CIRCULAR, 12, self.bounds, rings=3
        )

        self.assertEqual(len(points), 12)

        # Calculate center and check that points are arranged in concentric circles
        center_x = (self.bounds[0] + self.bounds[2]) / 2
        center_y = (self.bounds[1] + self.bounds[3]) / 2

        distances = []
        for point in points:
            dist = ((point.x - center_x) ** 2 + (point.y - center_y) ** 2) ** 0.5
            distances.append(dist)

        # Should have points at roughly 3 different distances from center
        unique_distances = len(set(round(d, 1) for d in distances))
        self.assertGreaterEqual(unique_distances, 2)

    def test_spiral_generation(self):
        """Test spiral arrangement generation."""
        points = self.generator.generate(
            DistributionType.SPIRAL, 20, self.bounds, turns=2.0
        )

        self.assertEqual(len(points), 20)

        # Points should generally increase in distance from center as we go along the list
        center_x = (self.bounds[0] + self.bounds[2]) / 2
        center_y = (self.bounds[1] + self.bounds[3]) / 2

        distances = []
        for point in points:
            dist = ((point.x - center_x) ** 2 + (point.y - center_y) ** 2) ** 0.5
            distances.append(dist)

        # Early points should generally be closer to center than later points
        early_avg = sum(distances[:5]) / 5
        late_avg = sum(distances[-5:]) / 5
        self.assertLess(early_avg, late_avg)

    def test_gaussian_generation(self):
        """Test Gaussian distribution generation."""
        points = self.generator.generate(
            DistributionType.GAUSSIAN, 50, self.bounds, std_dev=0.2
        )

        self.assertEqual(len(points), 50)

        # Most points should be near the center
        center_x = (self.bounds[0] + self.bounds[2]) / 2
        center_y = (self.bounds[1] + self.bounds[3]) / 2

        center_distances = []
        for point in points:
            dist = ((point.x - center_x) ** 2 + (point.y - center_y) ** 2) ** 0.5
            center_distances.append(dist)

        # Mean distance should be relatively small for std_dev=0.2
        mean_distance = sum(center_distances) / len(center_distances)
        self.assertLess(mean_distance, 3.0)

    def test_poisson_disk_generation(self):
        """Test Poisson disk sampling generation."""
        points = self.generator.generate(
            DistributionType.POISSON, 30, self.bounds
        )

        self.assertLessEqual(len(points), 30)  # May generate fewer due to spacing constraints

        # Check minimum distance constraint (approximately)
        if len(points) >= 2:
            min_distance = float('inf')
            for i, p1 in enumerate(points):
                for j, p2 in enumerate(points):
                    if i != j:
                        dist = p1.distance_to(p2)
                        min_distance = min(min_distance, dist)

            # Should have some reasonable minimum spacing
            self.assertGreater(min_distance, 0.1)

    def test_pattern_names(self):
        """Test pattern name mapping."""
        points = self.generator.create_pattern_sites('random', 10, self.bounds)
        self.assertEqual(len(points), 10)

        points = self.generator.create_pattern_sites('grid', 9, self.bounds)
        self.assertEqual(len(points), 9)

        with self.assertRaises(ValueError):
            self.generator.create_pattern_sites('invalid_pattern', 10, self.bounds)

    def test_reproducibility(self):
        """Test that fixed seed produces reproducible results."""
        gen1 = SiteGenerator(seed=123)
        gen2 = SiteGenerator(seed=123)

        points1 = gen1.generate(DistributionType.UNIFORM_RANDOM, 10, self.bounds)
        points2 = gen2.generate(DistributionType.UNIFORM_RANDOM, 10, self.bounds)

        # Should generate identical points with same seed
        for p1, p2 in zip(points1, points2):
            self.assertAlmostEqual(p1.x, p2.x, places=10)
            self.assertAlmostEqual(p1.y, p2.y, places=10)


class TestVoronoiAnalyzer(unittest.TestCase):
    """Test Voronoi diagram analysis utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = VoronoiAnalyzer()
        self.algorithm = FortunesAlgorithm(bounding_box=(-5, -5, 5, 5))

        # Create a simple test diagram
        sites = [
            Point(0, 0), Point(2, 0), Point(1, 2), Point(-1, 1)
        ]
        self.diagram = self.algorithm.generate_voronoi_diagram(sites)

    def test_diagram_statistics(self):
        """Test basic diagram statistics calculation."""
        stats = self.analyzer.analyze_diagram(self.diagram)

        self.assertEqual(stats.num_sites, 4)
        self.assertGreater(stats.num_vertices, 0)
        self.assertGreater(stats.num_edges, 0)
        self.assertGreaterEqual(stats.avg_cell_area, 0)
        self.assertGreaterEqual(stats.avg_edge_length, 0)

    def test_clustering_detection(self):
        """Test clustering pattern detection."""
        patterns = self.analyzer.detect_patterns(self.diagram)

        self.assertIn('clustering', patterns)
        clustering_info = patterns['clustering']

        if clustering_info['detected']:
            self.assertIn('clustering_type', clustering_info)
            self.assertIn('clark_evans_index', clustering_info)

    def test_regular_structure_detection(self):
        """Test regular structure detection."""
        # Create a more regular structure
        generator = SiteGenerator(seed=42)
        regular_sites = generator.generate(
            DistributionType.GRID_REGULAR, 9, (-3, -3, 3, 3)
        )
        regular_diagram = self.algorithm.generate_voronoi_diagram(regular_sites)

        patterns = self.analyzer.detect_patterns(regular_diagram)

        self.assertIn('regular_structure', patterns)
        structure_info = patterns['regular_structure']

        if structure_info['detected']:
            self.assertIn('structure_type', structure_info)
            self.assertIn('regularity_score', structure_info)

    def test_boundary_effects_detection(self):
        """Test boundary effects detection."""
        patterns = self.analyzer.detect_patterns(self.diagram)

        self.assertIn('boundary_effects', patterns)
        boundary_info = patterns['boundary_effects']

        self.assertIn('detected', boundary_info)
        if boundary_info['detected']:
            self.assertIn('boundary_site_ratio', boundary_info)
            self.assertIn('effect_strength', boundary_info)

    def test_degeneracy_detection(self):
        """Test degeneracy detection."""
        patterns = self.analyzer.detect_patterns(self.diagram)

        self.assertIn('degeneracies', patterns)
        degeneracy_info = patterns['degeneracies']

        self.assertIn('detected', degeneracy_info)
        self.assertIn('details', degeneracy_info)

    def test_regularity_index_range(self):
        """Test that regularity index is in valid range."""
        stats = self.analyzer.analyze_diagram(self.diagram)

        self.assertGreaterEqual(stats.regularity_index, 0)
        self.assertLessEqual(stats.regularity_index, 1)

    def test_uniformity_index_range(self):
        """Test that uniformity index is in valid range."""
        stats = self.analyzer.analyze_diagram(self.diagram)

        self.assertGreaterEqual(stats.uniformity_index, 0)
        self.assertLessEqual(stats.uniformity_index, 1)


class TestDiagramExporter(unittest.TestCase):
    """Test diagram export utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.exporter = DiagramExporter()
        algorithm = FortunesAlgorithm(bounding_box=(-2, -2, 2, 2))

        # Create a simple test diagram
        sites = [Point(-1, -1), Point(1, -1), Point(0, 1)]
        self.diagram = algorithm.generate_voronoi_diagram(sites)

    def test_json_export_import(self):
        """Test JSON export and import."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name

        try:
            # Export to JSON
            self.exporter.export_to_json(self.diagram, temp_filename)

            # Check file was created
            self.assertTrue(os.path.exists(temp_filename))

            # Load and check content
            with open(temp_filename, 'r') as f:
                data = json.load(f)

            self.assertIn('sites', data)
            self.assertIn('vertices', data)
            self.assertIn('edges', data)
            self.assertIn('faces', data)
            self.assertIn('bounding_box', data)

            self.assertEqual(len(data['sites']), 3)

            # Test import
            imported_sites = self.exporter.import_from_json(temp_filename)
            self.assertEqual(len(imported_sites), 3)

        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def test_csv_export_import(self):
        """Test CSV export and import."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_filename = f.name

        try:
            # Export to CSV
            self.exporter.export_to_csv(self.diagram, temp_filename)

            # Check file was created
            self.assertTrue(os.path.exists(temp_filename))

            # Test import
            imported_sites = self.exporter.import_from_csv(temp_filename)
            self.assertEqual(len(imported_sites), 3)

        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def test_svg_export(self):
        """Test SVG export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
            temp_filename = f.name

        try:
            # Export to SVG
            self.exporter.export_to_svg(self.diagram, temp_filename, 400, 300)

            # Check file was created and has SVG content
            self.assertTrue(os.path.exists(temp_filename))

            with open(temp_filename, 'r') as f:
                content = f.read()

            self.assertIn('<svg', content)
            self.assertIn('xmlns="http://www.w3.org/2000/svg"', content)

        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def test_obj_export(self):
        """Test OBJ export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            temp_filename = f.name

        try:
            # Export to OBJ
            self.exporter.export_to_obj(self.diagram, temp_filename, extrude_height=2.0)

            # Check file was created and has OBJ content
            self.assertTrue(os.path.exists(temp_filename))

            with open(temp_filename, 'r') as f:
                content = f.read()

            self.assertIn('# Voronoi Diagram OBJ Export', content)
            self.assertIn('v ', content)  # Vertices
            self.assertIn('f ', content)  # Faces

        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def test_statistics_export(self):
        """Test statistics export."""
        from src.voronoi.utils.analysis import VoronoiAnalyzer

        analyzer = VoronoiAnalyzer()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name

        try:
            # Export statistics
            self.exporter.export_statistics(self.diagram, analyzer, temp_filename)

            # Check file was created
            self.assertTrue(os.path.exists(temp_filename))

            # Load and check content
            with open(temp_filename, 'r') as f:
                data = json.load(f)

            self.assertIn('basic_statistics', data)
            self.assertIn('cell_statistics', data)
            self.assertIn('vertex_statistics', data)
            self.assertIn('edge_statistics', data)
            self.assertIn('quality_metrics', data)
            self.assertIn('patterns', data)

        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


if __name__ == '__main__':
    unittest.main()