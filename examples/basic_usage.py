#!/usr/bin/env python3
"""
Basic usage examples for the Advanced Voronoi Diagram Generator.

This script demonstrates the fundamental operations for generating
and visualizing Voronoi diagrams using Fortune's algorithm.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from voronoi import FortunesAlgorithm, VoronoiRenderer, Point
from voronoi.visualization import ColorScheme
from voronoi.utils import SiteGenerator, DistributionType


def example_simple_diagram():
    """Create a simple Voronoi diagram with a few points."""
    print("Example 1: Simple Voronoi diagram with 5 points")

    # Create some site points
    sites = [
        Point(2, 3),
        Point(5, 1),
        Point(7, 6),
        Point(1, 7),
        Point(8, 2)
    ]

    # Initialize the algorithm
    algorithm = FortunesAlgorithm(bounding_box=(0, 0, 10, 8))

    # Generate the Voronoi diagram
    diagram = algorithm.generate_voronoi_diagram(sites)

    # Print basic statistics
    stats = diagram.get_statistics()
    print(f"Generated diagram with {stats['num_sites']} sites, "
          f"{stats['num_vertices']} vertices, {stats['num_edges']} edges")

    # Visualize the diagram
    renderer = VoronoiRenderer(ColorScheme.VIBRANT)
    renderer.render(diagram, title="Simple Voronoi Diagram", show=True)


def example_random_sites():
    """Generate random sites and create a diagram."""
    print("\nExample 2: Random sites")

    # Use site generator for random points
    generator = SiteGenerator(seed=42)
    sites = generator.generate(
        DistributionType.UNIFORM_RANDOM,
        count=20,
        bounds=(0, 0, 15, 10)
    )

    # Create and visualize diagram
    algorithm = FortunesAlgorithm(bounding_box=(0, 0, 15, 10))
    diagram = algorithm.generate_voronoi_diagram(sites)

    # Use different color scheme and style
    renderer = VoronoiRenderer(ColorScheme.OCEAN)
    renderer.set_style('artistic')
    renderer.render(diagram, title="Random Sites - Ocean Theme", show=True)


def example_pattern_generation():
    """Demonstrate different site patterns."""
    print("\nExample 3: Different site patterns")

    patterns = [
        ('grid', DistributionType.GRID_REGULAR),
        ('jittered_grid', DistributionType.GRID_JITTERED),
        ('circular', DistributionType.CIRCULAR),
        ('spiral', DistributionType.SPIRAL)
    ]

    generator = SiteGenerator(seed=123)
    algorithm = FortunesAlgorithm(bounding_box=(0, 0, 12, 8))

    for pattern_name, pattern_type in patterns:
        print(f"  Generating {pattern_name} pattern...")

        sites = generator.generate(pattern_type, 25, (0, 0, 12, 8))
        diagram = algorithm.generate_voronoi_diagram(sites)

        # Use different color scheme for each pattern
        color_schemes = [ColorScheme.DEFAULT, ColorScheme.SUNSET,
                        ColorScheme.FOREST, ColorScheme.RAINBOW]
        scheme_index = patterns.index((pattern_name, pattern_type))

        renderer = VoronoiRenderer(color_schemes[scheme_index])
        renderer.render(diagram,
                       title=f"{pattern_name.title()} Pattern",
                       show=True)


def example_analysis():
    """Demonstrate diagram analysis capabilities."""
    print("\nExample 4: Diagram analysis")

    from voronoi.utils import VoronoiAnalyzer

    # Generate a test diagram
    generator = SiteGenerator(seed=999)
    sites = generator.generate(
        DistributionType.POISSON,
        30,
        (0, 0, 20, 15)
    )

    algorithm = FortunesAlgorithm(bounding_box=(0, 0, 20, 15))
    diagram = algorithm.generate_voronoi_diagram(sites)

    # Analyze the diagram
    analyzer = VoronoiAnalyzer()
    stats = analyzer.analyze_diagram(diagram)
    patterns = analyzer.detect_patterns(diagram)

    print(f"Analysis Results:")
    print(f"  Number of sites: {stats.num_sites}")
    print(f"  Number of vertices: {stats.num_vertices}")
    print(f"  Number of edges: {stats.num_edges}")
    print(f"  Average cell area: {stats.avg_cell_area:.2f}")
    print(f"  Average edge length: {stats.avg_edge_length:.2f}")
    print(f"  Regularity index: {stats.regularity_index:.3f}")
    print(f"  Uniformity index: {stats.uniformity_index:.3f}")

    # Pattern detection
    if patterns['clustering']['detected']:
        print(f"  Clustering: {patterns['clustering']['clustering_type']}")

    if patterns['regular_structure']['detected']:
        print(f"  Structure: {patterns['regular_structure']['structure_type']}")

    # Visualize with analysis info in title
    renderer = VoronoiRenderer(ColorScheme.SCIENTIFIC)
    title = (f"Poisson Distribution (Regularity: {stats.regularity_index:.2f}, "
             f"Uniformity: {stats.uniformity_index:.2f})")
    renderer.render(diagram, title=title, show=True)


def example_export():
    """Demonstrate exporting diagrams to various formats."""
    print("\nExample 5: Exporting diagrams")

    from voronoi.utils import DiagramExporter, VoronoiAnalyzer

    # Create a diagram
    generator = SiteGenerator(seed=777)
    sites = generator.generate(DistributionType.CIRCULAR, 16, (0, 0, 10, 10))

    algorithm = FortunesAlgorithm(bounding_box=(0, 0, 10, 10))
    diagram = algorithm.generate_voronoi_diagram(sites)

    # Export to different formats
    exporter = DiagramExporter()

    print("  Exporting to JSON...")
    exporter.export_to_json(diagram, 'example_diagram.json')

    print("  Exporting to SVG...")
    exporter.export_to_svg(diagram, 'example_diagram.svg', width=800, height=600)

    print("  Exporting to CSV...")
    exporter.export_to_csv(diagram, 'example_diagram.csv')

    print("  Exporting statistics...")
    analyzer = VoronoiAnalyzer()
    exporter.export_statistics(diagram, analyzer, 'example_stats.json')

    print("  Export complete! Files created:")
    print("    - example_diagram.json")
    print("    - example_diagram.svg")
    print("    - example_diagram.csv")
    print("    - example_stats.json")

    # Visualize the exported diagram
    renderer = VoronoiRenderer(ColorScheme.PASTEL)
    renderer.render(diagram, title="Circular Pattern (Exported)", show=True)


def main():
    """Run all examples."""
    print("Advanced Voronoi Diagram Generator - Basic Usage Examples")
    print("=" * 60)

    try:
        example_simple_diagram()
        example_random_sites()
        example_pattern_generation()
        example_analysis()
        example_export()

        print("\nAll examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()