#!/usr/bin/env python3
"""
Advanced Voronoi Diagram Generator

A comprehensive implementation of Fortune's sweep line algorithm for generating
beautiful and mathematically accurate Voronoi diagrams.

Usage:
    python main.py [options]

Examples:
    python main.py --interactive         # Launch interactive GUI
    python main.py --sites 50 --render   # Generate and render 50 random sites
    python main.py --pattern grid --count 25 --export diagram.svg
"""

import argparse
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from voronoi import FortunesAlgorithm, VoronoiRenderer, Point
from voronoi.utils import SiteGenerator, PerformanceProfiler, DiagramExporter, VoronoiAnalyzer
from voronoi.visualization import ColorScheme
from voronoi.visualization.interactive import InteractiveVoronoiApp
from voronoi.config import get_config


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced Voronoi Diagram Generator using Fortune's Algorithm"
    )

    # Mode selection
    parser.add_argument('--interactive', action='store_true',
                       help='Launch interactive GUI application')
    parser.add_argument('--batch', action='store_true',
                       help='Run in batch mode (default)')

    # Site generation options
    parser.add_argument('--sites', '-n', type=int, default=20,
                       help='Number of sites to generate (default: 20)')
    parser.add_argument('--pattern', choices=[
        'random', 'grid', 'jittered_grid', 'circle', 'spiral',
        'gaussian', 'poisson', 'fractal', 'relaxed', 'blue_noise'
    ], default='random', help='Site distribution pattern (default: random)')
    config = get_config()
    default_bounds = config.get('algorithm.default_bounding_box', [0, 0, 800, 600])
    parser.add_argument('--bounds', nargs=4, type=float,
                       default=default_bounds, metavar=('MIN_X', 'MIN_Y', 'MAX_X', 'MAX_Y'),
                       help=f'Bounding box for site generation (default: {" ".join(map(str, default_bounds))})')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')

    # Visualization options
    parser.add_argument('--render', action='store_true',
                       help='Render the diagram')
    parser.add_argument('--color-scheme', choices=[scheme.value for scheme in ColorScheme],
                       default='default', help='Color scheme for visualization')
    parser.add_argument('--style', choices=['minimal', 'scientific', 'artistic'],
                       default='scientific', help='Visualization style')

    # Export options
    parser.add_argument('--export', metavar='FILENAME',
                       help='Export diagram to file (format determined by extension)')
    parser.add_argument('--export-stats', metavar='FILENAME',
                       help='Export statistics to JSON file')

    # Analysis options
    parser.add_argument('--analyze', action='store_true',
                       help='Perform detailed analysis of the diagram')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')

    # Performance options
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')

    args = parser.parse_args()

    # Launch interactive mode
    if args.interactive:
        print("Launching interactive Voronoi diagram application...")
        app = InteractiveVoronoiApp()
        app.run()
        return

    # Batch mode processing
    print(f"Advanced Voronoi Diagram Generator")
    print(f"Generating {args.sites} sites using {args.pattern} pattern...")

    # Initialize components
    generator = SiteGenerator(seed=args.seed)
    algorithm = FortunesAlgorithm(bounding_box=tuple(args.bounds))
    profiler = PerformanceProfiler() if args.profile else None
    analyzer = VoronoiAnalyzer() if args.analyze else None
    exporter = DiagramExporter()

    try:
        # Generate sites
        sites = generator.create_pattern_sites(args.pattern, args.sites, tuple(args.bounds))
        print(f"Generated {len(sites)} sites")

        # Generate Voronoi diagram
        if profiler:
            with profiler.profile_operation("voronoi_generation", len(sites)):
                diagram = algorithm.generate_voronoi_diagram(sites)
        else:
            diagram = algorithm.generate_voronoi_diagram(sites)

        stats = diagram.get_statistics()
        print(f"Generated diagram: {stats['num_vertices']} vertices, {stats['num_edges']} edges")

        # Perform analysis
        if analyzer:
            print("\nPerforming diagram analysis...")
            analysis = analyzer.analyze_diagram(diagram)
            patterns = analyzer.detect_patterns(diagram)

            print(f"Analysis results:")
            print(f"  Regularity index: {analysis.regularity_index:.3f}")
            print(f"  Uniformity index: {analysis.uniformity_index:.3f}")
            print(f"  Average cell area: {analysis.avg_cell_area:.2f}")
            print(f"  Average edge length: {analysis.avg_edge_length:.2f}")

            if patterns['clustering']['detected']:
                print(f"  Clustering detected: {patterns['clustering']['clustering_type']}")
            if patterns['regular_structure']['detected']:
                print(f"  Structure type: {patterns['regular_structure']['structure_type']}")

        # Export statistics
        if args.export_stats and analyzer:
            exporter.export_statistics(diagram, analyzer, args.export_stats)
            print(f"Statistics exported to {args.export_stats}")

        # Render diagram
        if args.render:
            print("Rendering diagram...")
            color_scheme = ColorScheme(args.color_scheme)
            renderer = VoronoiRenderer(color_scheme)
            renderer.set_style(args.style)

            figure = renderer.render(diagram,
                                   title=f"Voronoi Diagram - {args.pattern.title()} Pattern ({args.sites} sites)",
                                   show=True)

        # Export diagram
        if args.export:
            print(f"Exporting diagram to {args.export}...")
            extension = os.path.splitext(args.export)[1].lower()

            if extension == '.json':
                exporter.export_to_json(diagram, args.export)
            elif extension == '.svg':
                exporter.export_to_svg(diagram, args.export)
            elif extension == '.csv':
                exporter.export_to_csv(diagram, args.export)
            elif extension == '.obj':
                exporter.export_to_obj(diagram, args.export)
            else:
                print(f"Unsupported export format: {extension}")
                return

            print(f"Exported successfully")

        # Performance profiling results
        if profiler:
            print("\nPerformance Results:")
            summary = profiler.get_performance_summary()
            for operation, data in summary['by_operation'].items():
                print(f"  {operation}: {data['avg_time']:.4f}s average")

        # Run benchmark
        if args.benchmark:
            print("\nRunning performance benchmark...")
            site_counts = [10, 25, 50, 100, 200, 500, 1000]

            if not profiler:
                profiler = PerformanceProfiler()

            def algorithm_func(sites_list):
                return algorithm.generate_voronoi_diagram(sites_list)

            results = profiler.benchmark_algorithm(algorithm_func, site_counts[:5])  # Limit for demo

            print("Benchmark Results:")
            for result in results['metrics']:
                print(f"  {result['site_count']} sites: {result['avg_time']:.4f}s")

            if results['analysis']:
                print(f"Estimated time complexity: {results['analysis']['time_complexity']}")

        print("\nComplete!")

    except Exception as e:
        print(f"Error: {e}")
        if args.profile and profiler:
            print("Profiling data may be incomplete due to error")
        sys.exit(1)


if __name__ == '__main__':
    main()