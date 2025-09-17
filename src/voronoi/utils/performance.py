"""Performance profiling utilities for Voronoi diagram generation."""

import time
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading


@dataclass
class PerformanceMetrics:
    """Container for performance measurement data."""
    operation_name: str
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_percent: float
    site_count: int
    vertices_generated: int
    edges_generated: int
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return (f"PerformanceMetrics("
                f"operation='{self.operation_name}', "
                f"time={self.execution_time:.4f}s, "
                f"memory={self.memory_usage_mb:.1f}MB, "
                f"sites={self.site_count})")


class PerformanceProfiler:
    """
    Advanced performance profiler for Voronoi diagram operations.
    Tracks execution time, memory usage, and algorithmic complexity.
    """

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self._current_operation: Optional[str] = None
        self._start_time: float = 0
        self._start_memory: float = 0
        self._peak_memory: float = 0
        self._memory_monitor: Optional[threading.Thread] = None
        self._monitoring: bool = False

    @contextmanager
    def profile_operation(self, operation_name: str, site_count: int = 0):
        """
        Context manager for profiling an operation.

        Args:
            operation_name: Name of the operation being profiled
            site_count: Number of sites being processed

        Usage:
            with profiler.profile_operation("voronoi_generation", len(sites)):
                diagram = algorithm.generate_voronoi_diagram(sites)
        """
        self.start_profiling(operation_name)
        try:
            yield self
        finally:
            metrics = self.stop_profiling(site_count)
            return metrics

    def start_profiling(self, operation_name: str) -> None:
        """Start profiling an operation."""
        gc.collect()  # Clean garbage before measuring

        self._current_operation = operation_name
        self._start_time = time.perf_counter()
        self._start_memory = self._get_memory_usage()
        self._peak_memory = self._start_memory

        # Start memory monitoring
        self._monitoring = True
        self._memory_monitor = threading.Thread(target=self._monitor_memory)
        self._memory_monitor.daemon = True
        self._memory_monitor.start()

    def stop_profiling(self, site_count: int = 0, vertices_generated: int = 0,
                      edges_generated: int = 0) -> PerformanceMetrics:
        """
        Stop profiling and record metrics.

        Args:
            site_count: Number of sites processed
            vertices_generated: Number of vertices generated
            edges_generated: Number of edges generated

        Returns:
            Performance metrics for the operation
        """
        if not self._current_operation:
            raise RuntimeError("No profiling operation in progress")

        self._monitoring = False
        if self._memory_monitor and self._memory_monitor.is_alive():
            self._memory_monitor.join(timeout=0.1)

        execution_time = time.perf_counter() - self._start_time
        final_memory = self._get_memory_usage()
        memory_used = final_memory - self._start_memory
        cpu_percent = psutil.cpu_percent()

        metrics = PerformanceMetrics(
            operation_name=self._current_operation,
            execution_time=execution_time,
            memory_usage_mb=memory_used,
            peak_memory_mb=self._peak_memory - self._start_memory,
            cpu_percent=cpu_percent,
            site_count=site_count,
            vertices_generated=vertices_generated,
            edges_generated=edges_generated
        )

        self.metrics_history.append(metrics)
        self._current_operation = None

        return metrics

    def _monitor_memory(self) -> None:
        """Monitor peak memory usage in a separate thread."""
        while self._monitoring:
            current_memory = self._get_memory_usage()
            if current_memory > self._peak_memory:
                self._peak_memory = current_memory
            time.sleep(0.01)  # Check every 10ms

    @staticmethod
    def _get_memory_usage() -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def benchmark_algorithm(self, algorithm_func: Callable, site_counts: List[int],
                          iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark an algorithm with different site counts.

        Args:
            algorithm_func: Function to benchmark (should take sites as argument)
            site_counts: List of site counts to test
            iterations: Number of iterations per site count

        Returns:
            Benchmark results with performance analysis
        """
        from ..utils.generators import SiteGenerator, DistributionType

        results = {
            'site_counts': site_counts,
            'metrics': [],
            'analysis': {}
        }

        generator = SiteGenerator()
        bounds = (0, 0, 1000, 1000)

        print(f"Running benchmark with {len(site_counts)} site counts, {iterations} iterations each...")

        for site_count in site_counts:
            iteration_metrics = []

            for iteration in range(iterations):
                # Generate sites
                sites = generator.generate(
                    DistributionType.UNIFORM_RANDOM,
                    site_count,
                    bounds
                )

                # Profile the algorithm
                with self.profile_operation(f"benchmark_{site_count}_sites", site_count) as profiler:
                    try:
                        diagram = algorithm_func(sites)
                        vertices = len(diagram.vertices) if hasattr(diagram, 'vertices') else 0
                        edges = len(diagram.edges) if hasattr(diagram, 'edges') else 0

                        # Update metrics with diagram info
                        metrics = self.metrics_history[-1]
                        metrics.vertices_generated = vertices
                        metrics.edges_generated = edges

                        iteration_metrics.append(metrics)

                    except Exception as e:
                        print(f"Error with {site_count} sites, iteration {iteration}: {e}")
                        continue

            if iteration_metrics:
                # Calculate average metrics for this site count
                avg_time = sum(m.execution_time for m in iteration_metrics) / len(iteration_metrics)
                avg_memory = sum(m.memory_usage_mb for m in iteration_metrics) / len(iteration_metrics)
                avg_peak_memory = sum(m.peak_memory_mb for m in iteration_metrics) / len(iteration_metrics)

                results['metrics'].append({
                    'site_count': site_count,
                    'avg_time': avg_time,
                    'avg_memory': avg_memory,
                    'avg_peak_memory': avg_peak_memory,
                    'all_metrics': iteration_metrics
                })

        # Analyze results
        results['analysis'] = self._analyze_benchmark_results(results)

        return results

    def _analyze_benchmark_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze benchmark results to determine complexity and performance characteristics."""
        if not results['metrics']:
            return {}

        site_counts = [m['site_count'] for m in results['metrics']]
        times = [m['avg_time'] for m in results['metrics']]
        memories = [m['avg_memory'] for m in results['metrics']]

        analysis = {
            'time_complexity': self._estimate_time_complexity(site_counts, times),
            'memory_complexity': self._estimate_memory_complexity(site_counts, memories),
            'performance_summary': {
                'min_time': min(times),
                'max_time': max(times),
                'time_per_site_avg': sum(t/n for t, n in zip(times, site_counts)) / len(times),
                'memory_efficiency': memories[-1] / site_counts[-1] if site_counts else 0
            }
        }

        return analysis

    def _estimate_time_complexity(self, site_counts: List[int], times: List[float]) -> str:
        """Estimate time complexity based on growth patterns."""
        if len(site_counts) < 2:
            return "insufficient_data"

        # Calculate growth ratios
        ratios = []
        for i in range(1, len(site_counts)):
            n_ratio = site_counts[i] / site_counts[i-1]
            t_ratio = times[i] / times[i-1]
            if n_ratio > 1 and t_ratio > 0:
                ratios.append(t_ratio / n_ratio)

        if not ratios:
            return "insufficient_data"

        avg_ratio = sum(ratios) / len(ratios)

        # Classify complexity
        if avg_ratio < 1.5:
            return "O(n)"  # Linear
        elif avg_ratio < 2.5:
            return "O(n log n)"  # Log-linear
        elif avg_ratio < 4.0:
            return "O(n²)"  # Quadratic
        else:
            return "O(n³) or worse"

    def _estimate_memory_complexity(self, site_counts: List[int], memories: List[float]) -> str:
        """Estimate memory complexity."""
        if len(site_counts) < 2:
            return "insufficient_data"

        # Simple linear regression on log-log scale
        import math

        try:
            log_n = [math.log(n) for n in site_counts]
            log_m = [math.log(m) if m > 0 else 0 for m in memories]

            # Estimate slope
            n_mean = sum(log_n) / len(log_n)
            m_mean = sum(log_m) / len(log_m)

            numerator = sum((n - n_mean) * (m - m_mean) for n, m in zip(log_n, log_m))
            denominator = sum((n - n_mean) ** 2 for n in log_n)

            if denominator == 0:
                return "constant"

            slope = numerator / denominator

            if slope < 0.5:
                return "O(1)"  # Constant
            elif slope < 1.5:
                return "O(n)"  # Linear
            elif slope < 2.5:
                return "O(n²)"  # Quadratic
            else:
                return "O(n³) or worse"

        except (ValueError, ZeroDivisionError):
            return "analysis_error"

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of all recorded performance metrics."""
        if not self.metrics_history:
            return {'message': 'No performance data recorded'}

        total_operations = len(self.metrics_history)
        total_time = sum(m.execution_time for m in self.metrics_history)
        total_sites = sum(m.site_count for m in self.metrics_history)

        # Group by operation type
        by_operation = {}
        for metric in self.metrics_history:
            if metric.operation_name not in by_operation:
                by_operation[metric.operation_name] = []
            by_operation[metric.operation_name].append(metric)

        operation_summaries = {}
        for operation, metrics in by_operation.items():
            operation_summaries[operation] = {
                'count': len(metrics),
                'avg_time': sum(m.execution_time for m in metrics) / len(metrics),
                'total_time': sum(m.execution_time for m in metrics),
                'avg_memory': sum(m.memory_usage_mb for m in metrics) / len(metrics),
                'total_sites': sum(m.site_count for m in metrics)
            }

        return {
            'total_operations': total_operations,
            'total_time': total_time,
            'total_sites': total_sites,
            'avg_time_per_operation': total_time / total_operations,
            'operations_per_second': total_operations / total_time if total_time > 0 else 0,
            'by_operation': operation_summaries,
            'recent_metrics': self.metrics_history[-10:]  # Last 10 operations
        }

    def clear_history(self) -> None:
        """Clear all recorded performance metrics."""
        self.metrics_history.clear()

    def export_metrics(self, filename: str) -> None:
        """Export metrics to a CSV file for analysis."""
        import csv

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'operation_name', 'execution_time', 'memory_usage_mb',
                'peak_memory_mb', 'cpu_percent', 'site_count',
                'vertices_generated', 'edges_generated', 'timestamp'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for metric in self.metrics_history:
                writer.writerow({
                    'operation_name': metric.operation_name,
                    'execution_time': metric.execution_time,
                    'memory_usage_mb': metric.memory_usage_mb,
                    'peak_memory_mb': metric.peak_memory_mb,
                    'cpu_percent': metric.cpu_percent,
                    'site_count': metric.site_count,
                    'vertices_generated': metric.vertices_generated,
                    'edges_generated': metric.edges_generated,
                    'timestamp': metric.timestamp
                })

        print(f"Exported {len(self.metrics_history)} metrics to {filename}")