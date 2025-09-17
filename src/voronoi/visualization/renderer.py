"""Advanced renderer for Voronoi diagrams with beautiful visualizations."""

import math
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection, PatchCollection
import numpy as np

from ..data_structures.voronoi_diagram import VoronoiDiagram
from ..data_structures.point import Point
from ..data_structures.vertex import Vertex
from ..data_structures.edge import Edge
from ..data_structures.face import Face
from .colors import ColorPalette, ColorScheme
from ..config import get_config


class VoronoiRenderer:
    """
    Advanced renderer for Voronoi diagrams with customizable styling and animations.
    """

    def __init__(self, color_scheme: ColorScheme = ColorScheme.DEFAULT):
        config = get_config()

        self.color_palette = ColorPalette(color_scheme)

        # Load settings from config
        self.figure_size = tuple(config.get('rendering.figure_size', [12, 10]))
        self.dpi = config.get('rendering.dpi', 100)

        # Rendering options
        self.show_sites = config.get('visualization.show_sites', True)
        self.show_vertices = config.get('visualization.show_vertices', False)
        self.show_edges = config.get('visualization.show_edges', False)
        self.show_faces = config.get('visualization.show_faces', True)
        self.show_grid = config.get('visualization.show_grid', False)
        self.show_bounding_box = config.get('visualization.show_bounding_box', False)

        # Debug output
        print(f"DEBUG: Loaded show_vertices = {self.show_vertices} (type: {type(self.show_vertices)})")
        print(f"DEBUG: Loaded show_edges = {self.show_edges} (type: {type(self.show_edges)})")

        # Style options
        self.site_size = config.get('sites.site_size', 50)
        self.vertex_size = config.get('vertices.vertex_size', 20)
        self.edge_width = config.get('edges.edge_width', 1.5)
        self.site_marker = config.get('sites.site_marker', 'o')
        self.vertex_marker = config.get('vertices.vertex_marker', 's')

        # Color options
        self.site_color = config.get('colors.site_color', 'red')
        self.vertex_color = config.get('colors.vertex_color', 'blue')
        self.edge_color = config.get('edges.edge_color', 'black')
        self.background_color = config.get('colors.background_color', None)
        self.face_alpha = config.get('rendering.face_alpha', 1.0)
        self.edge_alpha = config.get('rendering.edge_alpha', 0.8)

        # Advanced options
        self.anti_aliasing = config.get('rendering.anti_aliasing', True)
        self.high_quality = config.get('rendering.high_quality', True)
        self.gradient_faces = config.get('colors.gradient_faces', False)

    def render(self, diagram: VoronoiDiagram, title: str = "Voronoi Diagram",
               save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Render the Voronoi diagram.

        Args:
            diagram: The Voronoi diagram to render
            title: Title for the plot
            save_path: Optional path to save the image
            show: Whether to display the plot

        Returns:
            The matplotlib Figure object
        """
        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size, dpi=self.dpi)

        # Set up the plot
        self._setup_plot(ax, diagram, title)

        # Render components in order
        if self.show_faces:
            print(f"Rendering faces: {len(diagram.faces)} faces")
            self._render_faces(ax, diagram)

        if self.show_edges:
            print(f"Rendering edges: {len(diagram.edges)} edges")
            self._render_edges(ax, diagram)
        else:
            print("Edges disabled - not rendering")

        if self.show_vertices:
            print(f"Rendering vertices: {len(diagram.vertices)} vertices")
            self._render_vertices(ax, diagram)
        else:
            print("Vertices disabled - not rendering")

        if self.show_sites:
            print(f"Rendering sites: {len(diagram.sites)} sites")
            self._render_sites(ax, diagram)

        if self.show_grid:
            self._render_grid(ax, diagram)

        if self.show_bounding_box:
            self._render_bounding_box(ax, diagram)

        # Final plot adjustments
        self._finalize_plot(ax, diagram)

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=self.background_color or 'white')

        # Show if requested
        if show:
            plt.show(block=False)  # Don't block so we can see debug output
            plt.pause(3)  # Show for 3 seconds

        return fig

    def _setup_plot(self, ax: plt.Axes, diagram: VoronoiDiagram, title: str) -> None:
        """Set up the basic plot properties."""
        # Set title
        ax.set_title(title, fontsize=16, fontweight='bold')

        # Get complementary colors
        comp_colors = self.color_palette.get_complementary_colors()

        # Set background color
        if self.background_color:
            ax.set_facecolor(self.background_color)
        else:
            bg_color = [c/255.0 for c in comp_colors['background']]
            ax.set_facecolor(bg_color)

        # Set text color
        text_color = [c/255.0 for c in comp_colors['text']]
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)

        # Equal aspect ratio
        ax.set_aspect('equal')

        # Remove axes if high quality mode
        if self.high_quality:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    def _render_faces(self, ax: plt.Axes, diagram: VoronoiDiagram) -> None:
        """Render the Voronoi faces (cells)."""
        patches_list = []
        colors = []

        site_index = 0
        bounds = diagram.bounds()

        for face in diagram.faces:
            if face.is_unbounded or not face.site:
                continue

            # Get boundary points
            boundary_points = face.get_boundary_points()
            if len(boundary_points) < 3:
                continue

            # Create polygon patch
            coords = [(p.x, p.y) for p in boundary_points]

            if self.gradient_faces:
                # Create gradient effect (simplified)
                color = self.color_palette.get_site_color_by_position(
                    face.site.x, face.site.y, bounds
                )
            else:
                color = self.color_palette.get_site_color(site_index, len(diagram.sites))

            # Convert to matplotlib color format
            color_rgb = [c/255.0 for c in color]
            colors.append(color_rgb)

            # Create polygon
            polygon = patches.Polygon(coords, closed=True)
            patches_list.append(polygon)

            site_index += 1

        # Add all patches at once for better performance
        if patches_list:
            patch_collection = PatchCollection(patches_list, facecolors=colors,
                                             alpha=1.0, edgecolors='none')
            ax.add_collection(patch_collection)

    def _render_edges(self, ax: plt.Axes, diagram: VoronoiDiagram) -> None:
        """Render the Voronoi edges."""
        lines = []
        edge_colors = []

        comp_colors = self.color_palette.get_complementary_colors()
        default_edge_color = [c/255.0 for c in comp_colors.get('text', (200, 200, 200))]

        for edge in diagram.edges:
            if not edge.is_complete:
                continue

            start = edge.half_edge1.origin
            end = edge.half_edge1.destination

            if start and end:
                line = [(start.x, start.y), (end.x, end.y)]
                lines.append(line)

                # Use custom edge color or default
                if isinstance(self.edge_color, str) and self.edge_color != 'black':
                    edge_colors.append(self.edge_color)
                else:
                    edge_colors.append(default_edge_color)

        # Add all lines at once for better performance
        if lines:
            if len(set(map(tuple, edge_colors))) == 1:
                # All edges have the same color
                line_collection = LineCollection(lines, colors=edge_colors[0],
                                               linewidths=self.edge_width,
                                               alpha=self.edge_alpha,
                                               antialiased=self.anti_aliasing)
            else:
                # Different colors for different edges
                line_collection = LineCollection(lines, colors=edge_colors,
                                               linewidths=self.edge_width,
                                               alpha=self.edge_alpha,
                                               antialiased=self.anti_aliasing)

            ax.add_collection(line_collection)

    def _render_vertices(self, ax: plt.Axes, diagram: VoronoiDiagram) -> None:
        """Render the Voronoi vertices."""
        if not diagram.vertices:
            return

        x_coords = [v.x for v in diagram.vertices]
        y_coords = [v.y for v in diagram.vertices]

        # Get vertex color
        comp_colors = self.color_palette.get_complementary_colors()
        if isinstance(self.vertex_color, str) and self.vertex_color != 'blue':
            color = self.vertex_color
        else:
            color = [c/255.0 for c in comp_colors.get('highlight', (255, 200, 100))]

        ax.scatter(x_coords, y_coords, s=self.vertex_size, color=color,
                  marker=self.vertex_marker, edgecolors='white', linewidths=0.5,
                  alpha=0.9, zorder=5)

    def _render_sites(self, ax: plt.Axes, diagram: VoronoiDiagram) -> None:
        """Render the site points."""
        if not diagram.sites:
            return

        x_coords = [s.x for s in diagram.sites]
        y_coords = [s.y for s in diagram.sites]

        # Get site color
        comp_colors = self.color_palette.get_complementary_colors()
        if isinstance(self.site_color, str) and self.site_color != 'red':
            color = self.site_color
        else:
            # Use a contrasting color for sites
            color = [c/255.0 for c in (255, 100, 100)]

        config = get_config()
        site_alpha = config.get('rendering.site_alpha', 0.3)
        ax.scatter(x_coords, y_coords, s=self.site_size, color=color,
                  marker=self.site_marker, edgecolors='white', linewidths=1,
                  alpha=site_alpha, zorder=10)

    def _render_grid(self, ax: plt.Axes, diagram: VoronoiDiagram) -> None:
        """Render a subtle grid."""
        comp_colors = self.color_palette.get_complementary_colors()
        grid_color = [c/255.0 for c in comp_colors['grid']]

        ax.grid(True, alpha=0.3, color=grid_color, linewidth=0.5)

    def _render_bounding_box(self, ax: plt.Axes, diagram: VoronoiDiagram) -> None:
        """Render the bounding box."""
        min_x, min_y, max_x, max_y = diagram.bounding_box

        # Create bounding box rectangle
        bbox_patch = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                     linewidth=2, edgecolor='gray',
                                     facecolor='none', alpha=0.5)
        ax.add_patch(bbox_patch)

    def _finalize_plot(self, ax: plt.Axes, diagram: VoronoiDiagram) -> None:
        """Final plot adjustments."""
        # Set axis limits with padding
        bounds = diagram.bounds()
        if bounds:
            min_x, min_y, max_x, max_y = bounds
            padding = max(max_x - min_x, max_y - min_y) * 0.05
            ax.set_xlim(min_x - padding, max_x + padding)
            ax.set_ylim(min_y - padding, max_y + padding)
        else:
            # Use bounding box as fallback
            min_x, min_y, max_x, max_y = diagram.bounding_box
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)

        # Tight layout
        plt.tight_layout()

    def render_step_by_step(self, algorithm: 'FortunesAlgorithm', sites: List[Point],
                           save_dir: Optional[str] = None) -> List[plt.Figure]:
        """
        Render the algorithm step by step (for animations/debugging).

        Args:
            algorithm: The Fortune's algorithm instance
            sites: List of sites
            save_dir: Optional directory to save step images

        Returns:
            List of figures for each step
        """
        algorithm.enable_debug_mode()
        figures = []

        # TODO: Implement step-by-step visualization
        # This would involve modifying the algorithm to pause at each step
        # and capture the current state of beach line and events

        return figures

    def create_animation(self, algorithm: 'FortunesAlgorithm', sites: List[Point],
                        filename: str = "voronoi_animation.gif") -> None:
        """
        Create an animated GIF of the algorithm execution.

        Args:
            algorithm: The Fortune's algorithm instance
            sites: List of sites
            filename: Output filename for the animation
        """
        # TODO: Implement animation creation using matplotlib.animation
        pass

    def set_style(self, style_name: str) -> None:
        """
        Apply a predefined style.

        Args:
            style_name: Name of the style ('minimal', 'scientific', 'artistic')
        """
        if style_name == 'minimal':
            self.show_vertices = False
            self.show_grid = False
            self.show_bounding_box = False
            self.edge_width = 0.8
            self.site_size = 30
            self.face_alpha = 0.3

        elif style_name == 'scientific':
            self.show_vertices = True
            self.show_grid = True
            self.show_bounding_box = True
            self.edge_width = 1.0
            self.site_size = 40
            self.face_alpha = 0.2
            self.high_quality = False

        elif style_name == 'artistic':
            self.show_vertices = False
            self.show_grid = False
            self.show_bounding_box = False
            self.edge_width = 2.0
            self.site_size = 60
            self.face_alpha = 0.8
            self.gradient_faces = True

    def export_svg(self, diagram: VoronoiDiagram, filename: str) -> None:
        """
        Export the diagram as SVG for scalable graphics.

        Args:
            diagram: The Voronoi diagram
            filename: Output SVG filename
        """
        fig = self.render(diagram, show=False)
        fig.savefig(filename, format='svg', bbox_inches='tight')
        plt.close(fig)