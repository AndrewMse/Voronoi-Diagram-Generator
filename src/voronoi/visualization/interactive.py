"""Interactive Voronoi diagram application using Pygame."""

import pygame
import sys
import math
import random
from typing import List, Tuple, Optional, Dict, Any
from ..data_structures.point import Point
from ..algorithms.fortune import FortunesAlgorithm
from ..data_structures.voronoi_diagram import VoronoiDiagram
from .colors import ColorPalette, ColorScheme
from ..config import get_config


class InteractiveVoronoiApp:
    """
    Interactive Voronoi diagram application with real-time updates.
    """

    def __init__(self, width: int = None, height: int = None):
        pygame.init()
        pygame.font.init()

        config = get_config()
        self.width = width or config.get('interactive.window_width', 1000)
        self.height = height or config.get('interactive.window_height', 700)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Advanced Voronoi Diagram Generator")

        # Colors and styling
        self.background_color = (20, 20, 30)
        self.site_color = (255, 100, 100)
        self.vertex_color = (100, 150, 255)
        self.edge_color = (200, 200, 200)
        self.grid_color = (50, 50, 60)
        self.text_color = (200, 200, 200)

        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

        # Application state
        self.sites: List[Point] = []
        self.diagram: Optional[VoronoiDiagram] = None
        self.color_palette = ColorPalette(ColorScheme.DEFAULT)

        # Visualization settings
        self.show_sites = True
        self.show_vertices = False
        self.show_edges = False
        self.show_faces = True
        self.show_grid = False
        self.show_stats = True
        self.animate_construction = False

        # Interaction state
        self.mouse_pos = (0, 0)
        self.dragging = False
        self.selected_site = None

        # Animation state
        self.animation_speed = 1.0
        self.is_animating = False

        # UI elements
        self.ui_height = 60
        self.buttons = self._create_buttons()

        # Performance tracking
        self.fps_clock = pygame.time.Clock()
        self.target_fps = 60

    def run(self) -> None:
        """Main application loop."""
        running = True

        # Generate some initial random sites across the full window
        config = get_config()
        initial_count = config.get('sites.default_count', 25)
        self._generate_random_sites(initial_count)
        self._update_diagram()

        while running:
            dt = self.fps_clock.tick(self.target_fps)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    self._handle_event(event)

            # Update
            self._update(dt)

            # Render
            self._render()

            pygame.display.flip()

        pygame.quit()
        sys.exit()

    def _handle_event(self, event: pygame.event.Event) -> None:
        """Handle pygame events."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            self._handle_mouse_down(event)
        elif event.type == pygame.MOUSEBUTTONUP:
            self._handle_mouse_up(event)
        elif event.type == pygame.MOUSEMOTION:
            self._handle_mouse_motion(event)
        elif event.type == pygame.KEYDOWN:
            self._handle_key_down(event)

    def _handle_mouse_down(self, event: pygame.event.Event) -> None:
        """Handle mouse button down events."""
        x, y = event.pos

        # Check if clicking in UI area
        if y < self.ui_height:
            self._handle_ui_click(x, y)
            return

        # Convert screen coordinates to world coordinates
        world_pos = self._screen_to_world(x, y)

        if event.button == 1:  # Left click
            if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                # Remove nearest site
                self._remove_nearest_site(world_pos)
            else:
                # Add new site
                self._add_site(world_pos)

        elif event.button == 3:  # Right click
            # Start dragging nearest site
            nearest_site = self._find_nearest_site(world_pos)
            if nearest_site and world_pos.distance_to(nearest_site) < 50:
                self.selected_site = nearest_site
                self.dragging = True

    def _handle_mouse_up(self, event: pygame.event.Event) -> None:
        """Handle mouse button up events."""
        if event.button == 3:  # Right click
            self.dragging = False
            self.selected_site = None

    def _handle_mouse_motion(self, event: pygame.event.Event) -> None:
        """Handle mouse motion events."""
        self.mouse_pos = event.pos

        if self.dragging and self.selected_site:
            world_pos = self._screen_to_world(*event.pos)
            # Update site position
            self.selected_site.x = world_pos.x
            self.selected_site.y = world_pos.y
            self._update_diagram()

    def _handle_key_down(self, event: pygame.event.Event) -> None:
        """Handle key press events."""
        if event.key == pygame.K_SPACE:
            self._generate_random_sites(10)
            self._update_diagram()
        elif event.key == pygame.K_c:
            self.sites.clear()
            self.diagram = None
        elif event.key == pygame.K_s:
            self.show_sites = not self.show_sites
        elif event.key == pygame.K_v:
            self.show_vertices = not self.show_vertices
        elif event.key == pygame.K_e:
            self.show_edges = not self.show_edges
        elif event.key == pygame.K_f:
            self.show_faces = not self.show_faces
        elif event.key == pygame.K_g:
            self.show_grid = not self.show_grid
        elif event.key == pygame.K_TAB:
            self._cycle_color_scheme()
        elif event.key == pygame.K_r:
            self._generate_random_sites(random.randint(5, 50))
            self._update_diagram()

    def _handle_ui_click(self, x: int, y: int) -> None:
        """Handle clicks in the UI area."""
        for button in self.buttons:
            if button['rect'].collidepoint(x, y):
                button['action']()

    def _create_buttons(self) -> List[Dict[str, Any]]:
        """Create UI buttons."""
        buttons = []
        button_width = 100
        button_height = 30
        button_spacing = 10
        start_x = 10

        # Random sites button
        buttons.append({
            'rect': pygame.Rect(start_x, 15, button_width, button_height),
            'text': 'Random',
            'action': lambda: self._generate_random_sites_ui()
        })

        # Clear button
        buttons.append({
            'rect': pygame.Rect(start_x + button_width + button_spacing, 15, button_width, button_height),
            'text': 'Clear',
            'action': lambda: self._clear_sites()
        })

        # Color scheme button
        buttons.append({
            'rect': pygame.Rect(start_x + 2 * (button_width + button_spacing), 15, button_width, button_height),
            'text': 'Colors',
            'action': lambda: self._cycle_color_scheme()
        })

        return buttons

    def _update(self, dt: float) -> None:
        """Update application state."""
        # Animation updates would go here
        pass

    def _render(self) -> None:
        """Render the application."""
        # Clear screen
        self.screen.fill(self.background_color)

        # Render grid
        if self.show_grid:
            self._render_grid()

        # Render Voronoi diagram
        if self.diagram:
            if self.show_faces:
                self._render_faces()
            if self.show_edges:
                self._render_edges()
            if self.show_vertices:
                self._render_vertices()

        # Render sites
        if self.show_sites:
            self._render_sites()

        # Render UI
        self._render_ui()

        # Render stats
        if self.show_stats:
            self._render_stats()

    def _render_grid(self) -> None:
        """Render background grid."""
        grid_spacing = 50
        for x in range(0, self.width, grid_spacing):
            pygame.draw.line(self.screen, self.grid_color,
                           (x, self.ui_height), (x, self.height))
        for y in range(self.ui_height, self.height, grid_spacing):
            pygame.draw.line(self.screen, self.grid_color,
                           (0, y), (self.width, y))

    def _render_faces(self) -> None:
        """Render Voronoi faces."""
        if not self.diagram:
            return

        bounds = self.diagram.bounds()
        site_index = 0

        for face in self.diagram.faces:
            if face.is_unbounded or not face.site:
                continue

            boundary_points = face.get_boundary_points()
            if len(boundary_points) < 3:
                continue

            # Convert to screen coordinates
            screen_points = []
            for point in boundary_points:
                screen_pos = self._world_to_screen(point.x, point.y)
                screen_points.append(screen_pos)

            # Get color for this face
            color = self.color_palette.get_site_color(site_index, len(self.diagram.sites))

            # Draw filled polygon
            if len(screen_points) >= 3:
                pygame.draw.polygon(self.screen, color, screen_points)

            site_index += 1

    def _render_edges(self) -> None:
        """Render Voronoi edges."""
        if not self.diagram:
            return

        for edge in self.diagram.edges:
            if not edge.is_complete:
                continue

            start = edge.half_edge1.origin
            end = edge.half_edge1.destination

            if start and end:
                start_screen = self._world_to_screen(start.x, start.y)
                end_screen = self._world_to_screen(end.x, end.y)

                pygame.draw.line(self.screen, self.edge_color,
                               start_screen, end_screen, 2)

    def _render_vertices(self) -> None:
        """Render Voronoi vertices."""
        if not self.diagram:
            return

        for vertex in self.diagram.vertices:
            screen_pos = self._world_to_screen(vertex.x, vertex.y)
            pygame.draw.circle(self.screen, self.vertex_color, screen_pos, 4)
            pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, 4, 1)

    def _render_sites(self) -> None:
        """Render site points."""
        for site in self.sites:
            screen_pos = self._world_to_screen(site.x, site.y)

            # Highlight selected site
            if site == self.selected_site:
                pygame.draw.circle(self.screen, (255, 255, 100), screen_pos, 12)

            pygame.draw.circle(self.screen, self.site_color, screen_pos, 8)
            pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, 8, 2)

    def _render_ui(self) -> None:
        """Render UI elements."""
        # UI background
        pygame.draw.rect(self.screen, (40, 40, 50),
                        pygame.Rect(0, 0, self.width, self.ui_height))

        # Buttons
        for button in self.buttons:
            pygame.draw.rect(self.screen, (60, 60, 80), button['rect'])
            pygame.draw.rect(self.screen, (100, 100, 120), button['rect'], 2)

            # Button text
            text_surface = self.font_medium.render(button['text'], True, self.text_color)
            text_rect = text_surface.get_rect(center=button['rect'].center)
            self.screen.blit(text_surface, text_rect)

    def _render_stats(self) -> None:
        """Render statistics."""
        if not self.diagram:
            return

        stats = self.diagram.get_statistics()
        y_offset = self.ui_height + 10

        stats_text = [
            f"Sites: {stats['num_sites']}",
            f"Vertices: {stats['num_vertices']}",
            f"Edges: {stats['num_edges']}",
            f"FPS: {int(self.fps_clock.get_fps())}"
        ]

        for i, text in enumerate(stats_text):
            surface = self.font_small.render(text, True, self.text_color)
            self.screen.blit(surface, (self.width - 120, y_offset + i * 20))

        # Controls help
        help_text = [
            "Left click: Add site",
            "Shift+click: Remove site",
            "Right drag: Move site",
            "Space: Random sites",
            "C: Clear all",
            "S/V/E/F/G: Toggle display",
            "Tab: Change colors"
        ]

        for i, text in enumerate(help_text):
            surface = self.font_small.render(text, True, (150, 150, 150))
            self.screen.blit(surface, (10, self.height - 140 + i * 18))

    def _screen_to_world(self, screen_x: int, screen_y: int) -> Point:
        """Convert screen coordinates to world coordinates."""
        # Simple mapping - could be enhanced with zoom/pan
        world_y = screen_y - self.ui_height
        return Point(screen_x, world_y)

    def _world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int(world_x)
        screen_y = int(world_y + self.ui_height)
        return (screen_x, screen_y)

    def _add_site(self, position: Point) -> None:
        """Add a new site at the given position."""
        self.sites.append(position)
        self._update_diagram()

    def _remove_nearest_site(self, position: Point) -> None:
        """Remove the site nearest to the given position."""
        if not self.sites:
            return

        nearest = self._find_nearest_site(position)
        if nearest and position.distance_to(nearest) < 50:
            self.sites.remove(nearest)
            self._update_diagram()

    def _find_nearest_site(self, position: Point) -> Optional[Point]:
        """Find the site nearest to the given position."""
        if not self.sites:
            return None

        nearest = self.sites[0]
        min_distance = position.distance_to(nearest)

        for site in self.sites[1:]:
            distance = position.distance_to(site)
            if distance < min_distance:
                min_distance = distance
                nearest = site

        return nearest

    def _generate_random_sites(self, count: int) -> None:
        """Generate random sites."""
        self.sites.clear()
        margin = 50

        for _ in range(count):
            x = random.randint(margin, self.width - margin)
            y = random.randint(margin, self.height - self.ui_height - margin)
            self.sites.append(Point(x, y))

    def _generate_random_sites_ui(self) -> None:
        """Generate random sites from UI."""
        count = random.randint(8, 25)
        self._generate_random_sites(count)
        self._update_diagram()

    def _clear_sites(self) -> None:
        """Clear all sites."""
        self.sites.clear()
        self.diagram = None

    def _cycle_color_scheme(self) -> None:
        """Cycle through available color schemes."""
        schemes = list(ColorScheme)
        current_index = schemes.index(self.color_palette.scheme)
        next_index = (current_index + 1) % len(schemes)
        self.color_palette = ColorPalette(schemes[next_index])

    def _update_diagram(self) -> None:
        """Update the Voronoi diagram."""
        if len(self.sites) < 2:
            self.diagram = None
            return

        try:
            # Use window size for bounding box
            bounding_box = (0, self.ui_height, self.width, self.height)
            algorithm = FortunesAlgorithm(bounding_box)
            self.diagram = algorithm.generate_voronoi_diagram(self.sites)
        except Exception as e:
            print(f"Error generating diagram: {e}")
            self.diagram = None


def main():
    """Run the interactive Voronoi application."""
    app = InteractiveVoronoiApp()
    app.run()


if __name__ == "__main__":
    main()