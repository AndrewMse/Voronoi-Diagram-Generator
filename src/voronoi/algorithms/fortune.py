"""Fortune's sweep line algorithm for Voronoi diagram generation."""

from typing import List, Optional, Tuple, Set
import math
from ..data_structures.point import Point
from ..data_structures.vertex import Vertex
from ..data_structures.edge import Edge, HalfEdge
from ..data_structures.face import Face
from ..data_structures.voronoi_diagram import VoronoiDiagram
from ..data_structures.event_queue import EventQueue, SiteEvent, CircleEvent
from ..data_structures.beach_line import BeachLine, Arc
from .geometry_utils import GeometryUtils


class FortunesAlgorithm:
    """
    Implementation of Fortune's sweep line algorithm for generating Voronoi diagrams.

    This algorithm operates by sweeping a line from top to bottom, maintaining a beach line
    that represents the current boundary of the constructed Voronoi diagram.
    """

    def __init__(self, bounding_box: Optional[Tuple[float, float, float, float]] = None):
        """
        Initialize the Fortune's algorithm.

        Args:
            bounding_box: (min_x, min_y, max_x, max_y) for diagram bounds
        """
        self.bounding_box = bounding_box or (-1000, -1000, 1000, 1000)
        self.diagram: Optional[VoronoiDiagram] = None

        # Algorithm state
        self.event_queue: EventQueue = EventQueue()
        self.beach_line: BeachLine = BeachLine()
        self.sweep_y: float = float('inf')

        # For debugging and visualization
        self.debug_mode = False
        self.step_count = 0
        self.event_history: List[str] = []

    def generate_voronoi_diagram(self, sites: List[Point]) -> VoronoiDiagram:
        """
        Generate a Voronoi diagram for the given set of sites.

        Args:
            sites: List of site points

        Returns:
            The complete Voronoi diagram
        """
        if len(sites) < 2:
            raise ValueError("Need at least 2 sites to generate a Voronoi diagram")

        # Initialize the diagram
        self.diagram = VoronoiDiagram(self.bounding_box)

        # Add all sites to the diagram and event queue
        self._initialize_sites(sites)

        # Process all events
        self._process_events()

        # Finalize the diagram
        self.diagram.finalize_diagram()

        return self.diagram

    def _initialize_sites(self, sites: List[Point]) -> None:
        """Initialize the algorithm with the given sites."""
        # Remove duplicate sites
        unique_sites = list(set(sites))

        # Sort sites by y-coordinate (top to bottom), then by x-coordinate
        sorted_sites = sorted(unique_sites, key=lambda p: (-p.y, p.x))

        # Add sites to the diagram and event queue
        for site in sorted_sites:
            self.diagram.add_site(site)
            self.event_queue.add_site_event(site)

        if self.debug_mode:
            print(f"Initialized with {len(sorted_sites)} sites")

    def _process_events(self) -> None:
        """Main event processing loop."""
        while not self.event_queue.is_empty():
            event = self.event_queue.pop_event()
            if event is None:
                continue

            self.step_count += 1
            self.sweep_y = event.point.y

            if isinstance(event, SiteEvent):
                self._handle_site_event(event)
            elif isinstance(event, CircleEvent):
                self._handle_circle_event(event)

            if self.debug_mode:
                event_type = "Site" if isinstance(event, SiteEvent) else "Circle"
                self.event_history.append(f"Step {self.step_count}: {event_type} event at {event.point}")

    def _handle_site_event(self, event: SiteEvent) -> None:
        """
        Handle a site event by inserting the new site into the beach line.

        Args:
            event: The site event to process
        """
        site = event.site

        if self.beach_line.is_empty():
            # First site: create the initial arc
            self.beach_line.insert_first_arc(site)
            return

        # Find the arc directly above the new site
        arc_above = self.beach_line.find_arc_above(site.x, self.sweep_y)

        if arc_above is None:
            # Shouldn't happen, but handle gracefully
            return

        # Remove any circle event associated with the arc we're splitting
        self.event_queue.invalidate_circle_events_for_arc(arc_above)

        # Split the arc
        left_arc, middle_arc, right_arc = self.beach_line.split_arc(arc_above, site)

        # Create new edges (bisectors)
        left_edge = self.diagram.add_edge(arc_above.site, site)
        right_edge = self.diagram.add_edge(site, arc_above.site)

        # Connect arcs to edges
        left_arc.right_edge = left_edge.half_edge1
        middle_arc.left_edge = left_edge.half_edge2
        middle_arc.right_edge = right_edge.half_edge1
        right_arc.left_edge = right_edge.half_edge2

        # Set up edge-site relationships
        left_edge.half_edge1.left_site = arc_above.site
        left_edge.half_edge1.right_site = site
        left_edge.half_edge2.left_site = site
        left_edge.half_edge2.right_site = arc_above.site

        right_edge.half_edge1.left_site = site
        right_edge.half_edge1.right_site = arc_above.site
        right_edge.half_edge2.left_site = arc_above.site
        right_edge.half_edge2.right_site = site

        # Check for new circle events
        self._check_circle_event(left_arc)
        self._check_circle_event(right_arc)

    def _handle_circle_event(self, event: CircleEvent) -> None:
        """
        Handle a circle event by removing an arc from the beach line.

        Args:
            event: The circle event to process
        """
        disappearing_arc = event.disappearing_arc

        # Get the triple of arcs
        triple = self.beach_line.get_triple(disappearing_arc)
        if triple is None:
            return

        left_arc, middle_arc, right_arc = triple

        # Create a new vertex at the circle center
        vertex = self.diagram.add_vertex(event.center)

        # Remove circle events for the neighboring arcs
        self.event_queue.invalidate_circle_events_for_arc(left_arc)
        self.event_queue.invalidate_circle_events_for_arc(right_arc)

        # Complete the edges that end at this vertex
        if middle_arc.left_edge:
            self._complete_edge_at_vertex(middle_arc.left_edge, vertex)
        if middle_arc.right_edge:
            self._complete_edge_at_vertex(middle_arc.right_edge, vertex)

        # Remove the disappearing arc
        self.beach_line.remove_arc(middle_arc)

        # Create new edge between the remaining sites
        new_edge = self.diagram.add_edge(left_arc.site, right_arc.site)

        # Start the new edge at the vertex
        new_edge.set_vertex(vertex, is_start=True)

        # Update arc edge relationships
        left_arc.right_edge = new_edge.half_edge1
        right_arc.left_edge = new_edge.half_edge2

        # Set up edge-site relationships
        new_edge.half_edge1.left_site = left_arc.site
        new_edge.half_edge1.right_site = right_arc.site
        new_edge.half_edge2.left_site = right_arc.site
        new_edge.half_edge2.right_site = left_arc.site

        # Check for new circle events
        self._check_circle_event(left_arc)
        self._check_circle_event(right_arc)

    def _complete_edge_at_vertex(self, half_edge: HalfEdge, vertex: Vertex) -> None:
        """Complete a half-edge by setting its destination vertex."""
        if half_edge.destination is None:
            half_edge.destination = vertex
            vertex.add_incident_edge(half_edge)

        # Also complete the twin edge
        if half_edge.twin and half_edge.twin.origin is None:
            half_edge.twin.origin = vertex
            vertex.add_incident_edge(half_edge.twin)

    def _check_circle_event(self, arc: Arc) -> None:
        """
        Check if an arc forms a circle event with its neighbors.

        Args:
            arc: The arc to check
        """
        # Need three consecutive arcs to form a circle event
        if not arc.left_arc or not arc.right_arc:
            return

        left_site = arc.left_arc.site
        middle_site = arc.site
        right_site = arc.right_arc.site

        # Check if the three sites form a valid circle
        circumcenter = GeometryUtils.circumcenter(left_site, middle_site, right_site)
        if circumcenter is None:
            return  # Sites are collinear

        circumradius = circumcenter.distance_to(middle_site)

        # Check if the circle event is below the current sweep line
        circle_bottom = circumcenter.y - circumradius
        if circle_bottom >= self.sweep_y - 1e-10:
            return  # Circle event is above or at sweep line

        # Check the correct ordering of sites around the circle
        if not self._is_valid_circle_event(left_site, middle_site, right_site, circumcenter):
            return

        # Create the circle event
        self.event_queue.add_circle_event(circumcenter, circumradius, arc)

    def _is_valid_circle_event(self, left: Point, middle: Point, right: Point, center: Point) -> bool:
        """
        Validate that a circle event represents a valid configuration.

        Args:
            left, middle, right: The three sites
            center: The circumcenter

        Returns:
            True if this is a valid circle event
        """
        # Check that the sites are in the correct order around the circle
        # The middle site should disappear when the sweep line reaches the bottom of the circle

        # Vector from center to middle site
        to_middle = middle - center

        # Vectors to left and right sites
        to_left = left - center
        to_right = right - center

        # Check if middle is between left and right in the circular order
        # Using cross products to determine orientation
        cross_left = to_left.cross(to_middle)
        cross_right = to_middle.cross(to_right)

        # Both should have the same sign for valid ordering
        return cross_left * cross_right >= 0

    def get_algorithm_state(self) -> dict:
        """
        Get the current state of the algorithm (for debugging/visualization).

        Returns:
            Dictionary containing current algorithm state
        """
        beach_arcs = []
        if not self.beach_line.is_empty():
            current = self.beach_line.leftmost
            while current:
                beach_arcs.append({
                    'site': current.site.to_tuple(),
                    'id': current.id
                })
                current = current.right_arc

        return {
            'sweep_y': self.sweep_y,
            'step_count': self.step_count,
            'beach_line_arcs': beach_arcs,
            'events_remaining': len(self.event_queue),
            'diagram_stats': self.diagram.get_statistics() if self.diagram else {}
        }

    def enable_debug_mode(self) -> None:
        """Enable debug mode for detailed algorithm tracing."""
        self.debug_mode = True

    def disable_debug_mode(self) -> None:
        """Disable debug mode."""
        self.debug_mode = False

    def get_event_history(self) -> List[str]:
        """Get the history of processed events."""
        return self.event_history.copy()

    def reset(self) -> None:
        """Reset the algorithm state for a new computation."""
        self.event_queue.clear()
        self.beach_line = BeachLine()
        self.sweep_y = float('inf')
        self.step_count = 0
        self.event_history.clear()
        self.diagram = None