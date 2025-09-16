"""Event queue implementation for Fortune's sweep line algorithm."""

import heapq
from typing import List, Optional, Union, Set
from abc import ABC, abstractmethod
from .point import Point


class Event(ABC):
    """Abstract base class for events in Fortune's algorithm."""

    def __init__(self, point: Point, priority: float):
        self.point = point
        self.priority = priority  # y-coordinate for site events, y-coordinate of circle event
        self.is_valid = True

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __lt__(self, other: 'Event') -> bool:
        """
        Comparison for heap ordering.
        Higher y-coordinates (processed first) have lower priority values.
        For ties, use x-coordinate as tiebreaker.
        """
        if abs(self.priority - other.priority) < 1e-10:
            return self.point.x < other.point.x
        return self.priority > other.priority  # Max heap behavior

    def __eq__(self, other: 'Event') -> bool:
        if not isinstance(other, Event):
            return False
        return (abs(self.priority - other.priority) < 1e-10 and
                self.point == other.point)

    def invalidate(self) -> None:
        """Mark this event as invalid (to be ignored during processing)."""
        self.is_valid = False


class SiteEvent(Event):
    """Event representing the processing of a new site point."""

    def __init__(self, site: Point):
        # For site events, priority is the y-coordinate (higher y processed first)
        super().__init__(site, site.y)
        self.site = site

    def __str__(self) -> str:
        return f"SiteEvent({self.site})"

    def __repr__(self) -> str:
        return f"SiteEvent(site={self.site}, priority={self.priority})"


class CircleEvent(Event):
    """Event representing the disappearance of an arc from the beach line."""

    def __init__(self, center: Point, radius: float, disappearing_arc: 'Arc'):
        # For circle events, priority is the bottom of the circle (y - radius)
        super().__init__(center, center.y - radius)
        self.center = center
        self.radius = radius
        self.disappearing_arc = disappearing_arc

        # The y-coordinate where the circle intersects the sweep line
        self.sweep_y = center.y - radius

    def __str__(self) -> str:
        return f"CircleEvent(center={self.center}, radius={self.radius:.3f})"

    def __repr__(self) -> str:
        return (f"CircleEvent(center={self.center}, radius={self.radius:.3f}, "
                f"priority={self.priority:.3f})")

    @property
    def vertex_point(self) -> Point:
        """The point where the Voronoi vertex will be created."""
        return self.center


class EventQueue:
    """
    Priority queue for events in Fortune's sweep line algorithm.
    Maintains events sorted by y-coordinate (descending).
    """

    def __init__(self):
        self._heap: List[Event] = []
        self._entry_count = 0

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        """Check if the event queue is empty."""
        return len(self._heap) == 0

    def add_site_event(self, site: Point) -> SiteEvent:
        """Add a site event to the queue."""
        event = SiteEvent(site)
        heapq.heappush(self._heap, event)
        self._entry_count += 1
        return event

    def add_circle_event(self, center: Point, radius: float, arc: 'Arc') -> CircleEvent:
        """Add a circle event to the queue."""
        event = CircleEvent(center, radius, arc)
        heapq.heappush(self._heap, event)
        self._entry_count += 1

        # Associate the event with the arc for easy invalidation
        arc.circle_event = event
        return event

    def pop_event(self) -> Optional[Event]:
        """
        Remove and return the next event to process.
        Automatically skips invalid events.
        """
        while self._heap:
            event = heapq.heappop(self._heap)
            if event.is_valid:
                return event
            # Skip invalid events
        return None

    def peek_event(self) -> Optional[Event]:
        """
        Look at the next event without removing it.
        Returns None if no valid events remain.
        """
        while self._heap and not self._heap[0].is_valid:
            heapq.heappop(self._heap)  # Remove invalid events

        return self._heap[0] if self._heap else None

    def add_sites(self, sites: List[Point]) -> None:
        """Add multiple site events at once."""
        for site in sites:
            self.add_site_event(site)

    def invalidate_circle_events_for_arc(self, arc: 'Arc') -> None:
        """Invalidate any circle events associated with the given arc."""
        if hasattr(arc, 'circle_event') and arc.circle_event:
            arc.circle_event.invalidate()
            arc.circle_event = None

    def get_all_events(self) -> List[Event]:
        """Get all valid events (for debugging/visualization)."""
        return [event for event in self._heap if event.is_valid]

    def clear(self) -> None:
        """Remove all events from the queue."""
        self._heap.clear()
        self._entry_count = 0

    @property
    def total_events_added(self) -> int:
        """Get the total number of events added to the queue."""
        return self._entry_count