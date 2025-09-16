"""Color schemes and palettes for Voronoi diagram visualization."""

import math
import random
from typing import List, Tuple, Dict, Optional
from enum import Enum


class ColorScheme(Enum):
    """Predefined color schemes for Voronoi diagram visualization."""
    DEFAULT = "default"
    PASTEL = "pastel"
    VIBRANT = "vibrant"
    OCEAN = "ocean"
    SUNSET = "sunset"
    FOREST = "forest"
    RAINBOW = "rainbow"
    MONOCHROME = "monochrome"
    NEON = "neon"


class ColorPalette:
    """Advanced color palette generator for beautiful Voronoi visualizations."""

    def __init__(self, scheme: ColorScheme = ColorScheme.DEFAULT):
        self.scheme = scheme
        self._color_cache: Dict[int, Tuple[int, int, int]] = {}
        self._site_colors: Dict[Tuple[float, float], Tuple[int, int, int]] = {}

    def get_site_color(self, site_index: int, total_sites: int) -> Tuple[int, int, int]:
        """
        Get a color for a specific site using the selected color scheme.

        Args:
            site_index: Index of the site (0-based)
            total_sites: Total number of sites

        Returns:
            RGB color tuple (0-255 range)
        """
        if site_index in self._color_cache:
            return self._color_cache[site_index]

        color = self._generate_color(site_index, total_sites)
        self._color_cache[site_index] = color
        return color

    def get_site_color_by_position(self, x: float, y: float, bounds: Tuple[float, float, float, float]) -> Tuple[int, int, int]:
        """
        Generate a color based on spatial position of the site.

        Args:
            x, y: Site coordinates
            bounds: (min_x, min_y, max_x, max_y) of the diagram

        Returns:
            RGB color tuple
        """
        key = (x, y)
        if key in self._site_colors:
            return self._site_colors[key]

        min_x, min_y, max_x, max_y = bounds
        width = max_x - min_x
        height = max_y - min_y

        if width <= 0 or height <= 0:
            return (128, 128, 128)

        # Normalize coordinates to [0, 1]
        norm_x = (x - min_x) / width
        norm_y = (y - min_y) / height

        color = self._generate_position_based_color(norm_x, norm_y)
        self._site_colors[key] = color
        return color

    def _generate_color(self, index: int, total: int) -> Tuple[int, int, int]:
        """Generate color based on scheme and index."""
        if self.scheme == ColorScheme.DEFAULT:
            return self._default_colors(index, total)
        elif self.scheme == ColorScheme.PASTEL:
            return self._pastel_colors(index, total)
        elif self.scheme == ColorScheme.VIBRANT:
            return self._vibrant_colors(index, total)
        elif self.scheme == ColorScheme.OCEAN:
            return self._ocean_colors(index, total)
        elif self.scheme == ColorScheme.SUNSET:
            return self._sunset_colors(index, total)
        elif self.scheme == ColorScheme.FOREST:
            return self._forest_colors(index, total)
        elif self.scheme == ColorScheme.RAINBOW:
            return self._rainbow_colors(index, total)
        elif self.scheme == ColorScheme.MONOCHROME:
            return self._monochrome_colors(index, total)
        elif self.scheme == ColorScheme.NEON:
            return self._neon_colors(index, total)
        else:
            return self._default_colors(index, total)

    def _generate_position_based_color(self, norm_x: float, norm_y: float) -> Tuple[int, int, int]:
        """Generate color based on normalized position."""
        if self.scheme == ColorScheme.OCEAN:
            # Blue to teal gradient
            r = int(50 + norm_x * 100)
            g = int(100 + norm_y * 155)
            b = int(200 + norm_x * 55)
            return (r, g, b)
        elif self.scheme == ColorScheme.SUNSET:
            # Orange to purple gradient
            r = int(255 - norm_y * 100)
            g = int(100 + norm_x * 100)
            b = int(50 + norm_y * 200)
            return (r, g, b)
        elif self.scheme == ColorScheme.FOREST:
            # Green variations
            r = int(50 + norm_x * 100)
            g = int(150 + norm_y * 105)
            b = int(50 + norm_x * 50)
            return (r, g, b)
        else:
            return self._hsl_to_rgb(norm_x, 0.7, 0.5 + norm_y * 0.3)

    def _default_colors(self, index: int, total: int) -> Tuple[int, int, int]:
        """Default color scheme with good contrast."""
        hue = (index * 137.5) % 360  # Golden angle for good distribution
        return self._hsl_to_rgb(hue / 360, 0.7, 0.6)

    def _pastel_colors(self, index: int, total: int) -> Tuple[int, int, int]:
        """Soft pastel colors."""
        hue = (index * 137.5) % 360
        return self._hsl_to_rgb(hue / 360, 0.4, 0.8)

    def _vibrant_colors(self, index: int, total: int) -> Tuple[int, int, int]:
        """High saturation vibrant colors."""
        hue = (index * 137.5) % 360
        return self._hsl_to_rgb(hue / 360, 0.9, 0.5)

    def _ocean_colors(self, index: int, total: int) -> Tuple[int, int, int]:
        """Ocean-inspired blue and teal colors."""
        base_hues = [180, 200, 220, 160, 240]
        hue = base_hues[index % len(base_hues)]
        variation = (index // len(base_hues)) * 10
        final_hue = (hue + variation) % 360
        return self._hsl_to_rgb(final_hue / 360, 0.7, 0.6)

    def _sunset_colors(self, index: int, total: int) -> Tuple[int, int, int]:
        """Sunset-inspired warm colors."""
        base_hues = [0, 15, 30, 45, 300, 315, 330]
        hue = base_hues[index % len(base_hues)]
        variation = (index // len(base_hues)) * 5
        final_hue = (hue + variation) % 360
        return self._hsl_to_rgb(final_hue / 360, 0.8, 0.6)

    def _forest_colors(self, index: int, total: int) -> Tuple[int, int, int]:
        """Forest-inspired green colors."""
        base_hues = [80, 100, 120, 140, 60]
        hue = base_hues[index % len(base_hues)]
        variation = (index // len(base_hues)) * 8
        final_hue = (hue + variation) % 360
        return self._hsl_to_rgb(final_hue / 360, 0.6, 0.5)

    def _rainbow_colors(self, index: int, total: int) -> Tuple[int, int, int]:
        """Full spectrum rainbow colors."""
        if total <= 1:
            return (255, 0, 0)
        hue = (index / total) * 360
        return self._hsl_to_rgb(hue / 360, 0.8, 0.6)

    def _monochrome_colors(self, index: int, total: int) -> Tuple[int, int, int]:
        """Grayscale colors."""
        if total <= 1:
            return (128, 128, 128)
        value = int(50 + (index / (total - 1)) * 150)
        return (value, value, value)

    def _neon_colors(self, index: int, total: int) -> Tuple[int, int, int]:
        """Bright neon colors."""
        hue = (index * 137.5) % 360
        return self._hsl_to_rgb(hue / 360, 1.0, 0.5)

    @staticmethod
    def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
        """
        Convert HSL to RGB.

        Args:
            h: Hue [0, 1]
            s: Saturation [0, 1]
            l: Lightness [0, 1]

        Returns:
            RGB tuple (0-255)
        """
        if s == 0:
            r = g = b = l  # Achromatic
        else:
            def hue_to_rgb(p: float, q: float, t: float) -> float:
                if t < 0:
                    t += 1
                if t > 1:
                    t -= 1
                if t < 1/6:
                    return p + (q - p) * 6 * t
                if t < 1/2:
                    return q
                if t < 2/3:
                    return p + (q - p) * (2/3 - t) * 6
                return p

            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q

            r = hue_to_rgb(p, q, h + 1/3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1/3)

        return (int(r * 255), int(g * 255), int(b * 255))

    def clear_cache(self) -> None:
        """Clear the color cache."""
        self._color_cache.clear()
        self._site_colors.clear()

    def get_complementary_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get complementary colors for UI elements."""
        if self.scheme == ColorScheme.OCEAN:
            return {
                'background': (10, 25, 40),
                'grid': (70, 120, 150),
                'text': (200, 220, 240),
                'highlight': (255, 200, 100)
            }
        elif self.scheme == ColorScheme.SUNSET:
            return {
                'background': (40, 20, 30),
                'grid': (150, 100, 80),
                'text': (255, 240, 200),
                'highlight': (100, 200, 255)
            }
        elif self.scheme == ColorScheme.FOREST:
            return {
                'background': (20, 30, 15),
                'grid': (100, 150, 80),
                'text': (220, 255, 200),
                'highlight': (255, 150, 100)
            }
        elif self.scheme == ColorScheme.MONOCHROME:
            return {
                'background': (20, 20, 20),
                'grid': (100, 100, 100),
                'text': (220, 220, 220),
                'highlight': (255, 255, 255)
            }
        else:  # Default and others
            return {
                'background': (30, 30, 30),
                'grid': (100, 100, 100),
                'text': (220, 220, 220),
                'highlight': (255, 200, 100)
            }