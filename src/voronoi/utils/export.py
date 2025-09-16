"""Export utilities for Voronoi diagrams in various formats."""

import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, List
import csv

from ..data_structures.voronoi_diagram import VoronoiDiagram
from ..data_structures.point import Point


class DiagramExporter:
    """Export Voronoi diagrams to various formats."""

    def __init__(self):
        pass

    def export_to_json(self, diagram: VoronoiDiagram, filename: str) -> None:
        """
        Export diagram to JSON format.

        Args:
            diagram: The Voronoi diagram to export
            filename: Output filename
        """
        data = self._diagram_to_dict(diagram)

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def export_to_svg(self, diagram: VoronoiDiagram, filename: str,
                      width: int = 800, height: int = 600) -> None:
        """
        Export diagram to SVG format.

        Args:
            diagram: The Voronoi diagram to export
            filename: Output filename
            width: SVG canvas width
            height: SVG canvas height
        """
        # Create SVG root
        svg = ET.Element('svg')
        svg.set('width', str(width))
        svg.set('height', str(height))
        svg.set('xmlns', 'http://www.w3.org/2000/svg')

        # Get diagram bounds
        bounds = diagram.bounds()
        if bounds:
            min_x, min_y, max_x, max_y = bounds
            diagram_width = max_x - min_x
            diagram_height = max_y - min_y

            # Calculate scale factors
            margin = 20
            scale_x = (width - 2 * margin) / diagram_width if diagram_width > 0 else 1
            scale_y = (height - 2 * margin) / diagram_height if diagram_height > 0 else 1
            scale = min(scale_x, scale_y)

            def transform_point(x: float, y: float) -> tuple[float, float]:
                svg_x = margin + (x - min_x) * scale
                svg_y = height - (margin + (y - min_y) * scale)  # Flip Y axis
                return svg_x, svg_y

            # Add background
            bg = ET.SubElement(svg, 'rect')
            bg.set('width', str(width))
            bg.set('height', str(height))
            bg.set('fill', '#f8f8f8')

            # Draw faces
            for i, face in enumerate(diagram.faces):
                if face.is_unbounded or not face.site:
                    continue

                boundary_points = face.get_boundary_points()
                if len(boundary_points) < 3:
                    continue

                # Create polygon
                points_str = ' '.join(f'{x},{y}' for x, y in
                                    (transform_point(p.x, p.y) for p in boundary_points))

                polygon = ET.SubElement(svg, 'polygon')
                polygon.set('points', points_str)
                polygon.set('fill', f'hsl({(i * 137.5) % 360}, 70%, 85%)')
                polygon.set('stroke', '#666')
                polygon.set('stroke-width', '0.5')

            # Draw edges
            for edge in diagram.edges:
                if not edge.is_complete:
                    continue

                start = edge.half_edge1.origin
                end = edge.half_edge1.destination

                if start and end:
                    start_x, start_y = transform_point(start.x, start.y)
                    end_x, end_y = transform_point(end.x, end.y)

                    line = ET.SubElement(svg, 'line')
                    line.set('x1', str(start_x))
                    line.set('y1', str(start_y))
                    line.set('x2', str(end_x))
                    line.set('y2', str(end_y))
                    line.set('stroke', '#333')
                    line.set('stroke-width', '1')

            # Draw vertices
            for vertex in diagram.vertices:
                x, y = transform_point(vertex.x, vertex.y)
                circle = ET.SubElement(svg, 'circle')
                circle.set('cx', str(x))
                circle.set('cy', str(y))
                circle.set('r', '2')
                circle.set('fill', '#0066cc')

            # Draw sites
            for site in diagram.sites:
                x, y = transform_point(site.x, site.y)
                circle = ET.SubElement(svg, 'circle')
                circle.set('cx', str(x))
                circle.set('cy', str(y))
                circle.set('r', '3')
                circle.set('fill', '#cc0000')

        # Write to file
        tree = ET.ElementTree(svg)
        tree.write(filename, encoding='utf-8', xml_declaration=True)

    def export_to_csv(self, diagram: VoronoiDiagram, filename: str) -> None:
        """
        Export diagram data to CSV format.

        Args:
            diagram: The Voronoi diagram to export
            filename: Output filename
        """
        with open(filename, 'w', newline='') as csvfile:
            # Export sites
            writer = csv.writer(csvfile)
            writer.writerow(['Type', 'ID', 'X', 'Y', 'Extra'])

            # Sites
            for i, site in enumerate(diagram.sites):
                writer.writerow(['site', i, site.x, site.y, ''])

            # Vertices
            for i, vertex in enumerate(diagram.vertices):
                writer.writerow(['vertex', i, vertex.x, vertex.y, f'degree={vertex.degree}'])

            # Edges (endpoints)
            for i, edge in enumerate(diagram.edges):
                if edge.is_complete:
                    start = edge.half_edge1.origin
                    end = edge.half_edge1.destination
                    if start and end:
                        writer.writerow(['edge_start', i, start.x, start.y, f'edge_id={i}'])
                        writer.writerow(['edge_end', i, end.x, end.y, f'edge_id={i}'])

    def export_to_obj(self, diagram: VoronoiDiagram, filename: str, extrude_height: float = 1.0) -> None:
        """
        Export diagram as 3D OBJ file with extruded cells.

        Args:
            diagram: The Voronoi diagram to export
            filename: Output filename
            extrude_height: Height to extrude the cells
        """
        with open(filename, 'w') as f:
            f.write("# Voronoi Diagram OBJ Export\n")
            f.write(f"# Generated from {len(diagram.sites)} sites\n\n")

            vertex_index = 1
            face_vertices = []

            # Export each cell as an extruded polygon
            for face in diagram.faces:
                if face.is_unbounded or not face.site:
                    continue

                boundary_points = face.get_boundary_points()
                if len(boundary_points) < 3:
                    continue

                cell_vertices = []

                # Bottom vertices
                for point in boundary_points:
                    f.write(f"v {point.x} {point.y} 0.0\n")
                    cell_vertices.append(vertex_index)
                    vertex_index += 1

                # Top vertices
                for point in boundary_points:
                    f.write(f"v {point.x} {point.y} {extrude_height}\n")
                    cell_vertices.append(vertex_index)
                    vertex_index += 1

                face_vertices.append(cell_vertices)

            f.write("\n")

            # Export faces
            for vertices in face_vertices:
                n = len(vertices) // 2

                # Bottom face (reverse order for correct normal)
                bottom_indices = vertices[:n]
                f.write(f"f {' '.join(str(i) for i in reversed(bottom_indices))}\n")

                # Top face
                top_indices = vertices[n:]
                f.write(f"f {' '.join(str(i) for i in top_indices)}\n")

                # Side faces
                for i in range(n):
                    next_i = (i + 1) % n
                    # Quad face: bottom[i], bottom[next_i], top[next_i], top[i]
                    f.write(f"f {vertices[i]} {vertices[next_i]} {vertices[n + next_i]} {vertices[n + i]}\n")

            f.write("\n")

    def export_statistics(self, diagram: VoronoiDiagram, analyzer, filename: str) -> None:
        """
        Export diagram statistics to JSON.

        Args:
            diagram: The Voronoi diagram
            analyzer: VoronoiAnalyzer instance
            filename: Output filename
        """
        stats = analyzer.analyze_diagram(diagram)
        patterns = analyzer.detect_patterns(diagram)

        data = {
            'basic_statistics': {
                'num_sites': stats.num_sites,
                'num_vertices': stats.num_vertices,
                'num_edges': stats.num_edges,
                'num_faces': stats.num_faces,
                'diagram_area': stats.diagram_area,
                'diagram_perimeter': stats.diagram_perimeter
            },
            'cell_statistics': {
                'avg_cell_area': stats.avg_cell_area,
                'cell_area_variance': stats.cell_area_variance,
                'min_cell_area': stats.min_cell_area,
                'max_cell_area': stats.max_cell_area
            },
            'vertex_statistics': {
                'avg_vertex_degree': stats.avg_vertex_degree,
                'vertex_degree_variance': stats.vertex_degree_variance,
                'min_vertex_degree': stats.min_vertex_degree,
                'max_vertex_degree': stats.max_vertex_degree
            },
            'edge_statistics': {
                'avg_edge_length': stats.avg_edge_length,
                'edge_length_variance': stats.edge_length_variance,
                'min_edge_length': stats.min_edge_length,
                'max_edge_length': stats.max_edge_length
            },
            'quality_metrics': {
                'regularity_index': stats.regularity_index,
                'uniformity_index': stats.uniformity_index,
                'convexity_deficiency': stats.convexity_deficiency
            },
            'patterns': patterns
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def _diagram_to_dict(self, diagram: VoronoiDiagram) -> Dict[str, Any]:
        """Convert diagram to dictionary representation."""
        return {
            'sites': [{'x': site.x, 'y': site.y} for site in diagram.sites],
            'vertices': [
                {'id': i, 'x': vertex.x, 'y': vertex.y, 'degree': vertex.degree}
                for i, vertex in enumerate(diagram.vertices)
            ],
            'edges': [
                {
                    'id': i,
                    'start': self._vertex_to_id(edge.half_edge1.origin, diagram.vertices) if edge.half_edge1.origin else None,
                    'end': self._vertex_to_id(edge.half_edge1.destination, diagram.vertices) if edge.half_edge1.destination else None,
                    'site1': {'x': edge.site1.x, 'y': edge.site1.y},
                    'site2': {'x': edge.site2.x, 'y': edge.site2.y},
                    'is_complete': edge.is_complete
                }
                for i, edge in enumerate(diagram.edges)
            ],
            'faces': [
                {
                    'id': i,
                    'site': {'x': face.site.x, 'y': face.site.y} if face.site else None,
                    'is_unbounded': face.is_unbounded,
                    'boundary_points': [
                        {'x': p.x, 'y': p.y} for p in face.get_boundary_points()
                    ] if not face.is_unbounded else [],
                    'area': face.area() if not face.is_unbounded else None
                }
                for i, face in enumerate(diagram.faces)
            ],
            'bounding_box': {
                'min_x': diagram.bounding_box[0],
                'min_y': diagram.bounding_box[1],
                'max_x': diagram.bounding_box[2],
                'max_y': diagram.bounding_box[3]
            },
            'statistics': diagram.get_statistics()
        }

    def _vertex_to_id(self, vertex, vertices_list: List) -> Optional[int]:
        """Get vertex ID in the vertices list."""
        if vertex is None:
            return None
        try:
            return vertices_list.index(vertex)
        except ValueError:
            return None

    def import_from_json(self, filename: str) -> List[Point]:
        """
        Import sites from JSON file.

        Args:
            filename: JSON file to import

        Returns:
            List of site points
        """
        with open(filename, 'r') as f:
            data = json.load(f)

        sites = []
        if 'sites' in data:
            for site_data in data['sites']:
                sites.append(Point(site_data['x'], site_data['y']))

        return sites

    def import_from_csv(self, filename: str) -> List[Point]:
        """
        Import sites from CSV file.

        Args:
            filename: CSV file to import (expects 'Type', 'X', 'Y' columns)

        Returns:
            List of site points
        """
        sites = []

        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('Type') == 'site':
                    x = float(row['X'])
                    y = float(row['Y'])
                    sites.append(Point(x, y))

        return sites