"""CVAT XML annotation parser for Frank Sign project.

This module parses CVAT 1.1 XML format annotations and extracts:
- Image metadata
- Polygon annotations (ear contour, Frank Sign region)
- Polyline annotations (Frank Sign line)
- Point annotations (anatomical landmarks)
- Label attributes

Example:
    >>> parser = CVATParser("annotations.xml")
    >>> data = parser.parse()
    >>> for image in data.images:
    ...     print(image.name, len(image.annotations))
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

from lxml import etree
import numpy as np


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Point:
    """A single 2D point."""
    x: float
    y: float
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass  
class PointAnnotation:
    """Point-type annotation (anatomical landmarks)."""
    label: str
    point: Point
    attributes: Dict[str, str] = field(default_factory=dict)
    z_order: int = 0


@dataclass
class PolylineAnnotation:
    """Polyline annotation (Frank Sign line)."""
    label: str
    points: List[Point]
    attributes: Dict[str, str] = field(default_factory=dict)
    z_order: int = 0
    
    def to_array(self) -> np.ndarray:
        """Convert points to numpy array of shape (N, 2)."""
        return np.array([[p.x, p.y] for p in self.points])
    
    @property
    def num_points(self) -> int:
        return len(self.points)


@dataclass
class PolygonAnnotation:
    """Polygon annotation (ear contour, Frank Sign region)."""
    label: str
    points: List[Point]
    attributes: Dict[str, str] = field(default_factory=dict)
    z_order: int = 0
    
    def to_array(self) -> np.ndarray:
        """Convert points to numpy array of shape (N, 2)."""
        return np.array([[p.x, p.y] for p in self.points])
    
    @property
    def num_points(self) -> int:
        return len(self.points)


@dataclass
class ImageAnnotations:
    """All annotations for a single image."""
    id: int
    name: str
    width: int
    height: int
    subset: str = "default"
    task_id: Optional[int] = None
    
    # Annotations grouped by type
    points: List[PointAnnotation] = field(default_factory=list)
    polylines: List[PolylineAnnotation] = field(default_factory=list)
    polygons: List[PolygonAnnotation] = field(default_factory=list)
    
    @property
    def all_annotations(self) -> List[Union[PointAnnotation, PolylineAnnotation, PolygonAnnotation]]:
        """Return all annotations."""
        return self.points + self.polylines + self.polygons
    
    @property
    def has_frank_sign(self) -> bool:
        """Check if Frank Sign line is annotated and present."""
        for polyline in self.polylines:
            if polyline.label == "franks_sign_line":
                presence = polyline.attributes.get("presence", "present")
                return presence == "present"
        return False
    
    def get_by_label(self, label: str) -> List[Union[PointAnnotation, PolylineAnnotation, PolygonAnnotation]]:
        """Get all annotations with a specific label."""
        return [a for a in self.all_annotations if a.label == label]


@dataclass
class LabelDefinition:
    """Label schema definition from CVAT project."""
    name: str
    color: str
    type: str  # polygon, polyline, points
    attributes: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class CVATProject:
    """Parsed CVAT project data."""
    id: int
    name: str
    created: str
    updated: str
    labels: List[LabelDefinition]
    images: List[ImageAnnotations]
    version: str = "1.1"
    
    @property
    def num_images(self) -> int:
        return len(self.images)
    
    @property
    def num_with_frank_sign(self) -> int:
        return sum(1 for img in self.images if img.has_frank_sign)
    
    def get_image_by_name(self, name: str) -> Optional[ImageAnnotations]:
        """Find image by filename."""
        for img in self.images:
            if img.name == name:
                return img
        return None


# ============================================================
# PARSER CLASS
# ============================================================

class CVATParser:
    """Parser for CVAT 1.1 XML annotation format.
    
    Attributes:
        xml_path: Path to the annotations.xml file.
        
    Example:
        >>> parser = CVATParser("data/annotations/annotations.xml")
        >>> project = parser.parse()
        >>> print(f"Loaded {project.num_images} images")
        >>> print(f"Frank Sign present in {project.num_with_frank_sign} images")
    """
    
    def __init__(self, xml_path: Union[str, Path]):
        """Initialize parser with path to XML file.
        
        Args:
            xml_path: Path to CVAT annotations.xml file.
            
        Raises:
            FileNotFoundError: If XML file doesn't exist.
        """
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.xml_path}")
        
        self._tree: Optional[etree._ElementTree] = None
        self._root: Optional[etree._Element] = None
    
    def parse(self) -> CVATProject:
        """Parse the XML file and return structured data.
        
        Returns:
            CVATProject containing all parsed annotations.
            
        Raises:
            ValueError: If XML format is invalid.
        """
        self._tree = etree.parse(str(self.xml_path))
        self._root = self._tree.getroot()
        
        # Get version
        version_elem = self._root.find("version")
        version = version_elem.text if version_elem is not None else "1.1"
        
        # Parse project metadata
        meta = self._root.find("meta")
        if meta is None:
            raise ValueError("Missing <meta> element in XML")
        
        project_elem = meta.find("project")
        if project_elem is None:
            raise ValueError("Missing <project> element in XML")
        
        # Parse labels
        labels = self._parse_labels(project_elem)
        
        # Parse images
        images = self._parse_images()
        
        return CVATProject(
            id=int(project_elem.findtext("id", "0")),
            name=project_elem.findtext("name", "Unknown"),
            created=project_elem.findtext("created", ""),
            updated=project_elem.findtext("updated", ""),
            labels=labels,
            images=images,
            version=version,
        )
    
    def _parse_labels(self, project_elem: etree._Element) -> List[LabelDefinition]:
        """Parse label definitions from project metadata."""
        labels = []
        labels_elem = project_elem.find("labels")
        
        if labels_elem is None:
            return labels
        
        for label_elem in labels_elem.findall("label"):
            attrs = []
            attrs_elem = label_elem.find("attributes")
            if attrs_elem is not None:
                for attr_elem in attrs_elem.findall("attribute"):
                    attrs.append({
                        "name": attr_elem.findtext("name", ""),
                        "input_type": attr_elem.findtext("input_type", "text"),
                        "default_value": attr_elem.findtext("default_value", ""),
                        "values": attr_elem.findtext("values", ""),
                    })
            
            labels.append(LabelDefinition(
                name=label_elem.findtext("name", ""),
                color=label_elem.findtext("color", "#000000"),
                type=label_elem.findtext("type", "polygon"),
                attributes=attrs,
            ))
        
        return labels
    
    def _parse_images(self) -> List[ImageAnnotations]:
        """Parse all image annotations."""
        images = []
        
        for image_elem in self._root.findall("image"):
            img = ImageAnnotations(
                id=int(image_elem.get("id", 0)),
                name=image_elem.get("name", ""),
                width=int(image_elem.get("width", 0)),
                height=int(image_elem.get("height", 0)),
                subset=image_elem.get("subset", "default"),
                task_id=int(image_elem.get("task_id", 0)) if image_elem.get("task_id") else None,
            )
            
            # Parse points
            for points_elem in image_elem.findall("points"):
                img.points.append(self._parse_point(points_elem))
            
            # Parse polylines
            for polyline_elem in image_elem.findall("polyline"):
                img.polylines.append(self._parse_polyline(polyline_elem))
            
            # Parse polygons
            for polygon_elem in image_elem.findall("polygon"):
                img.polygons.append(self._parse_polygon(polygon_elem))
            
            images.append(img)
        
        return images
    
    def _parse_point(self, elem: etree._Element) -> PointAnnotation:
        """Parse a point annotation element.
        
        Note: Some point annotations in CVAT have multiple coordinates
        separated by semicolons (e.g., "1024.70;574.68"). In such cases,
        we parse all points but use only the first one.
        """
        points_str = elem.get("points", "0,0")
        
        # Handle multi-point format: "x1,y1;x2,y2;..."
        if ";" in points_str:
            # Split by semicolon and take first point
            first_point = points_str.split(";")[0]
            x, y = map(float, first_point.split(","))
        else:
            # Single point format: "x,y"
            x, y = map(float, points_str.split(","))
        
        return PointAnnotation(
            label=elem.get("label", ""),
            point=Point(x=x, y=y),
            attributes=self._parse_attributes(elem),
            z_order=int(elem.get("z_order", 0)),
        )
    
    def _parse_polyline(self, elem: etree._Element) -> PolylineAnnotation:
        """Parse a polyline annotation element."""
        points = self._parse_points_string(elem.get("points", ""))
        
        return PolylineAnnotation(
            label=elem.get("label", ""),
            points=points,
            attributes=self._parse_attributes(elem),
            z_order=int(elem.get("z_order", 0)),
        )
    
    def _parse_polygon(self, elem: etree._Element) -> PolygonAnnotation:
        """Parse a polygon annotation element."""
        points = self._parse_points_string(elem.get("points", ""))
        
        return PolygonAnnotation(
            label=elem.get("label", ""),
            points=points,
            attributes=self._parse_attributes(elem),
            z_order=int(elem.get("z_order", 0)),
        )
    
    def _parse_points_string(self, points_str: str) -> List[Point]:
        """Parse CVAT points string format: 'x1,y1;x2,y2;...'"""
        if not points_str:
            return []
        
        points = []
        for pair in points_str.split(";"):
            if "," in pair:
                x, y = map(float, pair.split(","))
                points.append(Point(x=x, y=y))
        
        return points
    
    def _parse_attributes(self, elem: etree._Element) -> Dict[str, str]:
        """Parse annotation attributes."""
        attrs = {}
        for attr_elem in elem.findall("attribute"):
            name = attr_elem.get("name", "")
            value = attr_elem.text or ""
            if name:
                attrs[name] = value
        return attrs


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def load_annotations(xml_path: Union[str, Path]) -> CVATProject:
    """Load CVAT annotations from XML file.
    
    Args:
        xml_path: Path to annotations.xml
        
    Returns:
        Parsed CVATProject object
        
    Example:
        >>> project = load_annotations("data/annotations/annotations.xml")
        >>> print(project.num_images)
    """
    parser = CVATParser(xml_path)
    return parser.parse()


def get_frank_sign_images(project: CVATProject) -> List[ImageAnnotations]:
    """Filter images that have Frank Sign present.
    
    Args:
        project: Parsed CVAT project
        
    Returns:
        List of images with Frank Sign annotated as present
    """
    return [img for img in project.images if img.has_frank_sign]


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
    else:
        xml_path = "data/annotations/annotations.xml"
    
    try:
        project = load_annotations(xml_path)
        print(f"✅ Loaded {project.num_images} images")
        print(f"📊 Frank Sign present in {project.num_with_frank_sign} images")
        print(f"🏷️  Labels defined: {len(project.labels)}")
        
        for label in project.labels:
            print(f"   - {label.name} ({label.type})")
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
