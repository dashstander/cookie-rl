from shapely.geometry import LineString, Point
from shapely.ops import unary_union, polygonize_full
from shapely import get_parts
import numpy as np


def nearest_point_on_boundary(boundary, angle, radius=1.0):
    """Find the nearest point on the actual boundary to the ideal circle point."""
    ideal_point = Point(radius * np.cos(angle), radius * np.sin(angle))
    nearest = boundary.interpolate(boundary.project(ideal_point))
    return (nearest.x, nearest.y)

def create_chord_from_angle_bias(boundary, center_angle, bias, radius=1.0):
    """
    Create a chord from a center angle and bias.
    
    Args:
        boundary: The circle boundary
        center_angle: The angle of the chord's center (0 to 2π)
        bias: How offset the chord is from center (-1 to 1)
              0 = chord passes through circle center
              ±1 = chord is tangent to a circle of radius 0 (essentially at the edge)
    
    Returns:
        LineString representing the chord
    """
    # The perpendicular direction to the chord
    perp_angle = center_angle + np.pi / 2
    
    # Distance from center (bias maps from [-1, 1] to [0, radius])
    # bias=0 means distance=0 (through center)
    # bias=±1 means distance=radius (tangent, but we'll clamp it slightly)
    distance_from_center = abs(bias) * radius * 0.99  # 0.99 to avoid true tangents
    
    # The center point of the chord (offset from origin)
    chord_center_x = distance_from_center * np.cos(perp_angle) * np.sign(bias)
    chord_center_y = distance_from_center * np.sin(perp_angle) * np.sign(bias)
    
    # Half-length of the chord (from Pythagorean theorem)
    # chord goes from circle at distance sqrt(r² - d²) from center
    half_length = np.sqrt(radius**2 - distance_from_center**2)
    
    # Endpoints of the chord
    angle1 = center_angle
    angle2 = center_angle + np.pi
    
    p1_x = chord_center_x + half_length * np.cos(angle1)
    p1_y = chord_center_y + half_length * np.sin(angle1)
    p2_x = chord_center_x + half_length * np.cos(angle2)
    p2_y = chord_center_y + half_length * np.sin(angle2)
    
    # Snap to actual boundary
    p1 = Point(p1_x, p1_y)
    p2 = Point(p2_x, p2_y)
    
    p1_snapped = boundary.interpolate(boundary.project(p1))
    p2_snapped = boundary.interpolate(boundary.project(p2))
    
    return LineString([(p1_snapped.x, p1_snapped.y), (p2_snapped.x, p2_snapped.y)])

# Create the circle
radius = 1.0
circle = Point(0, 0).buffer(radius)
boundary = circle.boundary

# Define chords using angle + bias
np.random.seed(42)
n_chords = 10
chords = []

for _ in range(n_chords):
    center_angle = np.random.uniform(0, 2*np.pi)
    bias = np.random.uniform(-1, 1)
    chords.append(create_chord_from_angle_bias(boundary, center_angle, bias, radius))

# Union all the chords
all_chords = unary_union(chords)
network = unary_union([boundary, all_chords])

# Polygonize
result = polygonize_full(network)
valid_polygons, cut_edges, dangles, invalid_rings = result
valid_faces = list(get_parts(valid_polygons))
valid_areas = [face.area for face in valid_faces]
