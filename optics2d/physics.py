from __future__ import annotations

from .i18n import tr

from dataclasses import dataclass, field
import math

import numpy as np

from .model import OpticalElement, OpticalSystem


C_MM_GHZ = 299.792458
EPS = 1e-8


@dataclass
class BeamSegment:
    start: np.ndarray
    direction: np.ndarray
    length: float
    q_start: complex
    wavelength_mm: float
    hit_uid: str | None = None
    source_uid: str | None = None

    def radius(self, distance: float | np.ndarray) -> np.ndarray:
        q = self.q_start + np.asarray(distance, dtype=float)
        inverse_q = 1.0 / q
        imag = np.minimum(np.imag(inverse_q), -1e-15)
        return np.sqrt(-self.wavelength_mm / (math.pi * imag))


@dataclass
class BeamWaist:
    distance: float
    position: np.ndarray
    radius: float


def segment_waist(segment: BeamSegment) -> BeamWaist | None:
    """Return a real Gaussian waist when Re(q) crosses zero inside a segment."""
    distance = -float(np.real(segment.q_start))
    tolerance = 1e-7 * max(1.0, segment.length)
    if distance < -tolerance or distance > segment.length + tolerance:
        return None
    distance = min(max(distance, 0.0), segment.length)
    position = segment.start + segment.direction * distance
    return BeamWaist(distance, position, float(segment.radius(distance)))


@dataclass
class TraceResult:
    segments: list[BeamSegment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    wavelength_mm: float = 0.0


@dataclass
class BeamTrace:
    horn_uid: str
    frequency_uid: str
    frequency_ghz: float
    result: TraceResult


def unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < EPS:
        raise ValueError("Zero-length direction vector")
    return vector / length


def tangent(element: OpticalElement) -> np.ndarray:
    if element.kind == "elliptical_mirror":
        geometry = ellipse_geometry(element)
        if geometry is not None:
            point = np.array([element.x, element.y], dtype=float)
            f1 = np.array([element.focus1_x, element.focus1_y], dtype=float)
            f2 = np.array([element.focus2_x, element.focus2_y], dtype=float)
            gradient = unit(point - f1) + unit(point - f2)
            if np.linalg.norm(gradient) > EPS:
                gradient = unit(gradient)
                return np.array([-gradient[1], gradient[0]])
    return np.array([math.cos(element.angle_rad), math.sin(element.angle_rad)], dtype=float)


def normal(element: OpticalElement) -> np.ndarray:
    t = tangent(element)
    return np.array([-t[1], t[0]])


def ellipse_geometry(
    element: OpticalElement,
) -> tuple[np.ndarray, float, float, float] | None:
    """Return center, semi-major a, semi-minor b and orientation for an ellipse.

    The ellipse is defined by two foci and the physical mirror point (x, y),
    exactly as in the source Mathematica notebook.
    """
    values = (element.focus1_x, element.focus1_y, element.focus2_x, element.focus2_y)
    if any(value is None for value in values):
        return None
    f1 = np.array([element.focus1_x, element.focus1_y], dtype=float)
    f2 = np.array([element.focus2_x, element.focus2_y], dtype=float)
    point = np.array([element.x, element.y], dtype=float)
    center = (f1 + f2) / 2.0
    c = float(np.linalg.norm(f2 - f1)) / 2.0
    a = (float(np.linalg.norm(point - f1)) + float(np.linalg.norm(point - f2))) / 2.0
    if a <= c + EPS:
        return None
    b = math.sqrt(max(a * a - c * c, 0.0))
    angle = math.atan2(f2[1] - f1[1], f2[0] - f1[0])
    return center, a, b, angle


def ellipse_vertices(element: OpticalElement) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the four global XY vertices: +a, -a, +b and -b."""
    geometry = ellipse_geometry(element)
    if geometry is None:
        return None
    center, a, b, angle = geometry
    major = np.array([math.cos(angle), math.sin(angle)], dtype=float)
    minor = np.array([-math.sin(angle), math.cos(angle)], dtype=float)
    return (
        center + a * major,
        center - a * major,
        center + b * minor,
        center - b * minor,
    )


def ellipse_effective_focal_length(element: OpticalElement) -> float:
    """Equivalent paraxial focal length used in the original notebook."""
    if None in (element.focus1_x, element.focus1_y, element.focus2_x, element.focus2_y):
        raise ValueError(tr('Both foci must be specified for an ellipse'))
    point = np.array([element.x, element.y], dtype=float)
    r1 = float(np.linalg.norm(point - np.array([element.focus1_x, element.focus1_y])))
    r2 = float(np.linalg.norm(point - np.array([element.focus2_x, element.focus2_y])))
    if r1 < EPS or r2 < EPS:
        raise ValueError(tr('The mirror point cannot coincide with a focus'))
    return 1.0 / (1.0 / r1 + 1.0 / r2)


def ray_element_intersection(
    origin: np.ndarray, direction: np.ndarray, element: OpticalElement
) -> tuple[float, np.ndarray, float] | None:
    """Intersect a ray with a finite optical-element line segment."""
    if element.kind == "elliptical_mirror":
        return ray_ellipse_intersection(origin, direction, element)
    t = tangent(element)
    center = np.array([element.x, element.y], dtype=float)
    matrix = np.column_stack((direction, -t))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < EPS:
        return None
    distance, offset = np.linalg.solve(matrix, center - origin)
    if distance <= 1e-5 or abs(offset) > element.aperture / 2.0:
        return None
    point = origin + distance * direction
    return float(distance), point, float(offset)


def ray_ellipse_intersection(
    origin: np.ndarray, direction: np.ndarray, element: OpticalElement
) -> tuple[float, np.ndarray, float] | None:
    geometry = ellipse_geometry(element)
    if geometry is None:
        return None
    center, a, b, angle = geometry
    ca, sa = math.cos(angle), math.sin(angle)
    rotation_inverse = np.array([[ca, sa], [-sa, ca]])
    local_origin = rotation_inverse @ (origin - center)
    local_direction = rotation_inverse @ direction
    aa = (local_direction[0] / a) ** 2 + (local_direction[1] / b) ** 2
    bb = 2.0 * (
        local_origin[0] * local_direction[0] / a**2
        + local_origin[1] * local_direction[1] / b**2
    )
    cc = (local_origin[0] / a) ** 2 + (local_origin[1] / b) ** 2 - 1.0
    discriminant = bb * bb - 4.0 * aa * cc
    if discriminant < 0.0 or abs(aa) < EPS:
        return None
    roots = sorted(((-bb - math.sqrt(discriminant)) / (2 * aa), (-bb + math.sqrt(discriminant)) / (2 * aa)))
    point_local = rotation_inverse @ (np.array([element.x, element.y]) - center)
    mirror_parameter = math.atan2(point_local[1] / b, point_local[0] / a)
    local_speed = math.hypot(a * math.sin(mirror_parameter), b * math.cos(mirror_parameter))
    for distance in roots:
        if distance <= 1e-5:
            continue
        hit = origin + distance * direction
        hit_local = rotation_inverse @ (hit - center)
        parameter = math.atan2(hit_local[1] / b, hit_local[0] / a)
        delta = (parameter - mirror_parameter + math.pi) % (2 * math.pi) - math.pi
        arc_offset = delta * local_speed
        if abs(arc_offset) <= element.aperture / 2.0:
            return float(distance), hit, float(arc_offset)
    return None


def surface_normal(element: OpticalElement, point: np.ndarray | None = None) -> np.ndarray:
    if element.kind != "elliptical_mirror" or point is None:
        return normal(element)
    geometry = ellipse_geometry(element)
    if geometry is None:
        return normal(element)
    center, a, b, angle = geometry
    ca, sa = math.cos(angle), math.sin(angle)
    inverse = np.array([[ca, sa], [-sa, ca]])
    local = inverse @ (point - center)
    gradient_local = np.array([local[0] / a**2, local[1] / b**2])
    rotation = np.array([[ca, -sa], [sa, ca]])
    return unit(rotation @ gradient_local)


def reflected(
    direction: np.ndarray, element: OpticalElement, point: np.ndarray | None = None
) -> np.ndarray:
    n = surface_normal(element, point)
    return unit(direction - 2.0 * float(np.dot(direction, n)) * n)


def refracted_by_thin_lens(
    direction: np.ndarray, element: OpticalElement, transverse_offset: float
) -> np.ndarray:
    """Paraxial central-ray transform theta_out = theta_in - y/f."""
    t = tangent(element)
    n = normal(element)
    if float(np.dot(direction, n)) < 0.0:
        n = -n
    alpha_in = math.atan2(float(np.dot(direction, t)), float(np.dot(direction, n)))
    alpha_out = alpha_in - transverse_offset / element.focal_length
    return unit(math.cos(alpha_out) * n + math.sin(alpha_out) * t)


def focused_reflection(
    direction: np.ndarray, element: OpticalElement, transverse_offset: float
) -> np.ndarray:
    base = reflected(direction, element)
    if abs(element.focal_length) < EPS:
        return base
    t = tangent(element)
    angle = -transverse_offset / element.focal_length
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotated = np.array(
        [cos_a * base[0] - sin_a * base[1], sin_a * base[0] + cos_a * base[1]]
    )
    return unit(rotated)


def apply_focusing_power(q_in: complex, focal_length: float) -> complex:
    """ABCD law for a thin lens or equivalent focusing mirror."""
    if abs(focal_length) < EPS:
        raise ValueError("Focal length cannot be zero")
    return q_in / (1.0 - q_in / focal_length)


def orient_mirror_to_target(
    element: OpticalElement, incoming_direction: np.ndarray
) -> float:
    if element.target_x is None or element.target_y is None:
        raise ValueError("The mirror has no target focus")
    outgoing = unit(np.array([element.target_x - element.x, element.target_y - element.y]))
    n = unit(incoming_direction - outgoing)
    t = np.array([-n[1], n[0]])
    return math.degrees(math.atan2(t[1], t[0]))


def trace_system(
    system: OpticalSystem,
    max_interactions: int = 30,
    horn: OpticalElement | None = None,
    frequency_ghz: float | None = None,
) -> TraceResult:
    horn = horn or system.horn
    if horn is None:
        return TraceResult(warnings=[tr('Add and enable a horn as the beam source.')])
    active = horn.active_frequencies()
    frequency = frequency_ghz or (active[0].frequency_ghz if active else horn.frequency_ghz)
    if frequency <= 0.0 or horn.waist_radius <= 0.0:
        return TraceResult(warnings=[tr('Frequency and waist radius must be positive.')])

    wavelength = C_MM_GHZ / frequency
    rayleigh = math.pi * horn.waist_radius**2 / wavelength
    q = complex(0.0, rayleigh)
    origin = np.array([horn.x, horn.y], dtype=float)
    direction = np.array([math.cos(horn.angle_rad), math.sin(horn.angle_rad)])
    travelled = 0.0
    ignored_uid: str | None = None
    result = TraceResult(wavelength_mm=wavelength)

    fallback_candidates = [
        item for item in system.elements
        if item.enabled and item.kind not in ("horn", "cryostat", "ruler", "target", "block")
    ]
    current_uid = horn.uid
    for _ in range(max_interactions):
        remaining = system.max_path_mm - travelled
        if remaining <= EPS:
            break
        # Links describe mechanical constraints and auto-aiming, not visibility.
        # Every enabled physical surface must interact with an incident beam.
        candidates = fallback_candidates
        hits: list[tuple[float, np.ndarray, float, OpticalElement]] = []
        for item in candidates:
            if item.uid == ignored_uid:
                continue
            hit = ray_element_intersection(origin, direction, item)
            if hit is not None and hit[0] <= remaining:
                hits.append((*hit, item))
        if not hits:
            result.segments.append(BeamSegment(origin, direction, remaining, q, wavelength, source_uid=current_uid))
            break
        distance, point, offset, item = min(hits, key=lambda value: value[0])
        result.segments.append(
            BeamSegment(origin, direction, distance, q, wavelength, hit_uid=item.uid, source_uid=current_uid)
        )
        q += distance
        try:
            if item.kind == "lens":
                direction = refracted_by_thin_lens(direction, item, offset)
                q = apply_focusing_power(q, item.focal_length)
            elif item.kind == "curved_mirror":
                direction = focused_reflection(direction, item, offset)
                q = apply_focusing_power(q, item.focal_length)
            elif item.kind == "elliptical_mirror":
                direction = reflected(direction, item, point)
                q = apply_focusing_power(q, ellipse_effective_focal_length(item))
            else:
                direction = reflected(direction, item)
        except ValueError as exc:
            result.warnings.append(f"{item.name}: {exc}")
            break
        travelled += distance
        origin = point + direction * 1e-4
        ignored_uid = item.uid
        current_uid = item.uid
    else:
        result.warnings.append(tr('The interaction limit was reached; the ray path may form a loop.'))
    return result


def trace_all_beams(system: OpticalSystem) -> list[BeamTrace]:
    traces: list[BeamTrace] = []
    for horn in system.horns:
        for channel in horn.active_frequencies():
            traces.append(
                BeamTrace(
                    horn.uid,
                    channel.uid,
                    channel.frequency_ghz,
                    trace_system(system, horn=horn, frequency_ghz=channel.frequency_ghz),
                )
            )
    return traces
