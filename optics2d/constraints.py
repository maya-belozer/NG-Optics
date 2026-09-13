from __future__ import annotations

import math
import numpy as np

from .model import OpticalElement, OpticalSystem


def horn_waveguide_position(horn: OpticalElement) -> np.ndarray:
    """Waveguide end derived from PCL, horn length and PCL depth."""
    return np.array([horn.waveguide_x, horn.waveguide_y], dtype=float)


def block_endpoints(block: OpticalElement) -> tuple[np.ndarray, np.ndarray]:
    """Block A/B endpoints. A is adjacent to an incoming horn."""
    direction = np.array([math.cos(block.angle_rad), math.sin(block.angle_rad)])
    first = np.array([block.x, block.y], dtype=float)
    second = first - max(block.block_length, 0.0) * direction
    return first, second


def enforce_blocks(system: OpticalSystem) -> None:
    """Attach block endpoints to horn waveguides and orient a second horn backwards."""
    for block in (item for item in system.elements if item.kind == "block"):
        incoming = next((item for item in system.incoming(block.uid) if item.kind == "horn"), None)
        if incoming is not None:
            block.angle_deg = incoming.angle_deg
            first = horn_waveguide_position(incoming)
            block.x, block.y = float(first[0]), float(first[1])
        _first, second = block_endpoints(block)
        outgoing = next((item for item in system.outgoing(block.uid) if item.kind == "horn"), None)
        if outgoing is not None:
            outgoing.angle_deg = block.angle_deg + 180.0
            back = max(outgoing.horn_length - outgoing.pcl_depth, 0.0)
            direction = np.array([math.cos(outgoing.angle_rad), math.sin(outgoing.angle_rad)])
            pcl = second + back * direction
            outgoing.x, outgoing.y = float(pcl[0]), float(pcl[1])


def attach_assembly_to_block(
    system: OpticalSystem,
    block: OpticalElement,
    horn: OpticalElement,
    member_uids: set[str],
) -> None:
    """Rigidly attach the selected horn assembly to block endpoint B."""
    if block.kind != "block" or horn.kind != "horn" or horn.uid not in member_uids:
        return
    members = [
        item for item in system.elements
        if item.uid in member_uids and item.uid != block.uid
    ]
    if not members:
        return
    pivot = horn_waveguide_position(horn)
    _first, destination = block_endpoints(block)
    delta = math.radians(block.angle_deg + 180.0 - horn.angle_deg)
    c, s = math.cos(delta), math.sin(delta)

    def transform(x: float, y: float) -> tuple[float, float]:
        relative = np.array([x, y], dtype=float) - pivot
        rotated = np.array([c * relative[0] - s * relative[1], s * relative[0] + c * relative[1]])
        result = destination + rotated
        return float(result[0]), float(result[1])

    for member in members:
        member.x, member.y = transform(member.x, member.y)
        member.angle_deg += math.degrees(delta)
        for prefix in ("focus1", "focus2", "target"):
            x_value = getattr(member, f"{prefix}_x")
            y_value = getattr(member, f"{prefix}_y")
            if x_value is not None and y_value is not None:
                x_new, y_new = transform(x_value, y_value)
                setattr(member, f"{prefix}_x", x_new)
                setattr(member, f"{prefix}_y", y_new)
    for member in members:
        if member.kind == "target":
            update_target_relative_from_position(system, member)


def ruler_attachments(system: OpticalSystem, ruler: OpticalElement) -> list[OpticalElement]:
    """Return unique objects attached to a ruler, independent of edge direction."""
    result: list[OpticalElement] = []
    seen: set[str] = set()
    for link in system.links:
        if not link.enabled:
            continue
        other_uid = (
            link.target_uid if link.source_uid == ruler.uid
            else link.source_uid if link.target_uid == ruler.uid
            else None
        )
        if other_uid is None or other_uid in seen:
            continue
        other = system.element(other_uid)
        if other is not None and other.kind != "ruler":
            seen.add(other_uid)
            result.append(other)
    return result


def target_predecessor(system: OpticalSystem, target: OpticalElement) -> OpticalElement | None:
    """First unique enabled incoming object that defines a target's polar origin."""
    seen: set[str] = set()
    for link in system.links:
        if not link.enabled or link.target_uid != target.uid or link.source_uid in seen:
            continue
        seen.add(link.source_uid)
        source = system.element(link.source_uid)
        if source is not None and source.kind not in ("ruler", "target"):
            return source
    return None


def update_target_relative_from_position(system: OpticalSystem, target: OpticalElement) -> None:
    predecessor = target_predecessor(system, target)
    if predecessor is None:
        return
    dx, dy = target.x - predecessor.x, target.y - predecessor.y
    target.relative_distance = math.hypot(dx, dy)
    target.relative_angle_deg = math.degrees(math.atan2(dy, dx))


def enforce_targets(system: OpticalSystem) -> None:
    for target in (item for item in system.elements if item.kind == "target"):
        predecessor = target_predecessor(system, target)
        if predecessor is None:
            continue
        if target.position_mode == "relative":
            angle = math.radians(target.relative_angle_deg)
            target.x = predecessor.x + target.relative_distance * math.cos(angle)
            target.y = predecessor.y + target.relative_distance * math.sin(angle)
        else:
            update_target_relative_from_position(system, target)


def primary_ellipse(system: OpticalSystem) -> OpticalElement | None:
    return next((item for item in system.elements if item.kind == "elliptical_mirror" and item.enabled), None)


def linked_ellipse_for_horn(system: OpticalSystem, horn: OpticalElement) -> OpticalElement | None:
    return next((item for item in system.outgoing(horn.uid) if item.kind == "elliptical_mirror"), None)


def linked_incoming_ellipse(system: OpticalSystem, element: OpticalElement) -> OpticalElement | None:
    return next((item for item in system.incoming(element.uid) if item.kind == "elliptical_mirror"), None)


def project_to_line(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator < 1e-12:
        return point
    return start + direction * float(np.dot(point - start, direction)) / denominator


def move_with_constraints(
    system: OpticalSystem,
    element: OpticalElement,
    x: float,
    y: float,
    handle: str = "center",
) -> None:
    """Apply optional mechanical/optical links during interactive dragging."""
    if element.kind == "target" and handle == "center":
        element.x, element.y = x, y
        update_target_relative_from_position(system, element)
        return
    if element.kind == "ruler":
        attachments = ruler_attachments(system, element)
        if handle == "ruler1":
            if not attachments:
                element.x, element.y = x, y
            return
        if handle == "ruler2":
            if len(attachments) < 2:
                element.target_x, element.target_y = x, y
            return
    ellipse = (
        linked_ellipse_for_horn(system, element)
        if element.kind == "horn" and system.link_horn_to_ellipse
        else linked_incoming_ellipse(system, element)
        if element.kind == "plane_mirror" and system.link_plane_to_ellipse
        else None
    )
    proposed = np.array([x, y], dtype=float)
    if handle == "focus1" and element.kind == "elliptical_mirror":
        element.focus1_x, element.focus1_y = x, y
        horn = next((item for item in system.incoming(element.uid) if item.kind == "horn"), None)
        if horn is not None:
            horn.x, horn.y = x, y
        return
    if handle == "focus2" and element.kind == "elliptical_mirror":
        element.focus2_x, element.focus2_y = x, y
        element.target_x, element.target_y = x, y
        enforce_links(system)
        return
    if handle != "center":
        return
    if element.kind == "horn" and ellipse is not None:
        focus = np.array([ellipse.focus1_x, ellipse.focus1_y], dtype=float)
        mirror_point = np.array([ellipse.x, ellipse.y], dtype=float)
        result = project_to_line(proposed, mirror_point, focus)
        element.x, element.y = float(result[0]), float(result[1])
        ellipse.focus1_x, ellipse.focus1_y = element.x, element.y
        return
    if element.kind == "plane_mirror" and ellipse is not None:
        focus = np.array([ellipse.focus2_x, ellipse.focus2_y], dtype=float)
        mirror_point = np.array([ellipse.x, ellipse.y], dtype=float)
        result = project_to_line(proposed, mirror_point, focus)
        element.x, element.y = float(result[0]), float(result[1])
        return
    element.x, element.y = x, y
    if element.kind == "elliptical_mirror":
        enforce_links(system)


def enforce_links(system: OpticalSystem) -> None:
    enforce_blocks(system)
    enforce_targets(system)
    for link in system.links:
        if not link.enabled:
            continue
        source, target = system.element(link.source_uid), system.element(link.target_uid)
        if source is None or target is None:
            continue
        if system.link_horn_to_ellipse and source.kind == "horn" and target.kind == "elliptical_mirror":
            target.focus1_x, target.focus1_y = source.x, source.y
        if system.link_plane_to_ellipse and source.kind == "elliptical_mirror" and target.kind == "plane_mirror":
            if None in (source.focus2_x, source.focus2_y):
                continue
            point = np.array([target.x, target.y], dtype=float)
            result = project_to_line(
                point,
                np.array([source.x, source.y], dtype=float),
                np.array([source.focus2_x, source.focus2_y], dtype=float),
            )
            target.x, target.y = float(result[0]), float(result[1])
        if source.kind in ("plane_mirror", "curved_mirror") and target.kind == "target":
            source.target_x, source.target_y = target.x, target.y
    enforce_rulers(system)


def enforce_dependents(system: OpticalSystem) -> None:
    """Update targets and rulers without applying mechanical line constraints."""
    enforce_blocks(system)
    enforce_targets(system)
    for link in system.links:
        if not link.enabled:
            continue
        source, target = system.element(link.source_uid), system.element(link.target_uid)
        if source is not None and target is not None:
            if source.kind in ("plane_mirror", "curved_mirror") and target.kind == "target":
                source.target_x, source.target_y = target.x, target.y
    enforce_rulers(system)


def enforce_rulers(system: OpticalSystem) -> None:
    """Update measuring objects without applying optical motion constraints."""
    for ruler in (item for item in system.elements if item.kind == "ruler"):
        attached = ruler_attachments(system, ruler)
        if attached:
            ruler.x, ruler.y = attached[0].x, attached[0].y
        if len(attached) >= 2:
            ruler.target_x, ruler.target_y = attached[1].x, attached[1].y
