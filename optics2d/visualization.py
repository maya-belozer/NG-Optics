from __future__ import annotations

from .i18n import tr, LocalizedLabels

import math
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QRubberBand,
)

from .constraints import (
    block_endpoints,
    enforce_links,
    enforce_dependents,
    move_with_constraints,
    ruler_attachments,
    target_predecessor,
    update_target_relative_from_position,
)
from .model import OpticalElement, OpticalSystem
from .physics import BeamSegment, ellipse_geometry, normal, segment_waist, tangent, trace_all_beams


KIND_LABELS = LocalizedLabels({
    'horn': 'Horn',
    'lens': 'Thin lens',
    'plane_mirror': 'Plane mirror',
    'curved_mirror': 'Focusing mirror',
    'elliptical_mirror': 'Elliptical mirror',
    'cryostat': 'Cryostat',
    'ruler': 'Ruler',
    'target': 'Target',
    'block': 'Block',
})

BEAM_COLORS = ("#d62728", "#9467bd", "#2ca02c", "#8c564b", "#e377c2", "#17becf", "#bcbd22")
SELECTED_COLOR = "#ff8c00"
OBJECT_COLOR = "#1874b4"

pg.setConfigOptions(antialias=True)


def _pen(color: str, width: float = 1.5, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> QPen:
    return pg.mkPen(color=color, width=width, style=style)


def cryostat_window_corners(element: OpticalElement, index: int) -> np.ndarray:
    """Four corners of a 40×36 mm window, spaced by 90° around the shell."""
    angle = element.angle_rad + index * math.pi / 2.0
    direction = np.array([math.cos(angle), math.sin(angle)])
    side = np.array([-direction[1], direction[0]])
    center = np.array([element.x, element.y]) + (element.radius + 20.0) * direction
    return np.vstack((
        center - 20.0 * direction - 18.0 * side,
        center + 20.0 * direction - 18.0 * side,
        center + 20.0 * direction + 18.0 * side,
        center - 20.0 * direction + 18.0 * side,
    ))


class SceneView(pg.PlotWidget):
    """PyQtGraph renderer with direct manipulation and rubber-band selection."""

    def __init__(
        self,
        get_system: Callable[[], OpticalSystem],
        on_select: Callable[[str, bool], None],
        on_change: Callable[[], None],
        on_box_select: Callable[[set[str], bool], None],
        on_delete: Callable[[], None],
        on_edit_started: Callable[[], None],
        on_edit_finished: Callable[[], None],
    ) -> None:
        super().__init__()
        self.get_system = get_system
        self.on_select = on_select
        self.on_change = on_change
        self.on_box_select = on_box_select
        self.on_delete = on_delete
        self.on_edit_started = on_edit_started
        self.on_edit_finished = on_edit_finished
        self.selected_uid: str | None = None
        self.selected_uids: set[str] = set()
        self.edge_handles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.drag: tuple[str, str] | None = None
        self.rotation_mode = False
        self.group_rotation: dict[str, object] | None = None
        self.group_translation: dict[str, object] | None = None
        self.target_orbit: dict[str, object] | None = None
        self.ruler_drag_snapshot: dict[str, object] | None = None
        self.edit_in_progress = False
        self._rubber_origin: QPoint | None = None
        self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self._emit_change)

        self.setBackground("white")
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setLabel("bottom", tr('X, mm'))
        self.setLabel("left", tr('Y, mm'))
        self.setTitle(tr('2D layout: Gaussian beam and optical objects'))
        self.setAspectLocked(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.getPlotItem().setMenuEnabled(False)
        self.getPlotItem().hideButtons()
        self.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_selected(self, uid: str | None, selected_uids: set[str] | None = None) -> None:
        self.selected_uid = uid
        self.selected_uids = set(selected_uids or ({uid} if uid else set()))

    def set_rotation_mode(self, enabled: bool) -> None:
        self.rotation_mode = enabled
        self.setCursor(Qt.CursorShape.OpenHandCursor if enabled else Qt.CursorShape.CrossCursor)
        self.draw()

    def rotatable_selection_count(self) -> int:
        return sum(
            element.uid in self.selected_uids
            for element in self.get_system().elements
        )

    def _queue_change(self) -> None:
        # Gaussian/ABCD redraws are intentionally capped near 40 FPS while dragging.
        if not self._redraw_timer.isActive():
            self._redraw_timer.start(25)

    def _emit_change(self) -> None:
        self.on_change()

    def _flush_change(self) -> None:
        if self._redraw_timer.isActive():
            self._redraw_timer.stop()
        self.on_change()

    def view_range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        ranges = self.getViewBox().viewRange()
        return (float(ranges[0][0]), float(ranges[0][1])), (float(ranges[1][0]), float(ranges[1][1]))

    def draw(self, reset_view: bool = False) -> None:
        old_range = self.view_range()
        self.clear()
        self.edge_handles.clear()
        system = self.get_system()
        mirrors = {
            item.uid for item in system.elements
            if item.kind in ("plane_mirror", "curved_mirror", "elliptical_mirror")
        }
        for beam_index, beam in enumerate(trace_all_beams(system)):
            color = BEAM_COLORS[beam_index % len(BEAM_COLORS)]
            boundaries: list[tuple[np.ndarray, np.ndarray]] = []
            for segment in beam.result.segments:
                boundaries.append(self._draw_gaussian_segment(segment, system.beam_scale, system.render_beam_bounds, color))
                waist = segment_waist(segment)
                if waist is not None:
                    self._draw_waist(
                        waist.position,
                        waist.radius,
                        segment.direction,
                        beam.frequency_ghz,
                        system.beam_scale,
                        color,
                    )
            if system.render_beam_bounds and system.simulate_boundary_reflections:
                for index in range(len(boundaries) - 1):
                    if beam.result.segments[index].hit_uid in mirrors:
                        self._draw_boundary_joint(
                            beam.result.segments[index], boundaries[index], boundaries[index + 1], color
                        )
        for element in system.elements:
            self._draw_element(element)
        if (self.rotation_mode and self.selected_uids) or self.group_rotation is not None:
            self._draw_rotation_marker()
        if reset_view:
            self.enableAutoRange()
            self.autoRange(padding=0.08)
        else:
            self.setRange(xRange=old_range[0], yRange=old_range[1], padding=0)

    def _add_curve(self, points: np.ndarray, color: str, width: float = 1.5, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> pg.PlotDataItem:
        item = pg.PlotDataItem(points[:, 0], points[:, 1], pen=_pen(color, width, style), connect="finite")
        self.addItem(item)
        return item

    def _draw_gaussian_segment(
        self, segment: BeamSegment, level: float, draw_bounds: bool, color: str
    ) -> tuple[np.ndarray, np.ndarray]:
        count = max(30, int(segment.length / 1.5))
        distance = np.linspace(0.0, segment.length, count)
        center = segment.start[:, None] + segment.direction[:, None] * distance
        side = np.array([-segment.direction[1], segment.direction[0]])[:, None]
        radii = segment.radius(distance) * level
        upper = (center + side * radii).T
        lower = (center - side * radii).T
        self._add_curve(center.T, color, 1.35)
        if draw_bounds:
            upper_item = self._add_curve(upper, color, 1.25)
            lower_item = self._add_curve(lower, color, 1.25)
            fill_color = QColor(color)
            fill_color.setAlpha(24)
            fill = pg.FillBetweenItem(upper_item, lower_item, brush=pg.mkBrush(fill_color))
            self.addItem(fill)
        return upper, lower

    def _draw_waist(
        self,
        position: np.ndarray,
        radius: float,
        direction: np.ndarray,
        frequency_ghz: float,
        level: float,
        color: str,
    ) -> None:
        side = np.array([-direction[1], direction[0]])
        ends = np.vstack((position - side * radius * level, position + side * radius * level))
        self._add_curve(ends, color, 2.5)
        self.addItem(pg.ScatterPlotItem(
            [position[0]], [position[1]], symbol="d", size=11,
            brush=pg.mkBrush("white"), pen=_pen(color, 2.2),
        ))
        self._add_text(
            tr('Waist {0:g} GHz\n({1:.2f}; {2:.2f}) mm, w₀={3:.3f} mm', frequency_ghz, position[0], position[1], radius),
            position[0] + 3,
            position[1] + 3,
            color,
            8,
        )

    def _draw_boundary_joint(
        self,
        incoming: BeamSegment,
        incoming_bounds: tuple[np.ndarray, np.ndarray],
        outgoing_bounds: tuple[np.ndarray, np.ndarray],
        color: str,
    ) -> None:
        center = incoming.start + incoming.direction * incoming.length
        starts = [incoming_bounds[0][-1], incoming_bounds[1][-1]]
        ends = [outgoing_bounds[0][0], outgoing_bounds[1][0]]
        if sum(np.linalg.norm(a - b) for a, b in zip(starts, ends[::-1])) < sum(np.linalg.norm(a - b) for a, b in zip(starts, ends)):
            ends.reverse()
        parameter = np.linspace(0.0, 1.0, 24)[:, None]
        for start, end in zip(starts, ends):
            curve = (1 - parameter) ** 2 * start + 2 * (1 - parameter) * parameter * center + parameter**2 * end
            self._add_curve(curve, color, 1.4)

    def _style(self, element: OpticalElement) -> tuple[str, float]:
        return (SELECTED_COLOR if element.uid in self.selected_uids else OBJECT_COLOR, 1.0 if element.enabled else 0.25)

    def _add_text(self, text: str, x: float, y: float, color: str, size: int = 9) -> None:
        label = pg.TextItem(text, color=color, anchor=(0, 1))
        label.setPos(x, y)
        label.setFont(pg.QtGui.QFont("Arial", size))
        self.addItem(label)

    def _draw_element(self, element: OpticalElement) -> None:
        color, alpha = self._style(element)
        faded = QColor(color)
        faded.setAlphaF(alpha)
        color = faded
        if element.kind == "target":
            self._draw_target(element, color)
            return
        if element.kind == "ruler":
            self._draw_ruler(element, color)
            return
        if element.kind == "cryostat":
            self._draw_cryostat(element, color)
            return
        if element.kind == "block":
            self._draw_block(element, color)
            return
        if element.kind == "elliptical_mirror":
            self._draw_ellipse(element, color)
            return
        center = np.array([element.x, element.y], dtype=float)
        if element.kind == "horn":
            direction = np.array([math.cos(element.angle_rad), math.sin(element.angle_rad)])
            side = np.array([-direction[1], direction[0]])
            mouth = center + element.pcl_depth * direction
            throat = center - max(element.horn_length - element.pcl_depth, 1.0) * direction
            points = np.vstack((throat - .5 * side, mouth - element.aperture / 2 * side,
                                mouth + element.aperture / 2 * side, throat + .5 * side))
            polygon = QGraphicsPolygonItem(QPolygonF([QPointF(float(x), float(y)) for x, y in points]))
            horn_fill = QColor(color)
            horn_fill.setAlphaF(0.22 * alpha)
            polygon.setPen(_pen(color, 1.5)); polygon.setBrush(pg.mkBrush(horn_fill))
            self.addItem(polygon)
        else:
            t, n = tangent(element), normal(element)
            ends = center[:, None] + t[:, None] * np.array([-element.aperture / 2, element.aperture / 2])
            self._add_curve(ends.T, color, 5 if element.kind == "lens" else 3,
                            Qt.PenStyle.DashLine if element.kind == "curved_mirror" else Qt.PenStyle.SolidLine)
            self._add_curve(np.vstack((center, center + n * 8)), color, 1.2)
            arrow = pg.ArrowItem(pos=center + n * 8, angle=math.degrees(math.atan2(-n[1], -n[0])),
                                 brush=pg.mkBrush(color), pen=_pen(color, 1), headLen=8)
            self.addItem(arrow)
            if "mirror" in element.kind:
                self.edge_handles[element.uid] = (ends[:, 0], ends[:, 1])
                if element.uid in self.selected_uids:
                    self._draw_handles(self.edge_handles[element.uid], color)
            if element.kind == "lens" and element.focal_length:
                foci = np.vstack((center - n * element.focal_length, center + n * element.focal_length))
                self.addItem(pg.ScatterPlotItem(foci[:, 0], foci[:, 1], symbol="x", size=10, pen=_pen(color, 2)))
        if element.target_x is not None and element.target_y is not None and "mirror" in element.kind:
            has_target_object = any(
                target.kind == "target" for target in self.get_system().outgoing(element.uid)
            )
            if not has_target_object:
                self._add_curve(np.array([[element.x, element.y], [element.target_x, element.target_y]]), color, 1, Qt.PenStyle.DotLine)
                self.addItem(pg.ScatterPlotItem(
                    [element.target_x], [element.target_y], symbol="x", size=14, pen=_pen("#d62728", 2.5)
                ))
                self._add_text(
                    tr('Target ({0:.2f}; {1:.2f})', element.target_x, element.target_y),
                    element.target_x + 3,
                    element.target_y + 3,
                    "#d62728",
                    9,
                )
        self.addItem(pg.ScatterPlotItem([element.x], [element.y], size=9, brush=pg.mkBrush(color), pen=_pen(color, 1)))
        self._add_text(element.name, element.x + 3, element.y + 3, color)

    def _draw_block(self, element: OpticalElement, color: QColor) -> None:
        first, second = block_endpoints(element)
        direction = np.array([math.cos(element.angle_rad), math.sin(element.angle_rad)])
        side = np.array([-direction[1], direction[0]])
        half_width = max(min(element.aperture / 2.0, 12.0), 3.0)
        corners = np.vstack((
            first - half_width * side,
            second - half_width * side,
            second + half_width * side,
            first + half_width * side,
        ))
        polygon = QGraphicsPolygonItem(QPolygonF([QPointF(float(x), float(y)) for x, y in corners]))
        fill = QColor(color)
        fill.setAlphaF(0.22)
        polygon.setPen(_pen(color, 2.0))
        polygon.setBrush(pg.mkBrush(fill))
        self.addItem(polygon)
        self.addItem(pg.ScatterPlotItem(
            [first[0], second[0]], [first[1], second[1]], symbol="s", size=10,
            brush=pg.mkBrush("white"), pen=_pen(color, 2),
        ))
        midpoint = (first + second) / 2.0
        self._add_text(tr('{0}: {1:g} mm', element.name, element.block_length), midpoint[0] + 3, midpoint[1] + 3, color)

    def _draw_target(self, element: OpticalElement, color: QColor) -> None:
        predecessor = target_predecessor(self.get_system(), element)
        if predecessor is not None:
            self._add_curve(
                np.array([[predecessor.x, predecessor.y], [element.x, element.y]]),
                "#c0392b",
                1.4,
                Qt.PenStyle.DashLine,
            )
        self.addItem(pg.ScatterPlotItem(
            [element.x], [element.y], symbol="o", size=20,
            brush=pg.mkBrush(255, 255, 255, 0), pen=_pen("#c0392b", 2.5),
        ))
        self.addItem(pg.ScatterPlotItem(
            [element.x], [element.y], symbol="x", size=16, pen=_pen("#c0392b", 2.5),
        ))
        self._add_text(
            tr('{0}\nR={1:.3f} mm, φ={2:.2f}°', element.name, element.relative_distance, element.relative_angle_deg),
            element.x + 3,
            element.y + 3,
            color,
            9,
        )

    def _draw_ruler(self, element: OpticalElement, color: str) -> None:
        if element.target_x is None or element.target_y is None:
            element.target_x, element.target_y = element.x + 100.0, element.y
        first = np.array([element.x, element.y], dtype=float)
        second = np.array([element.target_x, element.target_y], dtype=float)
        distance = float(np.linalg.norm(second - first))
        ruler_color = QColor("#6f42c1") if element.uid not in self.selected_uids else QColor(SELECTED_COLOR)
        self._add_curve(np.vstack((first, second)), ruler_color, 2.5)
        self.addItem(pg.ScatterPlotItem(
            [first[0], second[0]], [first[1], second[1]], symbol="o", size=11,
            brush=pg.mkBrush("white"), pen=_pen(ruler_color, 2.5),
        ))
        self._add_text("A", first[0] + 2, first[1] + 2, ruler_color, 9)
        self._add_text("B", second[0] + 2, second[1] + 2, ruler_color, 9)
        midpoint = (first + second) / 2.0
        attached = ruler_attachments(self.get_system(), element)
        mode = tr('attached') if attached else tr('free')
        self._add_text(
            tr('{0}: {1:.3f} mm ({2})', element.name, distance, mode),
            midpoint[0] + 3,
            midpoint[1] + 3,
            ruler_color,
            9,
        )

    def _draw_handles(self, handles: tuple[np.ndarray, np.ndarray], color: str) -> None:
        points = np.vstack(handles)
        self.addItem(pg.ScatterPlotItem(points[:, 0], points[:, 1], symbol="s", size=10,
                                        brush=pg.mkBrush("white"), pen=_pen(color, 2)))

    def _draw_cryostat(self, element: OpticalElement, color: str) -> None:
        direction = np.array([math.cos(element.angle_rad), math.sin(element.angle_rad)])
        side = np.array([-direction[1], direction[0]])
        inner_center = np.array([element.x, element.y]) + element.inner_offset_x * direction + element.inner_offset_y * side
        for x, y, radius in ((element.x, element.y, element.radius),
                             (inner_center[0], inner_center[1], element.inner_radius)):
            item = QGraphicsEllipseItem(x - radius, y - radius, 2 * radius, 2 * radius)
            item.setPen(_pen(color, 2)); self.addItem(item)
        for index in range(max(1, min(int(element.window_count), 4))):
            corners = cryostat_window_corners(element, index)
            window = QGraphicsPolygonItem(QPolygonF([QPointF(float(x), float(y)) for x, y in corners]))
            window.setPen(_pen("#169c46", 2.5)); window.setBrush(pg.mkBrush(0, 0, 0, 0)); self.addItem(window)
        if element.uid in self.selected_uids:
            self.addItem(pg.ScatterPlotItem(
                [element.x], [element.y], symbol="+", size=16, pen=_pen(SELECTED_COLOR, 3)
            ))
        self._add_text(element.name, element.x - element.radius, element.y + element.radius + 5, color)

    def _draw_ellipse(self, element: OpticalElement, color: str) -> None:
        geometry = ellipse_geometry(element)
        if geometry is None:
            self._add_text(tr('Invalid ellipse geometry'), element.x, element.y, "#d62728")
            return
        center, a, b, angle = geometry
        parameter = np.linspace(0.0, 2.0 * math.pi, 500)
        ca, sa = math.cos(angle), math.sin(angle)
        x = center[0] + ca * a * np.cos(parameter) - sa * b * np.sin(parameter)
        y = center[1] + sa * a * np.cos(parameter) + ca * b * np.sin(parameter)
        self._add_curve(np.column_stack((x, y)), color, 1.3)
        index = int(np.argmin((x - element.x) ** 2 + (y - element.y) ** 2))
        half = max(4, int(500 * element.aperture / max(2 * math.pi * a, 1.0) / 2))
        indices = (np.arange(index - half, index + half + 1) % 500).astype(int)
        self._add_curve(np.column_stack((x[indices], y[indices])), color, 5)
        edge1, edge2 = np.array([x[indices[0]], y[indices[0]]]), np.array([x[indices[-1]], y[indices[-1]]])
        self.edge_handles[element.uid] = (edge1, edge2)
        if element.uid in self.selected_uids:
            self._draw_handles((edge1, edge2), color)
        foci = np.array([[element.focus1_x, element.focus1_y], [element.focus2_x, element.focus2_y]], dtype=float)
        self.addItem(pg.ScatterPlotItem(foci[:, 0], foci[:, 1], symbol="x", size=12, pen=_pen(color, 2)))
        self._add_curve(np.vstack((foci[0], [element.x, element.y], foci[1])), color, 1, Qt.PenStyle.DotLine)
        self._add_text("F1", foci[0, 0] + 2, foci[0, 1] + 2, color)
        self._add_text("F2", foci[1, 0] + 2, foci[1, 1] + 2, color)
        self.addItem(pg.ScatterPlotItem([element.x], [element.y], size=9, brush=pg.mkBrush(color), pen=_pen(color, 1)))
        self._add_text(element.name, element.x + 3, element.y + 3, color)

    def _selection_center(self) -> np.ndarray | None:
        elements = [e for e in self.get_system().elements if e.uid in self.selected_uids]
        return np.mean([self._element_center(e) for e in elements], axis=0) if elements else None

    @staticmethod
    def _element_center(element: OpticalElement) -> np.ndarray:
        if element.kind == "ruler" and element.target_x is not None and element.target_y is not None:
            return np.array([(element.x + element.target_x) / 2.0, (element.y + element.target_y) / 2.0])
        if element.kind == "block":
            first, second = block_endpoints(element)
            return (first + second) / 2.0
        return np.array([element.x, element.y])

    def _draw_rotation_marker(self) -> None:
        center = self._selection_center()
        if center is None:
            return
        x_range, y_range = self.view_range()
        radius = 0.025 * max(x_range[1] - x_range[0], y_range[1] - y_range[0])
        circle = QGraphicsEllipseItem(center[0] - radius, center[1] - radius, 2 * radius, 2 * radius)
        circle.setPen(_pen("#e31a1c", 2, Qt.PenStyle.DashLine)); self.addItem(circle)
        self.addItem(pg.ScatterPlotItem([center[0]], [center[1]], symbol="+", size=18, pen=_pen("#e31a1c", 3)))
        self._add_text(tr('ROTATION CENTER — right-drag'), center[0] + radius, center[1] + radius, "#e31a1c", 10)

    def _pick(self, x: float, y: float) -> tuple[OpticalElement, str] | None:
        x_range, y_range = self.view_range()
        tolerance = 0.018 * max(x_range[1] - x_range[0], y_range[1] - y_range[0])
        options: list[tuple[float, OpticalElement, str]] = []
        for element in self.get_system().elements:
            if element.kind == "ruler":
                second = np.array([
                    element.target_x if element.target_x is not None else element.x + 100.0,
                    element.target_y if element.target_y is not None else element.y,
                ])
                first = np.array([element.x, element.y])
                segment = second - first
                denominator = float(np.dot(segment, segment))
                projection = 0.0 if denominator < 1e-12 else min(
                    max(float(np.dot(np.array([x, y]) - first, segment)) / denominator, 0.0), 1.0
                )
                body_distance = float(np.linalg.norm(np.array([x, y]) - (first + projection * segment)))
                options.extend((
                    (float(np.linalg.norm(first - [x, y])), element, "ruler1"),
                    (float(np.linalg.norm(second - [x, y])), element, "ruler2"),
                    (body_distance + 0.25 * tolerance, element, "ruler_body"),
                ))
                continue
            if element.kind == "cryostat":
                center_distance = math.hypot(element.x - x, element.y - y)
                direction = np.array([math.cos(element.angle_rad), math.sin(element.angle_rad)])
                side = np.array([-direction[1], direction[0]])
                inner = np.array([element.x, element.y]) + element.inner_offset_x * direction + element.inner_offset_y * side
                inner_distance = math.hypot(inner[0] - x, inner[1] - y)
                local = np.array([x - element.x, y - element.y])
                window_distances: list[float] = []
                for index in range(max(1, min(int(element.window_count), 4))):
                    window_angle = element.angle_rad + index * math.pi / 2.0
                    window_direction = np.array([math.cos(window_angle), math.sin(window_angle)])
                    window_side = np.array([-window_direction[1], window_direction[0]])
                    along = float(np.dot(local, window_direction))
                    across = float(np.dot(local, window_side))
                    window_dx = max(element.radius - along, 0.0, along - (element.radius + 40.0))
                    window_dy = max(abs(across) - 18.0, 0.0)
                    window_distances.append(math.hypot(window_dx, window_dy))
                window_distance = min(window_distances)
                options.append((
                    min(abs(center_distance - element.radius), abs(inner_distance - element.inner_radius), window_distance),
                    element,
                    "center",
                ))
                continue
            options.append((math.hypot(element.x - x, element.y - y), element, "center"))
            if element.kind == "block":
                first, second = block_endpoints(element)
                segment = second - first
                denominator = float(np.dot(segment, segment))
                projection = 0.0 if denominator < 1e-12 else min(
                    max(float(np.dot(np.array([x, y]) - first, segment)) / denominator, 0.0), 1.0
                )
                body_distance = float(np.linalg.norm(np.array([x, y]) - (first + projection * segment)))
                half_width = max(min(element.aperture / 2.0, 12.0), 3.0)
                options.append((max(body_distance - half_width, 0.0), element, "center"))
            if element.kind == "horn":
                direction = np.array([math.cos(element.angle_rad), math.sin(element.angle_rad)])
                side = np.array([-direction[1], direction[0]])
                local = np.array([x - element.x, y - element.y])
                along, across = float(np.dot(local, direction)), abs(float(np.dot(local, side)))
                back = max(element.horn_length - element.pcl_depth, 1.0)
                if -back <= along <= element.pcl_depth:
                    fraction = (along + back) / (back + element.pcl_depth)
                    half_width = 0.5 + fraction * max(element.aperture / 2.0 - 0.5, 0.0)
                    if across <= half_width + tolerance:
                        options.append((max(across - half_width, 0.0), element, "center"))
            if element.kind == "elliptical_mirror":
                options.extend((
                    (math.hypot(element.focus1_x - x, element.focus1_y - y), element, "focus1"),
                    (math.hypot(element.focus2_x - x, element.focus2_y - y), element, "focus2"),
                ))
            if element.uid in self.edge_handles:
                e1, e2 = self.edge_handles[element.uid]
                segment = e2 - e1
                denominator = float(np.dot(segment, segment))
                projection = 0.0 if denominator < 1e-12 else min(
                    max(float(np.dot(np.array([x, y]) - e1, segment)) / denominator, 0.0),
                    1.0,
                )
                body_distance = float(np.linalg.norm(np.array([x, y]) - (e1 + projection * segment)))
                options.append((body_distance + 0.25 * tolerance, element, "center"))
                options.extend(((float(np.linalg.norm(e1 - [x, y])), element, "edge1"),
                                (float(np.linalg.norm(e2 - [x, y])), element, "edge2")))
        if not options:
            return None
        distance, element, handle = min(options, key=lambda value: value[0])
        return (element, handle) if distance <= tolerance else None

    def _data_pos(self, event: QMouseEvent) -> QPointF:
        return self.getViewBox().mapSceneToView(self.mapToScene(event.position().toPoint()))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        point = self._data_pos(event)
        additive = bool(event.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier))
        if event.button() == Qt.MouseButton.RightButton:
            picked = self._pick(point.x(), point.y())
            if picked is not None:
                element, _handle = picked
                if element.uid not in self.selected_uids:
                    self.on_select(element.uid, additive)
                self.selected_uid = element.uid
                self.on_edit_started()
                self.edit_in_progress = True
                if element.kind == "target" and len(self.selected_uids) == 1:
                    self._start_target_orbit(element, point.x(), point.y())
                else:
                    self._start_group_rotation(point.x(), point.y())
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        if self.rotation_mode and self.selected_uids:
            self._start_group_rotation(point.x(), point.y())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        picked = self._pick(point.x(), point.y())
        if picked is not None:
            element, handle = picked
            moving_selected_group = (
                not additive
                and element.uid in self.selected_uids
                and len(self.selected_uids) > 1
                and handle in ("center", "ruler_body")
            )
            if not moving_selected_group:
                self.on_select(element.uid, additive)
            self.selected_uid = element.uid
            self.on_edit_started()
            self.edit_in_progress = True
            if moving_selected_group:
                self._start_group_translation(point.x(), point.y())
                self.drag = (element.uid, "translate_group")
            else:
                self.drag = (element.uid, handle)
            if element.kind == "ruler" and handle == "ruler_body" and not moving_selected_group:
                self.ruler_drag_snapshot = {
                    "cursor": (point.x(), point.y()),
                    "first": (element.x, element.y),
                    "second": (element.target_x, element.target_y),
                }
        else:
            self._rubber_origin = event.position().toPoint()
            self._rubber.setGeometry(QRect(self._rubber_origin, self._rubber_origin))
            self._rubber.show()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        point = self._data_pos(event)
        if self.target_orbit is not None:
            self._orbit_target_to(point.x(), point.y())
            event.accept(); return
        if self.group_rotation is not None:
            self._rotate_group_to(point.x(), point.y())
            event.accept(); return
        if self.drag is not None:
            self._move_dragged(point.x(), point.y())
            event.accept(); return
        if self._rubber_origin is not None:
            self._rubber.setGeometry(QRect(self._rubber_origin, event.position().toPoint()).normalized())
            event.accept(); return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._rubber_origin is not None:
                rect = self._rubber.geometry().normalized()
                self._rubber.hide(); self._rubber_origin = None
                top_left = self.getViewBox().mapSceneToView(self.mapToScene(rect.topLeft()))
                bottom_right = self.getViewBox().mapSceneToView(self.mapToScene(rect.bottomRight()))
                xmin, xmax = sorted((top_left.x(), bottom_right.x()))
                ymin, ymax = sorted((top_left.y(), bottom_right.y()))
                selected = {
                    e.uid for e in self.get_system().elements
                    if (
                        (
                            e.kind != "cryostat"
                            and (
                                xmin <= e.x <= xmax and ymin <= e.y <= ymax
                                or e.kind == "ruler" and e.target_x is not None and e.target_y is not None
                                and xmin <= e.target_x <= xmax and ymin <= e.target_y <= ymax
                            )
                        )
                        or (
                            e.kind == "cryostat"
                            and xmin <= e.x - e.radius and e.x + e.radius <= xmax
                            and ymin <= e.y - e.radius and e.y + e.radius <= ymax
                        )
                    )
                }
                additive = bool(event.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier))
                self.on_box_select(selected, additive)
            self.drag = None
            self.group_rotation = None
            self.group_translation = None
            self.target_orbit = None
            self.ruler_drag_snapshot = None
            self._flush_change()
            if self.edit_in_progress:
                self.edit_in_progress = False
                self.on_edit_finished()
            self.setCursor(Qt.CursorShape.OpenHandCursor if self.rotation_mode else Qt.CursorShape.CrossCursor)
            event.accept(); return
        elif event.button() == Qt.MouseButton.RightButton:
            self.group_rotation = None
            self.target_orbit = None
            self._flush_change()
            if self.edit_in_progress:
                self.edit_in_progress = False
                self.on_edit_finished()
            self.setCursor(Qt.CursorShape.CrossCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.on_delete()
            event.accept()
            return
        movement = {
            Qt.Key.Key_Left: (-1.0, 0.0),
            Qt.Key.Key_Right: (1.0, 0.0),
            Qt.Key.Key_Up: (0.0, 1.0),
            Qt.Key.Key_Down: (0.0, -1.0),
        }.get(event.key())
        if movement is not None and self.selected_uids:
            self.on_edit_started()
            self.translate_selected_by(*movement)
            self.on_edit_finished()
            event.accept()
            return
        super().keyPressEvent(event)

    def _start_group_rotation(self, x: float, y: float) -> None:
        members = [e for e in self.get_system().elements if e.uid in self.selected_uids]
        if not members:
            return
        center = np.mean([self._element_center(e) for e in members], axis=0)
        if math.hypot(x - center[0], y - center[1]) < 1e-6:
            x += 1.0
        snapshots = {
            e.uid: {
                "x": e.x, "y": e.y, "angle_deg": e.angle_deg,
                "focus1": (e.focus1_x, e.focus1_y), "focus2": (e.focus2_x, e.focus2_y),
                "target": (e.target_x, e.target_y), "inner_offset": (e.inner_offset_x, e.inner_offset_y),
            } for e in members
        }
        self.group_rotation = {
            "center": center,
            "start_angle": math.atan2(y - center[1], x - center[0]),
            "snapshots": snapshots,
        }

    def _start_group_translation(self, x: float, y: float) -> None:
        members = [e for e in self.get_system().elements if e.uid in self.selected_uids]
        self.group_translation = {
            "cursor": (x, y),
            "snapshots": {
                e.uid: {
                    "x": e.x,
                    "y": e.y,
                    "focus1": (e.focus1_x, e.focus1_y),
                    "focus2": (e.focus2_x, e.focus2_y),
                    "target": (e.target_x, e.target_y),
                }
                for e in members
            },
        }

    def _start_target_orbit(self, target: OpticalElement, x: float, y: float) -> None:
        predecessor = target_predecessor(self.get_system(), target)
        if predecessor is None:
            return
        dx, dy = target.x - predecessor.x, target.y - predecessor.y
        self.target_orbit = {
            "uid": target.uid,
            "pivot": np.array([predecessor.x, predecessor.y], dtype=float),
            "distance": math.hypot(dx, dy),
            "target_angle": math.atan2(dy, dx),
            "pointer_angle": math.atan2(y - predecessor.y, x - predecessor.x),
        }

    def _orbit_target_to(self, x: float, y: float) -> None:
        state = self.target_orbit
        if state is None:
            return
        target = self.get_system().element(state["uid"])
        if target is None:
            return
        pivot = state["pivot"]
        delta = math.atan2(y - pivot[1], x - pivot[0]) - state["pointer_angle"]
        angle = state["target_angle"] + delta
        target.relative_distance = state["distance"]
        target.relative_angle_deg = math.degrees(angle)
        target.position_mode = "relative"
        target.x = pivot[0] + state["distance"] * math.cos(angle)
        target.y = pivot[1] + state["distance"] * math.sin(angle)
        enforce_links(self.get_system())
        self._queue_change()

    def _translate_group_to(self, x: float, y: float) -> None:
        state = self.group_translation
        if state is None:
            return
        start_x, start_y = state["cursor"]
        dx, dy = x - start_x, y - start_y
        snapshots = state["snapshots"]
        for member in self.get_system().elements:
            snapshot = snapshots.get(member.uid)
            if snapshot is None:
                continue
            member.x, member.y = snapshot["x"] + dx, snapshot["y"] + dy
            for prefix in ("focus1", "focus2", "target"):
                point = snapshot[prefix]
                if point[0] is not None and point[1] is not None:
                    setattr(member, f"{prefix}_x", point[0] + dx)
                    setattr(member, f"{prefix}_y", point[1] + dy)
        for member in self.get_system().elements:
            if member.uid in snapshots and member.kind == "target":
                update_target_relative_from_position(self.get_system(), member)
        enforce_dependents(self.get_system())
        self._queue_change()

    @staticmethod
    def _rotated_point(point: tuple[float, float], center: np.ndarray, angle: float) -> tuple[float, float]:
        vector = np.array(point, dtype=float) - center
        c, s = math.cos(angle), math.sin(angle)
        result = center + np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])
        return float(result[0]), float(result[1])

    def _rotate_group_to(self, x: float, y: float) -> None:
        state = self.group_rotation
        if state is None:
            return
        center = state["center"]
        angle = math.atan2(y - center[1], x - center[0]) - state["start_angle"]
        self._apply_group_rotation(angle, state["snapshots"], center)

    def rotate_selected_by(self, angle_deg: float) -> None:
        members = [e for e in self.get_system().elements if e.uid in self.selected_uids]
        if not members:
            return
        center = np.mean([self._element_center(e) for e in members], axis=0)
        snapshots = {
            e.uid: {"x": e.x, "y": e.y, "angle_deg": e.angle_deg,
                    "focus1": (e.focus1_x, e.focus1_y), "focus2": (e.focus2_x, e.focus2_y),
                    "target": (e.target_x, e.target_y), "inner_offset": (e.inner_offset_x, e.inner_offset_y)}
            for e in members
        }
        self._apply_group_rotation(math.radians(angle_deg), snapshots, center)
        self._flush_change()

    def translate_selected_by(self, dx: float, dy: float) -> None:
        """Translate the selected objects and their auxiliary geometry as a rigid group."""
        selected = set(self.selected_uids)
        if not selected or (abs(dx) < 1e-15 and abs(dy) < 1e-15):
            return
        for member in self.get_system().elements:
            if member.uid not in selected:
                continue
            member.x += dx
            member.y += dy
            for prefix in ("focus1", "focus2", "target"):
                x_value = getattr(member, f"{prefix}_x")
                y_value = getattr(member, f"{prefix}_y")
                if x_value is not None and y_value is not None:
                    setattr(member, f"{prefix}_x", x_value + dx)
                    setattr(member, f"{prefix}_y", y_value + dy)
        for member in self.get_system().elements:
            if member.uid in selected and member.kind == "target":
                update_target_relative_from_position(self.get_system(), member)
        enforce_dependents(self.get_system())
        self._flush_change()

    def _apply_group_rotation(self, angle: float, snapshots: dict[str, dict[str, object]], center: np.ndarray) -> None:
        rotating_assembly = len(snapshots) > 1
        for member in self.get_system().elements:
            if member.uid not in snapshots:
                continue
            snapshot = snapshots[member.uid]
            member.x, member.y = self._rotated_point((snapshot["x"], snapshot["y"]), center, angle)
            member.angle_deg = snapshot["angle_deg"] + math.degrees(angle)
            for prefix in ("focus1", "focus2", "target"):
                if prefix == "target" and not rotating_assembly and member.kind not in ("elliptical_mirror", "ruler"):
                    continue
                point = snapshot[prefix]
                if point[0] is not None and point[1] is not None:
                    nx, ny = self._rotated_point(point, center, angle)
                    setattr(member, f"{prefix}_x", nx); setattr(member, f"{prefix}_y", ny)
        for member in self.get_system().elements:
            if member.uid in snapshots and member.kind == "target":
                update_target_relative_from_position(self.get_system(), member)
        enforce_dependents(self.get_system())
        self._queue_change()

    def _move_dragged(self, x: float, y: float) -> None:
        uid, handle = self.drag
        if handle == "translate_group":
            self._translate_group_to(x, y)
            return
        element = next((e for e in self.get_system().elements if e.uid == uid), None)
        if element is None:
            return
        if handle in ("center", "focus1", "focus2"):
            move_with_constraints(self.get_system(), element, x, y, handle)
        elif handle in ("ruler1", "ruler2"):
            move_with_constraints(self.get_system(), element, x, y, handle)
        elif handle == "ruler_body" and self.ruler_drag_snapshot is not None:
            if not ruler_attachments(self.get_system(), element):
                start_x, start_y = self.ruler_drag_snapshot["cursor"]
                dx, dy = x - start_x, y - start_y
                first_x, first_y = self.ruler_drag_snapshot["first"]
                second_x, second_y = self.ruler_drag_snapshot["second"]
                element.x, element.y = first_x + dx, first_y + dy
                element.target_x, element.target_y = second_x + dx, second_y + dy
        elif handle in ("edge1", "edge2"):
            center = np.array([element.x, element.y], dtype=float)
            cursor = np.array([x, y], dtype=float)
            if element.kind == "elliptical_mirror":
                element.aperture = max(1.0, 2.0 * float(np.linalg.norm(cursor - center)))
            else:
                element.aperture = max(1.0, 2.0 * abs(float(np.dot(cursor - center, tangent(element)))))
        enforce_links(self.get_system())
        self._queue_change()
