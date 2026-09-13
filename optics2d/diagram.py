from __future__ import annotations

from .i18n import tr

from collections.abc import Callable

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QMessageBox,
)

from .model import ObjectLink, OpticalElement, OpticalSystem
from .physics import segment_waist, trace_all_beams
from .visualization import KIND_LABELS


class PortItem(QGraphicsEllipseItem):
    def __init__(self, node: "NodeItem", is_output: bool) -> None:
        super().__init__(-6, -6, 12, 12, node)
        self.node = node
        self.is_output = is_output
        self.setBrush(QBrush(QColor("#db6d28") if is_output else QColor("#2878b5")))
        self.setPen(QPen(QColor("#ffffff"), 1.2))
        self.setZValue(4)

    def mousePressEvent(self, event: object) -> None:
        if self.is_output:
            self.node.diagram.start_connection(self)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: object) -> None:
        if self.is_output:
            self.node.diagram.update_connection(event.scenePos())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: object) -> None:
        if self.is_output:
            self.node.diagram.finish_connection(event.scenePos())
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class AimButtonItem(QGraphicsRectItem):
    def __init__(self, node: "NodeItem") -> None:
        super().__init__(0, 0, 139, 22, node)
        self.node = node
        self.setPos(8, 52)
        self.setBrush(QBrush(QColor("#e8eef4")))
        self.setPen(QPen(QColor("#a8b4bf"), 1.0))
        label = QGraphicsSimpleTextItem(tr('→ aim at next'), self)
        label.setPos(18, 3)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event: object) -> None:
        self.node.diagram.on_aim(self.node.element_uid)
        event.accept()


class NodeItem(QGraphicsRectItem):
    WIDTH = 155.0
    HEIGHT = 82.0

    def __init__(self, diagram: "DiagramView", element: OpticalElement) -> None:
        super().__init__(0, 0, self.WIDTH, self.HEIGHT)
        self.diagram = diagram
        self.element_uid = element.uid
        self.setBrush(QBrush(QColor("#f4f6f8")))
        self.setPen(QPen(QColor("#8292a2"), 1.2))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        title = QGraphicsSimpleTextItem(element.name, self)
        title.setPos(10, 8)
        subtitle = QGraphicsSimpleTextItem(KIND_LABELS[element.kind], self)
        subtitle.setBrush(QBrush(QColor("#66717d")))
        subtitle.setPos(10, 30)
        system = diagram.get_system()
        incoming = sum(link.target_uid == element.uid and link.enabled for link in system.links)
        outgoing = sum(link.source_uid == element.uid and link.enabled for link in system.links)
        degree = QGraphicsSimpleTextItem(tr('in:{0}  out:{1}', incoming, outgoing), self)
        degree.setBrush(QBrush(QColor("#59636e")))
        degree.setPos(88, 30)
        self.input_port = PortItem(self, False)
        self.output_port = PortItem(self, True)
        self.aim_button = AimButtonItem(self)
        self.input_port.setPos(0, self.HEIGHT / 2)
        self.output_port.setPos(self.WIDTH, self.HEIGHT / 2)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: object) -> object:
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.diagram.node_moved(self)
        return result

    def mousePressEvent(self, event: object) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.diagram.on_edit_started()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: object) -> None:
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.diagram.on_edit_finished()


class LinkItem(QGraphicsPathItem):
    def __init__(self, link_uid: str, waist_text: str = "") -> None:
        super().__init__()
        self.link_uid = link_uid
        self.setPen(QPen(QColor("#44596d"), 2.0))
        self.setZValue(-1)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.waist_label = QGraphicsSimpleTextItem(waist_text, self)
        self.waist_label.setBrush(QBrush(QColor("#9b1c31")))
        self.waist_label.setZValue(2)
        self.waist_label.setVisible(bool(waist_text))

    def update_path(self, start: QPointF, end: QPointF, offset: float = 0.0) -> None:
        path = QPainterPath(start)
        distance = max(45.0, abs(end.x() - start.x()) * 0.5)
        dx, dy = end.x() - start.x(), end.y() - start.y()
        length = max((dx * dx + dy * dy) ** 0.5, 1.0)
        normal = QPointF(-dy / length * offset, dx / length * offset)
        path.cubicTo(
            start + QPointF(distance, 0) + normal,
            end - QPointF(distance, 0) + normal,
            end,
        )
        self.setPath(path)
        middle = path.pointAtPercent(0.5)
        self.waist_label.setPos(middle + QPointF(-55, 8))


class DiagramView(QGraphicsView):
    def __init__(
        self,
        get_system: Callable[[], OpticalSystem],
        on_links_changed: Callable[..., None],
        on_select: Callable[[str], None],
        on_aim: Callable[[str], None],
        on_edit_started: Callable[[], None],
        on_edit_finished: Callable[[], None],
        get_selected_uids: Callable[[], set[str]],
    ) -> None:
        super().__init__()
        self.get_system = get_system
        self.on_links_changed = on_links_changed
        self.on_select = on_select
        self.on_aim = on_aim
        self.on_edit_started = on_edit_started
        self.on_edit_finished = on_edit_finished
        self.get_selected_uids = get_selected_uids
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.nodes: dict[str, NodeItem] = {}
        self.links: dict[str, LinkItem] = {}
        self.link_offsets: dict[str, float] = {}
        self.pending_port: PortItem | None = None
        self.pending_line: QGraphicsPathItem | None = None
        self.connection_selection: set[str] = set()
        self._rebuilding = False
        self.scene().selectionChanged.connect(self._selection_changed)

    def rebuild(self) -> None:
        self._rebuilding = True
        self.scene().clear()
        self.nodes.clear()
        self.links.clear()
        self.link_offsets.clear()
        system = self.get_system()
        for index, element in enumerate(system.elements):
            node = NodeItem(self, element)
            position = system.diagram_positions.get(
                element.uid,
                [20.0 + (index % 4) * 195, 20.0 + (index // 4) * 105],
            )
            node.setPos(float(position[0]), float(position[1]))
            self.scene().addItem(node)
            self.nodes[element.uid] = node
        valid_links = [
            link for link in system.links
            if link.source_uid in self.nodes and link.target_uid in self.nodes
        ]
        parallel_groups: dict[tuple[str, str], list[ObjectLink]] = {}
        for link in valid_links:
            key = tuple(sorted((link.source_uid, link.target_uid)))
            parallel_groups.setdefault(key, []).append(link)
        for group in parallel_groups.values():
            total = len(group)
            for index, link in enumerate(group):
                self.link_offsets[link.uid] = (index - (total - 1) / 2.0) * 20.0
                item = LinkItem(link.uid, self._waist_text(link))
                self.scene().addItem(item)
                self.links[link.uid] = item
                self._update_link(link)
        self.scene().setSceneRect(self.scene().itemsBoundingRect().adjusted(-40, -40, 80, 80))
        self._rebuilding = False

    def _waist_text(self, link: ObjectLink, traces: object | None = None) -> str:
        labels: list[str] = []
        seen: set[tuple[float, float, float]] = set()
        for beam in traces if traces is not None else trace_all_beams(self.get_system()):
            for segment in beam.result.segments:
                if segment.source_uid != link.source_uid or segment.hit_uid != link.target_uid:
                    continue
                waist = segment_waist(segment)
                if waist is None:
                    continue
                key = (round(beam.frequency_ghz, 6), round(float(waist.position[0]), 6), round(float(waist.position[1]), 6))
                if key in seen:
                    continue
                seen.add(key)
                labels.append(
                    tr('Waist {0:g} GHz: center ({1:.2f}; {2:.2f}) mm', beam.frequency_ghz, waist.position[0], waist.position[1])
                )
        return "\n".join(labels)

    def refresh_waists(self, traces: object | None = None) -> None:
        links_by_uid = {link.uid: link for link in self.get_system().links}
        for uid, item in self.links.items():
            link = links_by_uid.get(uid)
            if link is None:
                continue
            text = self._waist_text(link, traces)
            item.waist_label.setText(text)
            item.waist_label.setVisible(bool(text))

    def node_moved(self, node: NodeItem) -> None:
        if self._rebuilding:
            return
        self.get_system().diagram_positions[node.element_uid] = [node.pos().x(), node.pos().y()]
        for link in self.get_system().links:
            if link.source_uid == node.element_uid or link.target_uid == node.element_uid:
                self._update_link(link)

    def _update_link(self, link: ObjectLink) -> None:
        item = self.links.get(link.uid)
        source, target = self.nodes.get(link.source_uid), self.nodes.get(link.target_uid)
        if item is None or source is None or target is None:
            return
        item.update_path(
            source.output_port.scenePos(),
            target.input_port.scenePos(),
            self.link_offsets.get(link.uid, 0.0),
        )

    def start_connection(self, port: PortItem) -> None:
        self.pending_port = port
        self.connection_selection = set(self.get_selected_uids())
        self.pending_line = QGraphicsPathItem()
        self.pending_line.setPen(QPen(QColor("#db6d28"), 2, Qt.PenStyle.DashLine))
        self.scene().addItem(self.pending_line)

    def update_connection(self, scene_pos: QPointF) -> None:
        if self.pending_port is None or self.pending_line is None:
            return
        start = self.pending_port.scenePos()
        path = QPainterPath(start)
        path.cubicTo(start + QPointF(50, 0), scene_pos - QPointF(50, 0), scene_pos)
        self.pending_line.setPath(path)

    def finish_connection(self, scene_pos: QPointF) -> None:
        source = self.pending_port
        target: PortItem | None = None
        for item in self.scene().items(scene_pos):
            if isinstance(item, PortItem) and not item.is_output:
                target = item
                break
        if self.pending_line is not None:
            self.scene().removeItem(self.pending_line)
        self.pending_port = None
        self.pending_line = None
        if source is None or target is None:
            return
        if source.node.element_uid == target.node.element_uid:
            return
        self.on_edit_started()
        try:
            link = self.get_system().connect(source.node.element_uid, target.node.element_uid)
        except ValueError as exc:
            QMessageBox.warning(self, tr('Invalid connection'), str(exc))
            self.on_edit_finished()
            return
        self.on_links_changed(link, self.connection_selection)
        self.connection_selection.clear()
        self.rebuild()
        self.on_edit_finished()

    def keyPressEvent(self, event: object) -> None:
        if event.key() == Qt.Key.Key_Delete:
            selected_links = [item for item in self.scene().selectedItems() if isinstance(item, LinkItem)]
            if selected_links:
                self.on_edit_started()
                selected_uids = {item.link_uid for item in selected_links}
                self.get_system().links = [link for link in self.get_system().links if link.uid not in selected_uids]
                self.on_links_changed()
                self.rebuild()
                self.on_edit_finished()
                return
        super().keyPressEvent(event)

    def _selection_changed(self) -> None:
        nodes = [item for item in self.scene().selectedItems() if isinstance(item, NodeItem)]
        if nodes:
            self.on_select(nodes[0].element_uid)
