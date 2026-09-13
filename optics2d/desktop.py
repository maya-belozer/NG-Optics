from __future__ import annotations

from .i18n import tr, LocalizedLabels, translator
from .version import APP_NAME, APP_TITLE, __version__

import math
import sys

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .constraints import attach_assembly_to_block, enforce_links, target_predecessor
from .diagram import DiagramView
from .model import FrequencyChannel, OpticalElement, OpticalSystem, default_band6_system
from .physics import (
    ellipse_effective_focal_length,
    ellipse_geometry,
    ellipse_vertices,
    orient_mirror_to_target,
    trace_all_beams,
    unit,
)
from .visualization import KIND_LABELS, SceneView


FIELD_LABELS = LocalizedLabels({
    'name': 'Name',
    'x': 'Point/center X, mm',
    'y': 'Point/center Y, mm',
    'angle_deg': 'Angle, °',
    'aperture': 'Aperture, mm',
    'focal_length': 'Focal length, mm',
    'target_x': 'Target X, mm',
    'target_y': 'Target Y, mm',
    'frequency_ghz': 'Frequency, GHz',
    'waist_radius': 'Waist radius w₀, mm',
    'pcl_depth': 'PCL from aperture, mm',
    'horn_length': 'Horn length, mm',
    'focus1_x': 'Focus F1 X, mm',
    'focus1_y': 'Focus F1 Y, mm',
    'focus2_x': 'Focus F2 X, mm',
    'focus2_y': 'Focus F2 Y, mm',
    'radius': 'Outer radius, mm',
    'inner_radius': 'Inner radius, mm',
    'inner_offset_x': 'Inner X offset',
    'inner_offset_y': 'Inner Y offset',
    'window_count': 'Number of windows (1–4)',
    'relative_distance': 'Distance from previous object, mm',
    'relative_angle_deg': 'Angle from previous object, °',
    'block_length': 'Block length, mm',
})
FIELDS_BY_KIND = {
    "horn": ("name", "x", "y", "angle_deg", "aperture", "waist_radius", "pcl_depth", "horn_length"),
    "lens": ("name", "x", "y", "angle_deg", "aperture", "focal_length"),
    "plane_mirror": ("name", "x", "y", "angle_deg", "aperture", "target_x", "target_y"),
    "curved_mirror": ("name", "x", "y", "angle_deg", "aperture", "focal_length", "target_x", "target_y"),
    "elliptical_mirror": ("name", "x", "y", "aperture", "focus1_x", "focus1_y", "focus2_x", "focus2_y"),
    "cryostat": ("name", "x", "y", "angle_deg", "radius", "inner_radius", "inner_offset_x", "inner_offset_y", "window_count"),
    "ruler": ("name", "x", "y", "target_x", "target_y"),
    "target": ("name", "x", "y", "relative_distance", "relative_angle_deg"),
    "block": ("name", "x", "y", "angle_deg", "block_length", "aperture"),
}

RULER_FIELD_LABELS = LocalizedLabels({
    'x': 'Point A: X, mm',
    'y': 'Point A: Y, mm',
    'target_x': 'Point B: X, mm',
    'target_y': 'Point B: Y, mm',
})


class OpticsDesktop(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1480, 880)
        self.system = default_band6_system()
        self.selected_uid: str | None = None
        self.selected_uids: set[str] = set()
        self.field_editors: dict[str, QLineEdit] = {}
        self._updating_list = False
        self._undo_stack: list[dict[str, object]] = []
        self._redo_stack: list[dict[str, object]] = []
        self._pending_history: dict[str, object] | None = None
        self._restoring_history = False
        self._build_ui()
        self._refresh_list(select_first=True)
        self.scene.draw(reset_view=True)
        self._update_info()
        self._update_history_actions()

    def _build_ui(self) -> None:
        language_toolbar = QToolBar(tr('Language'), self)
        language_toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, language_toolbar)
        language_toolbar.addWidget(QLabel(tr('Language') + ': '))
        self.language_combo = QComboBox()
        for code, catalog in translator.catalogs.items():
            self.language_combo.addItem(catalog['name'], code)
        self.language_combo.setCurrentIndex(self.language_combo.findData(translator.code))
        self.language_combo.currentIndexChanged.connect(self._language_selected)
        language_toolbar.addWidget(self.language_combo)
        load_language = QAction(tr('Load language file…'), self)
        load_language.triggered.connect(self.load_language_file)
        language_toolbar.addAction(load_language)
        toolbar = QToolBar(tr('Objects'), self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.undo_action = QAction(tr('Undo'), self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self.undo)
        toolbar.addAction(self.undo_action)
        self.redo_action = QAction(tr('Redo'), self)
        self.redo_action.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        self.redo_action.triggered.connect(self.redo)
        toolbar.addAction(self.redo_action)
        toolbar.addSeparator()
        additions = (
            (tr('+ Horn'), "horn"), (tr('+ Lens'), "lens"), (tr('+ Plane mirror'), "plane_mirror"),
            (tr('+ Focusing mirror'), "curved_mirror"), (tr('+ Elliptical mirror'), "elliptical_mirror"),
            (tr('+ Cryostat'), "cryostat"), (tr('+ Ruler'), "ruler"), (tr('+ Target'), "target"),
            (tr('+ Block'), "block"),
        )
        for label, kind in additions:
            action = QAction(label, self)
            action.triggered.connect(lambda _checked=False, value=kind: self.add_element(value))
            toolbar.addAction(action)
        toolbar.addSeparator()
        for label, callback in (
            (tr('Delete'), self.delete_selected), (tr('Band 6 example'), self.load_default),
            (tr('Insert assembly'), self.insert_optical_assembly),
            (tr('Open JSON'), self.open_json), (tr('Save JSON'), self.save_json),
        ):
            action = QAction(label, self)
            action.triggered.connect(callback)
            toolbar.addAction(action)

        toolbar.addSeparator()
        rotate_left = QAction("↶ 15°", self)
        rotate_left.triggered.connect(lambda: self.rotate_selected(-15.0))
        toolbar.addAction(rotate_left)
        rotate_right = QAction("↷ 15°", self)
        rotate_right.triggered.connect(lambda: self.rotate_selected(15.0))
        toolbar.addAction(rotate_right)

        root_splitter = QSplitter(Qt.Orientation.Horizontal)
        left_splitter = QSplitter(Qt.Orientation.Vertical)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_help = QLabel(
            tr('Left drag: move object/selection · rectangle: select multiple · Shift: add · Right drag on object: rotate · wheel: zoom · middle drag: pan')
        )
        plot_help.setWordWrap(True)
        plot_help.setStyleSheet(
            "padding: 5px 8px; color: #102a43; background: #dbeafe; "
            "font-weight: 600; border-bottom: 1px solid #93c5fd;"
        )
        plot_layout.addWidget(plot_help)
        self.scene = SceneView(
            lambda: self.system,
            self.select_uid,
            self.model_changed,
            self.select_uids,
            self.delete_selected,
            self.begin_history_action,
            self.commit_history_action,
        )
        plot_layout.addWidget(self.scene, 1)
        left_splitter.addWidget(plot_panel)
        left_splitter.addWidget(self._build_diagram_panel())
        left_splitter.setStretchFactor(0, 4)
        left_splitter.setStretchFactor(1, 2)
        left_splitter.setSizes([590, 280])

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(6, 6, 6, 6)
        self.object_list = QListWidget()
        self.object_list.setMinimumHeight(135)
        self.object_list.setMaximumHeight(210)
        self.object_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.object_list.currentRowChanged.connect(self._row_selected)
        panel_layout.addWidget(QLabel(tr('Optical objects')))
        panel_layout.addWidget(self.object_list)

        translate_row = QHBoxLayout()
        translate_row.addWidget(QLabel(tr('Translate assembly, mm:')))
        self.translate_dx_editor = QLineEdit("0")
        self.translate_dx_editor.setPlaceholderText("dx")
        self.translate_dx_editor.setMaximumWidth(72)
        self.translate_dy_editor = QLineEdit("0")
        self.translate_dy_editor.setPlaceholderText("dy")
        self.translate_dy_editor.setMaximumWidth(72)
        translate_button = QPushButton(tr('Translate'))
        translate_button.clicked.connect(self.translate_selected_from_fields)
        self.translate_dx_editor.returnPressed.connect(self.translate_selected_from_fields)
        self.translate_dy_editor.returnPressed.connect(self.translate_selected_from_fields)
        translate_row.addWidget(QLabel("dx"))
        translate_row.addWidget(self.translate_dx_editor)
        translate_row.addWidget(QLabel("dy"))
        translate_row.addWidget(self.translate_dy_editor)
        translate_row.addWidget(translate_button)
        panel_layout.addLayout(translate_row)
        panel_layout.addWidget(QLabel(tr('Arrow keys on the 2D plot: move by 1 mm')))

        self.property_widget = QWidget()
        self.property_form = QFormLayout(self.property_widget)
        property_scroll = QScrollArea()
        property_scroll.setWidgetResizable(True)
        property_scroll.setWidget(self.property_widget)
        panel_layout.addWidget(property_scroll, 1)

        panel_layout.addWidget(QLabel(tr('Gaussian beam display')))
        self.render_bounds = QCheckBox(tr('Show boundaries ±N·w(z)'))
        self.reflect_bounds = QCheckBox(tr('Reflect Gaussian beam boundaries'))
        self.limit_horn_motion = QCheckBox(tr('Constrain horn motion to its connection line'))
        self.limit_plane_motion = QCheckBox(tr('Constrain plane mirror motion to its connection line'))
        self.render_bounds.toggled.connect(self._system_toggles_changed)
        self.reflect_bounds.toggled.connect(self._system_toggles_changed)
        self.limit_horn_motion.toggled.connect(self._system_toggles_changed)
        self.limit_plane_motion.toggled.connect(self._system_toggles_changed)
        panel_layout.addWidget(self.render_bounds)
        panel_layout.addWidget(self.reflect_bounds)
        panel_layout.addWidget(self.limit_horn_motion)
        panel_layout.addWidget(self.limit_plane_motion)

        system_form = QFormLayout()
        self.max_path_editor = QLineEdit()
        self.beam_scale_editor = QLineEdit()
        system_form.addRow(tr('Max. path, mm'), self.max_path_editor)
        system_form.addRow(tr('Boundary level, ×w'), self.beam_scale_editor)
        panel_layout.addLayout(system_form)
        apply_system = QPushButton(tr('Apply tracing parameters'))
        apply_system.clicked.connect(self.apply_system)
        panel_layout.addWidget(apply_system)
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        panel_layout.addWidget(self.info_label)
        panel_layout.addWidget(QLabel(tr('Left drag: move M1/F1/F2/object · Right drag: rotate')))
        panel.setMinimumWidth(390)
        root_splitter.addWidget(left_splitter)
        root_splitter.addWidget(panel)
        root_splitter.setStretchFactor(0, 4)
        root_splitter.setStretchFactor(1, 1)
        root_splitter.setSizes([1060, 420])
        self.setCentralWidget(root_splitter)
        self._sync_system_controls()

    def _language_selected(self, _index):
        code = self.language_combo.currentData()
        if code != translator.code:
            self._change_language(code)

    def load_language_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr('Load language file…'), '', tr('Language file (*.json)'))
        if not path:
            return
        try:
            code = translator.load(path)
        except (OSError, ValueError, TypeError, UnicodeError) as exc:
            QMessageBox.critical(self, tr('Cannot load language'),
                                 tr('Invalid language file: {0}', str(exc)))
            return
        self._change_language(code)

    def _change_language(self, code):
        # Recreate translated views while keeping the model, history and pending edits.
        old = self.centralWidget()
        edits = [widget.text() for widget in old.findChildren(QLineEdit)]
        checks = [widget.isChecked() for widget in old.findChildren(QCheckBox)]
        combos = [widget.currentIndex() for widget in old.findChildren(QComboBox)]
        tables = [[[ (table.item(row, col).text(), table.item(row, col).checkState())
                     if table.item(row, col) else None
                     for col in range(table.columnCount())]
                    for row in range(table.rowCount())]
                  for table in old.findChildren(QTableWidget)]
        view_range = self.scene.view_range()
        translator.set_language(code)
        for toolbar in self.findChildren(QToolBar):
            self.removeToolBar(toolbar)
            for action in toolbar.actions():
                action.deleteLater()
            toolbar.deleteLater()
        old = self.takeCentralWidget()
        self._build_ui()
        self._refresh_list()
        self.scene.draw()
        self.scene.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)
        self._update_info()
        self._update_history_actions()
        for widget, value in zip(self.centralWidget().findChildren(QLineEdit), edits):
            widget.setText(value)
        for widget, value in zip(self.centralWidget().findChildren(QCheckBox), checks):
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)
        for widget, value in zip(self.centralWidget().findChildren(QComboBox), combos):
            widget.blockSignals(True)
            widget.setCurrentIndex(value)
            widget.blockSignals(False)
        for table, rows in zip(self.centralWidget().findChildren(QTableWidget), tables):
            table.blockSignals(True)
            table.setRowCount(len(rows))
            for row, cells in enumerate(rows):
                for col, cell in enumerate(cells):
                    if cell is not None:
                        item = table.item(row, col)
                        if item is None:
                            item = QTableWidgetItem()
                            table.setItem(row, col, item)
                        item.setText(cell[0])
                        item.setCheckState(cell[1])
            table.blockSignals(False)
        old.deleteLater()

    def _build_diagram_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 0)
        add_row = QHBoxLayout()
        add_row.addWidget(QLabel(tr('Connection blocks')))
        self.diagram_kind = QComboBox()
        for kind, label in KIND_LABELS.items():
            self.diagram_kind.addItem(label, kind)
        add_button = QPushButton(tr('Add block'))
        add_button.clicked.connect(lambda: self.add_element(self.diagram_kind.currentData()))
        add_row.addWidget(self.diagram_kind, 1)
        add_row.addWidget(add_button)
        layout.addLayout(add_row)
        help_label = QLabel(tr('Drag from an orange output to a blue input. Delete removes a connection.'))
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        self.diagram = DiagramView(
            lambda: self.system,
            self.links_changed,
            self.select_uid,
            self.aim_object_at_next,
            self.begin_history_action,
            self.commit_history_action,
            lambda: set(self.selected_uids),
        )
        layout.addWidget(self.diagram, 1)
        self.diagram.rebuild()
        return container

    def _history_state(self) -> dict[str, object]:
        return {
            "system": self.system.to_dict(),
            "selected_uid": self.selected_uid,
            "selected_uids": sorted(self.selected_uids),
        }

    def begin_history_action(self) -> None:
        if not self._restoring_history and self._pending_history is None:
            self._pending_history = self._history_state()

    def commit_history_action(self) -> None:
        before = self._pending_history
        self._pending_history = None
        current = self._history_state()
        if (
            self._restoring_history
            or before is None
            or before.get("system") == current.get("system")
        ):
            self._update_history_actions()
            return
        self._undo_stack.append(before)
        del self._undo_stack[:-10]
        self._redo_stack.clear()
        self._update_history_actions()

    def rollback_history_action(self) -> None:
        state = self._pending_history
        self._pending_history = None
        if state is not None:
            self._restore_history_state(state)
        self._update_history_actions()

    def _restore_history_state(self, state: dict[str, object]) -> None:
        self._restoring_history = True
        try:
            self.system = OpticalSystem.from_dict(state["system"])
            valid = {item.uid for item in self.system.elements}
            selected = {uid for uid in state.get("selected_uids", []) if uid in valid}
            selected_uid = state.get("selected_uid")
            self.selected_uid = selected_uid if selected_uid in valid else next(iter(selected), None)
            self.selected_uids = selected or ({self.selected_uid} if self.selected_uid else set())
            enforce_links(self.system)
            self._sync_system_controls()
            self._refresh_list(select_first=self.selected_uid is None)
            self.diagram.rebuild()
            self.scene.draw()
            self._update_info()
        finally:
            self._restoring_history = False

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._pending_history = None
        current = self._history_state()
        state = self._undo_stack.pop()
        self._redo_stack.append(current)
        del self._redo_stack[:-10]
        self._restore_history_state(state)
        self._update_history_actions()

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._pending_history = None
        current = self._history_state()
        state = self._redo_stack.pop()
        self._undo_stack.append(current)
        del self._undo_stack[:-10]
        self._restore_history_state(state)
        self._update_history_actions()

    def _update_history_actions(self) -> None:
        if hasattr(self, "undo_action"):
            self.undo_action.setEnabled(bool(self._undo_stack))
            self.redo_action.setEnabled(bool(self._redo_stack))

    def selected(self) -> OpticalElement | None:
        return next((item for item in self.system.elements if item.uid == self.selected_uid), None)

    def _clear_form(self) -> None:
        while self.property_form.rowCount():
            self.property_form.removeRow(0)
        self.field_editors.clear()

    def _build_property_form(self) -> None:
        self._clear_form()
        item = self.selected()
        if item is None:
            return
        if item.kind == "target":
            self.target_mode_combo = QComboBox()
            self.target_mode_combo.addItem(tr('X/Y coordinates'), "coordinates")
            self.target_mode_combo.addItem(tr('Distance and angle'), "relative")
            index = self.target_mode_combo.findData(item.position_mode)
            self.target_mode_combo.setCurrentIndex(max(index, 0))
            self.property_form.addRow(tr('Position mode'), self.target_mode_combo)
        for key in FIELDS_BY_KIND[item.kind]:
            editor = QLineEdit()
            value = getattr(item, key)
            editor.setText("" if value is None else f"{value:g}" if isinstance(value, float) else str(value))
            editor.returnPressed.connect(self.apply_properties)
            self.field_editors[key] = editor
            label = RULER_FIELD_LABELS.get(key, FIELD_LABELS[key]) if item.kind == "ruler" else FIELD_LABELS[key]
            self.property_form.addRow(label, editor)
        self.enabled_checkbox = QCheckBox(tr('Enabled'))
        self.enabled_checkbox.setChecked(item.enabled)
        self.property_form.addRow(self.enabled_checkbox)
        apply_button = QPushButton(tr('Apply'))
        apply_button.clicked.connect(self.apply_properties)
        self.property_form.addRow(apply_button)
        if item.kind == "horn":
            self.waveguide_info = QLabel(tr('Waveguide: X={0:.3f} mm; Y={1:.3f} mm', item.waveguide_x, item.waveguide_y))
            self.waveguide_info.setWordWrap(True)
            self.property_form.addRow(self.waveguide_info)
            aim = QPushButton(tr('Aim horn at connected mirror'))
            aim.clicked.connect(self.aim_horn_at_linked_mirror)
            self.property_form.addRow(aim)
            self._build_frequency_editor(item)
        if item.kind == "plane_mirror":
            aim = QPushButton(tr('Aim reflection at next target'))
            aim.clicked.connect(self.aim_plane_at_next_target)
            self.property_form.addRow(aim)
        if item.kind == "elliptical_mirror":
            self.ellipse_info = QLabel()
            self.ellipse_info.setWordWrap(True)
            self.ellipse_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.property_form.addRow(self.ellipse_info)
            self._update_ellipse_info(item)
        if item.kind == "ruler" and item.target_x is not None and item.target_y is not None:
            length = math.hypot(item.target_x - item.x, item.target_y - item.y)
            attachments = [
                other for link in self.system.links
                for other in (
                    [self.system.element(link.target_uid)] if link.source_uid == item.uid
                    else [self.system.element(link.source_uid)] if link.target_uid == item.uid
                    else []
                )
                if other is not None and other.kind != "ruler"
            ]
            info = QLabel(tr('Distance: {0:.3f} mm · attached objects: {1}/2', length, len({x.uid for x in attachments})))
            self.property_form.addRow(info)
        if item.kind == "target":
            predecessor = target_predecessor(self.system, item)
            predecessor_name = predecessor.name if predecessor is not None else tr('no incoming connection')
            self.property_form.addRow(QLabel(tr('Previous object: {0}', predecessor_name)))

    def _build_frequency_editor(self, horn: OpticalElement) -> None:
        visible_rows = max(5, len(horn.frequency_channels))
        self.frequency_table = QTableWidget(visible_rows, 2)
        self.frequency_table.setHorizontalHeaderLabels([tr('On'), tr('Frequency, GHz')])
        self.frequency_table.verticalHeader().setVisible(False)
        self.frequency_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.frequency_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.frequency_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        row_height = self.frequency_table.verticalHeader().defaultSectionSize()
        header_height = self.frequency_table.horizontalHeader().height()
        self.frequency_table.setFixedHeight(header_height + row_height * 5 + 4)
        for row, channel in enumerate(horn.frequency_channels):
            enabled = QTableWidgetItem()
            enabled.setFlags(enabled.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            enabled.setCheckState(Qt.CheckState.Checked if channel.enabled else Qt.CheckState.Unchecked)
            enabled.setData(Qt.ItemDataRole.UserRole, channel.uid)
            self.frequency_table.setItem(row, 0, enabled)
            self.frequency_table.setItem(row, 1, QTableWidgetItem(f"{channel.frequency_ghz:g}"))
        for row in range(len(horn.frequency_channels), visible_rows):
            enabled = QTableWidgetItem()
            enabled.setFlags(enabled.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            enabled.setCheckState(Qt.CheckState.Unchecked)
            self.frequency_table.setItem(row, 0, enabled)
            self.frequency_table.setItem(row, 1, QTableWidgetItem(""))
        controls = QWidget()
        row_layout = QHBoxLayout(controls)
        row_layout.setContentsMargins(0, 0, 0, 0)
        add = QPushButton(tr('+ Frequency'))
        remove = QPushButton(tr('− Frequency'))
        add.clicked.connect(self.add_frequency)
        remove.clicked.connect(self.remove_frequency)
        row_layout.addWidget(add)
        row_layout.addWidget(remove)
        self.property_form.addRow(QLabel(tr('Frequency channels')))
        self.property_form.addRow(self.frequency_table)
        self.property_form.addRow(controls)

    def add_frequency(self) -> None:
        horn = self.selected()
        if horn is None or horn.kind != "horn":
            return
        self.begin_history_action()
        base = horn.frequency_channels[-1].frequency_ghz if horn.frequency_channels else horn.frequency_ghz
        horn.frequency_channels.append(FrequencyChannel(base + 10.0, True))
        self._build_property_form()
        self.scene.draw()
        self.commit_history_action()

    def remove_frequency(self) -> None:
        horn = self.selected()
        if horn is None or horn.kind != "horn" or not hasattr(self, "frequency_table"):
            return
        row = self.frequency_table.currentRow()
        if row < 0 or row >= len(horn.frequency_channels) or len(horn.frequency_channels) <= 1:
            return
        self.begin_history_action()
        horn.frequency_channels.pop(row)
        self._build_property_form()
        self.scene.draw()
        self.commit_history_action()

    def _update_ellipse_info(self, item: OpticalElement) -> None:
        if not hasattr(self, "ellipse_info"):
            return
        try:
            geometry = ellipse_geometry(item)
            if geometry is None:
                raise ValueError(tr('Invalid geometry'))
            center, a, b, _ = geometry
            vertices = ellipse_vertices(item)
            if vertices is None:
                raise ValueError(tr('Invalid geometry'))
            major_plus, major_minus, minor_plus, minor_minus = vertices
            focal = ellipse_effective_focal_length(item)
            self.ellipse_info.setText(
                tr('Center C: ({0:.3f}; {1:.3f}) mm\nV1 (+a): ({2:.3f}; {3:.3f}) mm\nV2 (−a): ({4:.3f}; {5:.3f}) mm\nV3 (+b): ({6:.3f}; {7:.3f}) mm\nV4 (−b): ({8:.3f}; {9:.3f}) mm\na={10:.3f} mm, b={11:.3f} mm, f_eff={12:.3f} mm', center[0], center[1], major_plus[0], major_plus[1], major_minus[0], major_minus[1], minor_plus[0], minor_plus[1], minor_minus[0], minor_minus[1], a, b, focal)
            )
        except ValueError as exc:
            self.ellipse_info.setText(str(exc))

    def _refresh_list(self, select_first: bool = False) -> None:
        if select_first and self.system.elements:
            first = next(
                (item for item in self.system.elements if item.kind != "cryostat"),
                self.system.elements[0],
            )
            self.selected_uid = first.uid
            self.selected_uids = {self.selected_uid}
        self._updating_list = True
        self.object_list.clear()
        selected_row = -1
        for index, item in enumerate(self.system.elements):
            self.object_list.addItem(f'{KIND_LABELS[item.kind]}: {item.name}{('' if item.enabled else tr(' (off)'))}')
            if item.uid == self.selected_uid:
                selected_row = index
        self.object_list.setCurrentRow(selected_row)
        self._updating_list = False
        self.scene.set_selected(self.selected_uid, self.selected_uids)
        self._build_property_form()

    def _row_selected(self, row: int) -> None:
        if self._updating_list or row < 0 or row >= len(self.system.elements):
            return
        self.select_uid(self.system.elements[row].uid)

    def select_uid(self, uid: str, additive: bool = False) -> None:
        if additive:
            if uid in self.selected_uids:
                self.selected_uids.remove(uid)
            else:
                self.selected_uids.add(uid)
        else:
            self.selected_uids = {uid}
        self.selected_uid = uid if uid in self.selected_uids else (next(iter(self.selected_uids), uid))
        self._refresh_list()
        self.scene.draw()

    def select_uids(self, uids: set[str], additive: bool = False) -> None:
        """Receive a rubber-band selection made directly on the 2D graph."""
        self.selected_uids = (self.selected_uids | uids) if additive else set(uids)
        self.selected_uid = next(iter(self.selected_uids), None)
        self._refresh_list()
        self.scene.draw()

    def rotate_selected(self, angle_deg: float) -> None:
        if self.scene.rotatable_selection_count() == 0:
            QMessageBox.information(self, tr('Rotation'), tr('Select objects with a rectangle or Shift on the 2D plot.'))
            return
        self.begin_history_action()
        self.scene.rotate_selected_by(angle_deg)
        self.commit_history_action()

    def translate_selected_from_fields(self) -> None:
        if not self.selected_uids:
            QMessageBox.information(self, tr('Translation'), tr('Select one or more objects on the 2D plot'))
            return
        try:
            dx = float(self.translate_dx_editor.text().strip().replace(",", "."))
            dy = float(self.translate_dy_editor.text().strip().replace(",", "."))
        except ValueError:
            QMessageBox.critical(self, tr('Translation'), tr('dx and dy must be numbers'))
            return
        self.begin_history_action()
        self.scene.translate_selected_by(dx, dy)
        self.commit_history_action()

    def model_changed(self) -> None:
        item = self.selected()
        if item is not None:
            for key, editor in self.field_editors.items():
                value = getattr(item, key)
                editor.setText("" if value is None else f"{value:g}" if isinstance(value, float) else str(value))
            if item.kind == "elliptical_mirror":
                self._update_ellipse_info(item)
            if item.kind == "target" and hasattr(self, "target_mode_combo"):
                self.target_mode_combo.blockSignals(True)
                self.target_mode_combo.setCurrentIndex(
                    max(self.target_mode_combo.findData(item.position_mode), 0)
                )
                self.target_mode_combo.blockSignals(False)
            if item.kind == "horn" and hasattr(self, "waveguide_info"):
                self.waveguide_info.setText(
                    tr('Waveguide: X={0:.3f} mm; Y={1:.3f} mm', item.waveguide_x, item.waveguide_y)
                )
        self.scene.draw()
        self._update_info()

    def apply_properties(self) -> None:
        item = self.selected()
        if item is None:
            return
        self.begin_history_action()
        try:
            for key, editor in self.field_editors.items():
                raw = editor.text().strip().replace(",", ".")
                if key == "name":
                    setattr(item, key, raw or KIND_LABELS[item.kind])
                elif key == "window_count":
                    numeric = float(raw)
                    if not numeric.is_integer():
                        raise ValueError(tr('The number of windows must be an integer'))
                    setattr(item, key, int(numeric))
                elif key in ("target_x", "target_y", "focus1_x", "focus1_y", "focus2_x", "focus2_y") and not raw:
                    setattr(item, key, None)
                else:
                    setattr(item, key, float(raw))
            item.enabled = self.enabled_checkbox.isChecked()
            if item.kind == "target":
                item.position_mode = self.target_mode_combo.currentData()
                if item.relative_distance < 0:
                    raise ValueError(tr('Target distance cannot be negative'))
            if item.kind == "horn" and item.horn_length < item.pcl_depth:
                raise ValueError(tr('Horn length cannot be less than the PCL distance from the aperture'))
            if item.kind == "block" and item.block_length < 0:
                raise ValueError(tr('Block length cannot be negative'))
            if item.kind == "horn" and hasattr(self, "frequency_table"):
                channels: list[FrequencyChannel] = []
                for row in range(self.frequency_table.rowCount()):
                    enabled_item = self.frequency_table.item(row, 0)
                    frequency_item = self.frequency_table.item(row, 1)
                    if frequency_item is None or not frequency_item.text().strip():
                        continue
                    frequency = float(frequency_item.text().replace(",", "."))
                    if frequency <= 0:
                        raise ValueError(tr('Frequencies must be positive'))
                    channel_uid = enabled_item.data(Qt.ItemDataRole.UserRole)
                    channels.append(FrequencyChannel(
                        frequency,
                        enabled_item.checkState() == Qt.CheckState.Checked,
                        channel_uid or FrequencyChannel().uid,
                    ))
                if not channels:
                    raise ValueError(tr('A horn must have at least one frequency'))
                item.frequency_channels = channels
                item.frequency_ghz = channels[0].frequency_ghz
            if item.aperture <= 0 or item.radius <= 0 or item.inner_radius <= 0:
                raise ValueError(tr('Dimensions must be positive'))
            if item.kind == "cryostat" and not 1 <= item.window_count <= 4:
                raise ValueError(tr('The cryostat must have between 1 and 4 windows'))
            enforce_links(self.system)
        except ValueError as exc:
            self.rollback_history_action()
            QMessageBox.critical(self, tr('Invalid parameters'), str(exc))
            return
        self._refresh_list()
        self.diagram.rebuild()
        self.scene.draw()
        self._update_info()
        self.commit_history_action()

    def _sync_system_controls(self) -> None:
        self.max_path_editor.setText(f"{self.system.max_path_mm:g}")
        self.beam_scale_editor.setText(f"{self.system.beam_scale:g}")
        for widget, value in (
            (self.render_bounds, self.system.render_beam_bounds),
            (self.reflect_bounds, self.system.simulate_boundary_reflections),
            (self.limit_horn_motion, self.system.link_horn_to_ellipse),
            (self.limit_plane_motion, self.system.link_plane_to_ellipse),
        ):
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)

    def _system_toggles_changed(self) -> None:
        self.begin_history_action()
        self.system.render_beam_bounds = self.render_bounds.isChecked()
        self.system.simulate_boundary_reflections = self.reflect_bounds.isChecked()
        self.system.link_horn_to_ellipse = self.limit_horn_motion.isChecked()
        self.system.link_plane_to_ellipse = self.limit_plane_motion.isChecked()
        enforce_links(self.system)
        self.model_changed()
        self.commit_history_action()

    def apply_system(self) -> None:
        self.begin_history_action()
        try:
            self.system.max_path_mm = float(self.max_path_editor.text().replace(",", "."))
            self.system.beam_scale = float(self.beam_scale_editor.text().replace(",", "."))
            if self.system.max_path_mm <= 0 or self.system.beam_scale <= 0:
                raise ValueError(tr('Values must be positive'))
        except ValueError as exc:
            self.rollback_history_action()
            QMessageBox.critical(self, tr('Invalid parameters'), str(exc))
            return
        self.scene.draw()
        self._update_info()
        self.commit_history_action()

    def _update_info(self) -> None:
        traces = trace_all_beams(self.system)
        if hasattr(self, "diagram"):
            self.diagram.refresh_waists(traces)
        if not self.system.horns:
            self.info_label.setText(tr('Add an enabled horn'))
            return
        text = tr('Active Gaussian beams: {0}; boundary ±{1:g}w(z)', len(traces), self.system.beam_scale)
        warnings = [warning for trace in traces for warning in trace.result.warnings]
        if warnings:
            text += "\n⚠ " + " ".join(warnings)
        self.info_label.setText(text)

    def links_changed(self, new_link: object | None = None, selected_before: set[str] | None = None) -> None:
        if new_link is not None and selected_before:
            source = self.system.element(new_link.source_uid)
            target = self.system.element(new_link.target_uid)
            if source is not None and target is not None and source.kind == "block" and target.kind == "horn":
                attach_assembly_to_block(self.system, source, target, selected_before)
        enforce_links(self.system)
        self.scene.draw()
        self._update_info()

    def aim_horn_at_linked_mirror(self) -> None:
        horn = self.selected()
        if horn is None or horn.kind != "horn":
            return
        mirror = next((item for item in self.system.outgoing(horn.uid) if item.kind == "elliptical_mirror"), None)
        if mirror is None:
            QMessageBox.information(self, tr('Aiming'), tr('Connect the horn output to the elliptical mirror input'))
            return
        self.begin_history_action()
        horn.angle_deg = math.degrees(math.atan2(mirror.y - horn.y, mirror.x - horn.x))
        enforce_links(self.system)
        self.model_changed()
        self.commit_history_action()

    def aim_plane_at_next_target(self) -> None:
        mirror = self.selected()
        if mirror is None or mirror.kind not in ("plane_mirror", "curved_mirror"):
            return
        next_objects = [item for item in self.system.outgoing(mirror.uid) if item.kind not in ("ruler", "cryostat")]
        if next_objects:
            target_x, target_y = next_objects[0].x, next_objects[0].y
        elif mirror.target_x is not None and mirror.target_y is not None:
            target_x, target_y = mirror.target_x, mirror.target_y
        else:
            QMessageBox.information(self, tr('Aiming'), tr('Set the next connection or target coordinates'))
            return
        incoming = None
        for beam in trace_all_beams(self.system):
            segment = next((part for part in beam.result.segments if part.hit_uid == mirror.uid), None)
            if segment is not None:
                incoming = segment.direction
                break
        if incoming is None:
            previous = self.system.incoming(mirror.uid)
            if previous:
                incoming = unit(np.array([mirror.x - previous[0].x, mirror.y - previous[0].y]))
        if incoming is None:
            QMessageBox.information(self, tr('Aiming'), tr('Cannot determine the incident beam direction'))
            return
        self.begin_history_action()
        mirror.target_x, mirror.target_y = target_x, target_y
        mirror.angle_deg = orient_mirror_to_target(mirror, incoming)
        self.model_changed()
        self.commit_history_action()

    def aim_object_at_next(self, uid: str) -> None:
        self.selected_uid = uid
        self.selected_uids = {uid}
        self._refresh_list()
        item = self.selected()
        if item is None:
            return
        if item.kind == "horn":
            self.aim_horn_at_linked_mirror()
            return
        if item.kind in ("plane_mirror", "curved_mirror"):
            self.aim_plane_at_next_target()
            return
        targets = [target for target in self.system.outgoing(item.uid) if target.kind not in ("ruler", "cryostat")]
        if not targets:
            QMessageBox.information(self, tr('Aiming'), tr('The object has no outgoing connection'))
            return
        target = targets[0]
        self.begin_history_action()
        if item.kind == "elliptical_mirror":
            item.focus2_x, item.focus2_y = target.x, target.y
            item.target_x, item.target_y = target.x, target.y
        elif item.kind == "lens":
            axis_angle = math.degrees(math.atan2(target.y - item.y, target.x - item.x))
            item.angle_deg = axis_angle + 90.0
        else:
            self.rollback_history_action()
            QMessageBox.information(self, tr('Aiming'), tr('Aiming is not available for this object type'))
            return
        enforce_links(self.system)
        self.model_changed()
        self.commit_history_action()

    def add_element(self, kind: str) -> None:
        self.begin_history_action()
        xlim, ylim = self.scene.view_range()
        x, y = sum(xlim) / 2, sum(ylim) / 2
        count = sum(item.kind == kind for item in self.system.elements) + 1
        item = OpticalElement(kind, f"{KIND_LABELS[kind]} {count}", x, y)
        if kind == "horn":
            item.aperture = 6.0
        elif kind == "elliptical_mirror":
            item.focus1_x, item.focus1_y = x - 60, y
            item.focus2_x, item.focus2_y = x + 60, y
            item.y += 50
            item.aperture = 40.0
            item.focal_length = 0.0
        elif kind == "ruler":
            item.x, item.y = x - 50.0, y
            item.target_x, item.target_y = x + 50.0, y
        elif kind == "target":
            item.position_mode = "coordinates"
            item.relative_distance = 100.0
            item.relative_angle_deg = 0.0
        elif kind == "block":
            item.block_length = 40.0
            item.aperture = 12.0
        self.system.elements.append(item)
        self.selected_uid = item.uid
        self.selected_uids = {item.uid}
        enforce_links(self.system)
        self._refresh_list()
        self.diagram.rebuild()
        self.scene.draw(reset_view=True)
        self.commit_history_action()

    def insert_optical_assembly(self) -> None:
        self.begin_history_action()
        template = default_band6_system()
        assembly = [item for item in template.elements if item.kind not in ("cryostat", "target")]
        xlim, ylim = self.scene.view_range()
        destination = np.array([sum(xlim) / 2.0, sum(ylim) / 2.0])
        source_center = np.mean([[item.x, item.y] for item in assembly], axis=0)
        shift = destination - source_center
        suffix = 1 + sum(item.kind == "horn" for item in self.system.elements)
        for item in assembly:
            item.x += float(shift[0])
            item.y += float(shift[1])
            item.name = f"{item.name} [{suffix}]"
            for prefix in ("focus1", "focus2", "target"):
                x_value = getattr(item, f"{prefix}_x")
                y_value = getattr(item, f"{prefix}_y")
                if x_value is not None and y_value is not None:
                    setattr(item, f"{prefix}_x", x_value + float(shift[0]))
                    setattr(item, f"{prefix}_y", y_value + float(shift[1]))
        assembly_uids = {item.uid for item in assembly}
        self.system.elements.extend(assembly)
        self.system.links.extend(
            link for link in template.links
            if link.source_uid in assembly_uids and link.target_uid in assembly_uids
        )
        self.selected_uids = assembly_uids
        self.selected_uid = assembly[0].uid
        enforce_links(self.system)
        self._refresh_list()
        self.diagram.rebuild()
        self.scene.draw(reset_view=True)
        self._update_info()
        self.commit_history_action()

    def delete_selected(self) -> None:
        removing = self.selected_uids or ({self.selected_uid} if self.selected_uid else set())
        if not removing:
            return
        self.begin_history_action()
        self.system.elements = [item for item in self.system.elements if item.uid not in removing]
        valid = {item.uid for item in self.system.elements}
        self.system.links = [
            link for link in self.system.links
            if link.source_uid in valid and link.target_uid in valid
        ]
        first = next((item for item in self.system.elements if item.kind != "cryostat"), None)
        if first is None and self.system.elements:
            first = self.system.elements[0]
        self.selected_uid = first.uid if first is not None else None
        self.selected_uids = {self.selected_uid} if self.selected_uid else set()
        self._refresh_list()
        self.diagram.rebuild()
        self.scene.draw()
        self.commit_history_action()

    def load_default(self) -> None:
        self.begin_history_action()
        self.system = default_band6_system()
        enforce_links(self.system)
        self.selected_uid = None
        self.selected_uids.clear()
        self._sync_system_controls()
        self._refresh_list(select_first=True)
        self.diagram.rebuild()
        self.scene.draw(reset_view=True)
        self._update_info()
        self.commit_history_action()

    def open_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr('Open system'), "", "Optics2D JSON (*.json)")
        if not path:
            return
        self.begin_history_action()
        try:
            self.system = OpticalSystem.load(path)
            enforce_links(self.system)
        except (OSError, ValueError, KeyError) as exc:
            self.rollback_history_action()
            QMessageBox.critical(self, tr('Open failed'), str(exc))
            return
        self.selected_uid = None
        self.selected_uids.clear()
        self._sync_system_controls()
        self._refresh_list(select_first=True)
        self.diagram.rebuild()
        self.scene.draw(reset_view=True)
        self._update_info()
        self.commit_history_action()

    def save_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, tr('Save system'), "optical-system.json", "Optics2D JSON (*.json)")
        if path:
            if not path.lower().endswith(".json"):
                path += ".json"
            self.system.save(path)


def run() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    window = OpticsDesktop()
    if "--smoke-test" in sys.argv:
        from .smoke import run_smoke_test
        run_smoke_test(app, window, sys.argv[sys.argv.index("--smoke-test") + 1])
        return
    window.show()
    raise SystemExit(app.exec())
