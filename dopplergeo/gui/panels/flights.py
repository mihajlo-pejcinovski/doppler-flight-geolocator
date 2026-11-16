"""Flights panel: gated candidate aircraft per tone family. Tick a candidate
out to exclude it from the solve; select one to highlight it on the map and
draw its predicted Doppler curve on the spectrogram (spec "GUI / Panels")."""

from __future__ import annotations

from datetime import datetime, timezone

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...flights.airlines import airline_name
from ..geojson import aircraft_color, family_color

COLUMNS = ["Use", "icao24", "Callsign", "Airline", "Model", "Family", "CPA time (UTC)", "CPA range", "Speed match", "chi2", "Assigned"]


class FlightsPanel(QWidget):
    candidate_selected = Signal(object)        # (family_id, icao24) or None
    inclusion_changed = Signal(str, bool)      # icao24, included

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.summary = QLabel("No flight data yet.")
        self.summary.setWordWrap(True)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSortingEnabled(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.summary)
        lay.addWidget(self.table, 1)
        self._filling = False
        # icao24 -> {"registration", "type", "model"} (from the adsb.lol API, when it served the flights)
        self.aircraft_info: dict[str, dict] = {}
        self.table.itemChanged.connect(self._item_changed)
        self.table.itemSelectionChanged.connect(self._selection)

    def set_summary(self, text: str) -> None:
        self.summary.setText(text)

    def set_candidates(self, solution, excluded: set[str]) -> None:
        self._filling = True
        try:
            cands = [] if solution is None else sorted(solution.candidates, key=lambda c: (c.family_id, c.chi2))
            self.table.setRowCount(len(cands))
            for r, c in enumerate(cands):
                assigned = solution.assignment.get(c.family_id) == c.icao24
                use = QTableWidgetItem()
                use.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                use.setCheckState(Qt.CheckState.Unchecked if c.icao24 in excluded else Qt.CheckState.Checked)
                use.setData(Qt.ItemDataRole.UserRole, (c.family_id, c.icao24))
                self.table.setItem(r, 0, use)
                info = self.aircraft_info.get(c.icao24.lower(), {})
                model = info.get("model") or info.get("type") or ""
                if model and info.get("registration"):
                    model = f"{model} ({info['registration']})"
                cells = [c.icao24, c.callsign or "", airline_name(c.callsign) or "", model, str(c.family_id),
                         datetime.fromtimestamp(c.t_cpa_utc, tz=timezone.utc).strftime("%H:%M:%S"),
                         f"{c.cpa_range_m / 1000:.2f} km", f"{c.speed_match:.2f}",
                         "" if c.chi2 != c.chi2 else f"{c.chi2:.1f}", "yes" if assigned else ""]
                for col, text in enumerate(cells, start=1):
                    it = QTableWidgetItem(text)
                    if col == 2:  # callsign in the aircraft's map colour
                        it.setForeground(QBrush(QColor(aircraft_color(c.icao24))))
                    if col == 5:
                        it.setForeground(QBrush(QColor(family_color(c.family_id))))
                    if assigned:
                        f = QFont()
                        f.setBold(True)
                        it.setFont(f)
                    self.table.setItem(r, col, it)
        finally:
            self._filling = False

    def _item_changed(self, it: QTableWidgetItem) -> None:
        if self._filling or it.column() != 0:
            return
        _, icao = it.data(Qt.ItemDataRole.UserRole)
        self.inclusion_changed.emit(icao, it.checkState() == Qt.CheckState.Checked)

    def _selection(self) -> None:
        rows = {i.row() for i in self.table.selectedItems()}
        if not rows:
            self.candidate_selected.emit(None)
            return
        self.candidate_selected.emit(self.table.item(min(rows), 0).data(Qt.ItemDataRole.UserRole))

    def select_aircraft(self, icao24: str) -> None:
        for r in range(self.table.rowCount()):
            if self.table.item(r, 1).text() == icao24:
                self.table.selectRow(r)
                return
