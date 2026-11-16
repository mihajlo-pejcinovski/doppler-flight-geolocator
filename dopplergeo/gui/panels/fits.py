"""Fit panel: per-family Doppler fit with 1-sigma errors, grade and quality
flags, plus the selected family's residuals (spec "GUI / Panels")."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..geojson import family_color

COLUMNS = ["Family", "Grade", "k", "f0 (Hz)", "v (m/s)", "d (m)", "t CPA", "chi2 red", "CPA", "Asympt.", "Jumps"]
GRADE_COLORS = {"A": "#c6f6d5", "B": "#fefcbf", "C": "#fed7d7"}


def _utc(epoch: float | None, t: float) -> str:
    if epoch is None:
        return f"{t:.1f} s"
    return datetime.fromtimestamp(epoch + t, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-4]


class FitPanel(QWidget):
    family_selected = Signal(object)   # family id or None
    refit_requested = Signal()
    locks_changed = Signal(int, object)  # family id, {"v": value, "d": value} (empty: unlocked)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.resid = pg.PlotWidget()
        self.resid.setLabel("bottom", "media time", units="s")
        self.resid.setLabel("left", "residual", units="Hz")
        self.resid.setMaximumHeight(150)
        self.resid.addLine(y=0, pen=pg.mkPen("#888", style=Qt.PenStyle.DashLine))
        self._resid_items: list = []
        self.refit_btn = QPushButton("Re-fit")
        self.refit_btn.setToolTip("Fit the families again from the tracks as edited")
        self.summary = QLabel("")
        # Lock v / d of the selected family for the next re-fit.
        self.lock_label = QLabel("Lock:")
        self.lock_v = QCheckBox("v")
        self.lock_v_val = QDoubleSpinBox(decimals=1, minimum=10.0, maximum=400.0, suffix=" m/s", value=80.0)
        self.lock_d = QCheckBox("d")
        self.lock_d_val = QDoubleSpinBox(decimals=0, minimum=10.0, maximum=30000.0, singleStep=50.0, suffix=" m",
                                         value=1000.0)
        for w in (self.lock_v, self.lock_v_val, self.lock_d, self.lock_d_val):
            w.setEnabled(False)
            w.setToolTip("Hold this parameter of the selected family at the value in the next re-fit "
                         "(e.g. a speed known from ADS-B)")
        self._locks: dict[int, dict[str, float]] = {}
        self._lock_fid: int | None = None
        bar = QHBoxLayout()
        bar.addWidget(self.summary, 1)
        for w in (self.lock_label, self.lock_v, self.lock_v_val, self.lock_d, self.lock_d_val):
            bar.addWidget(w)
        bar.addWidget(self.refit_btn)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(bar)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.resid)
        self._results: list = []
        self.table.itemSelectionChanged.connect(self._selection)
        self.refit_btn.clicked.connect(self.refit_requested)
        for w in (self.lock_v, self.lock_d):
            w.toggled.connect(lambda _on: self._lock_edited())
        for w in (self.lock_v_val, self.lock_d_val):
            w.valueChanged.connect(lambda _v: self._lock_edited())

    def set_locks(self, locks: dict[int, dict[str, float]]) -> None:
        self._locks = {int(k): dict(v) for k, v in locks.items()}
        self._show_lock(self._lock_fid)

    def _show_lock(self, fid) -> None:
        """Fill the lock controls for a family: its locked values, else its fit's."""
        self._lock_fid = None  # no echo while filling
        fr = next((x for x in self._results if x.family.id == fid), None)
        on = fr is not None
        for w in (self.lock_v, self.lock_v_val, self.lock_d, self.lock_d_val):
            w.setEnabled(on)
        if on:
            lk = self._locks.get(int(fid), {})
            f = fr.fit
            self.lock_v.setChecked("v" in lk)
            self.lock_d.setChecked("d" in lk)
            self.lock_v_val.setValue(lk.get("v", f.v if f else 80.0))
            self.lock_d_val.setValue(lk.get("d", f.d if f else 1000.0))
            self.lock_label.setText(f"Lock #{fid}:")
            self._lock_fid = int(fid)
        else:
            self.lock_label.setText("Lock:")

    def _lock_edited(self) -> None:
        fid = self._lock_fid
        if fid is None:
            return
        lk = {}
        if self.lock_v.isChecked():
            lk["v"] = float(self.lock_v_val.value())
        if self.lock_d.isChecked():
            lk["d"] = float(self.lock_d_val.value())
        if lk:
            self._locks[fid] = lk
        else:
            self._locks.pop(fid, None)
        self.locks_changed.emit(fid, lk)

    def set_results(self, results, epoch: float | None) -> None:
        """``pipeline.FamilyResult`` list."""
        self._results = list(results)
        self.table.setRowCount(len(self._results))
        good = 0
        for r, fr in enumerate(self._results):
            f = fr.fit
            flags = fr.flags
            if f is None:
                cells = [str(fr.family.id), "-", ",".join(f"{k:g}" for k in fr.family.ks), "", "", "", "", "",
                         "", "", str(flags["jumps"])]
            else:
                good += f.grade in ("A", "B")
                cells = [
                    str(fr.family.id), f.grade, ",".join(f"{k:g}" for k in fr.family.ks),
                    f"{f.f0:.2f}", f"{f.v:.1f} ± {f.stderr('v'):.1f}", f"{f.d:.0f} ± {f.stderr('d'):.0f}",
                    f"{_utc(epoch, f.t_cpa)} ± {f.stderr('t_cpa'):.1f}s", f"{f.chi2_red:.2f}",
                    "yes" if flags["cpa_covered"] else "NO", "yes" if flags["asymptotes_visible"] else "no",
                    str(flags["jumps"])]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if c == 0:
                    it.setForeground(QBrush(QColor(family_color(fr.family.id))))
                    it.setData(Qt.ItemDataRole.UserRole, fr.family.id)
                if c == 1 and text in GRADE_COLORS:
                    it.setBackground(QBrush(QColor(GRADE_COLORS[text])))
                    it.setForeground(QBrush(QColor("#000")))
                if (c == 8 and text == "NO") or (c == 10 and text not in ("0", "")):
                    it.setForeground(QBrush(QColor("#c53030")))
                self.table.setItem(r, c, it)
        self.summary.setText(f"{len(self._results)} families, {good} graded A/B")
        self._plot_residuals(None)

    def _selection(self) -> None:
        rows = {i.row() for i in self.table.selectedItems()}
        fid = None
        if rows:
            fid = self.table.item(min(rows), 0).data(Qt.ItemDataRole.UserRole)
        self._plot_residuals(fid)
        self._show_lock(fid)
        self.family_selected.emit(fid)

    def _plot_residuals(self, fid) -> None:
        for it in self._resid_items:
            self.resid.removeItem(it)
        self._resid_items = []
        fr = next((x for x in self._results if x.family.id == fid), None)
        if fr is None or fr.fit is None:
            return
        t, f, k, s = fr.family.stacked()
        res = f - fr.fit.predict(t, k)
        order = np.argsort(t)
        sc = pg.ErrorBarItem(x=t[order], y=res[order], height=2 * s[order], pen=pg.mkPen("#aaa"))
        pts = pg.ScatterPlotItem(t[order], res[order], size=4, pen=None,
                                 brush=pg.mkBrush(family_color(fid)))
        for it in (sc, pts):
            self.resid.addItem(it)
            self._resid_items.append(it)
