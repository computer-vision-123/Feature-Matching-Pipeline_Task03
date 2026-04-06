"""
tab.py  –  Harris / λ- Feature Extraction Tab
Communicates with the C++ backend via cv_backend.run_extraction().
"""

import os
import sys
import time

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QDoubleSpinBox, QSpinBox, QGroupBox,
    QGridLayout, QScrollArea, QSizePolicy, QSplitter,
    QFrame, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

# ── Try importing the C++ backend ────────────────────────────────────────────
try:
    import cv_backend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Worker thread – keeps UI responsive during heavy computation
# ─────────────────────────────────────────────────────────────────────────────
class ExtractionWorker(QThread):
    finished = pyqtSignal(object)   # emits PyExtractionResult or Exception
    error    = pyqtSignal(str)

    def __init__(self, img_bytes, params):
        super().__init__()
        self.img_bytes = img_bytes
        self.params    = params

    def run(self):
        try:
            result = cv_backend.run_extraction(
                self.img_bytes,
                k          = self.params["k"],
                block_size = self.params["block_size"],
                sigma      = self.params["sigma"],
                threshold  = self.params["threshold"],
                nms_radius = self.params["nms_radius"],
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _bytes_to_pixmap(png_bytes: bytes) -> QPixmap:
    """Convert raw PNG bytes → QPixmap."""
    img = QImage.fromData(png_bytes, "PNG")
    return QPixmap.fromImage(img)


def _make_label(text: str, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    if bold:
        f = lbl.font()
        f.setBold(True)
        lbl.setFont(f)
    return lbl


def _image_display_label() -> QLabel:
    """A QLabel configured for centred, scaled image display."""
    lbl = QLabel("No image loaded")
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setMinimumSize(320, 240)
    lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    lbl.setStyleSheet(
        "QLabel { background: #1a1a2e; color: #888; border: 1px solid #333;"
        " border-radius: 6px; font-size: 12px; }"
    )
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
#  Main Tab
# ─────────────────────────────────────────────────────────────────────────────
class MainTab(QWidget):
    """Harris + λ- feature extraction tab."""

    STYLE = """
    QWidget {
        background-color: #0d0d1a;
        color: #e0e0f0;
        font-family: 'Segoe UI', sans-serif;
        font-size: 13px;
    }
    QGroupBox {
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        margin-top: 10px;
        padding: 8px;
        background: #12122a;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        color: #7b7bff;
        font-weight: bold;
    }
    QPushButton {
        background: #2a2a6e;
        color: #c8c8ff;
        border: 1px solid #4a4aaa;
        border-radius: 6px;
        padding: 6px 14px;
        min-height: 28px;
    }
    QPushButton:hover  { background: #3a3a9e; }
    QPushButton:pressed{ background: #1a1a5e; }
    QPushButton:disabled { background: #1a1a3a; color: #555; }
    QDoubleSpinBox, QSpinBox {
        background: #1a1a3a;
        color: #c8c8ff;
        border: 1px solid #3a3a6a;
        border-radius: 4px;
        padding: 2px 6px;
    }
    QLabel#stat { color: #a0ffa0; font-weight: bold; }
    QProgressBar {
        border: 1px solid #3a3a6a;
        border-radius: 4px;
        background: #1a1a3a;
        text-align: center;
        color: #c8c8ff;
    }
    QProgressBar::chunk { background: #5555ff; border-radius: 3px; }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(self.STYLE)

        self._img_bytes: bytes | None = None
        self._worker:    ExtractionWorker | None = None

        self._build_ui()
        self._check_backend()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        # ── Top bar: load + parameters ──────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(10)

        self._btn_load = QPushButton("📂  Load Image")
        self._btn_load.clicked.connect(self._on_load)
        top.addWidget(self._btn_load)

        top.addWidget(self._param_box())

        self._btn_run = QPushButton("▶  Run Extraction")
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run)
        top.addWidget(self._btn_run)

        root.addLayout(top)

        # ── Progress bar (hidden until running) ─────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setVisible(False)
        self._progress.setFixedHeight(6)
        root.addWidget(self._progress)

        # ── Image panels (splitter) ──────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)

        self._lbl_orig   = _image_display_label()
        self._lbl_harris = _image_display_label()
        self._lbl_lambda = _image_display_label()

        for lbl, title in [
            (self._lbl_orig,   "Original"),
            (self._lbl_harris, "Harris  (red)"),
            (self._lbl_lambda, "λ-  (green)"),
        ]:
            box = QGroupBox(title)
            bl  = QVBoxLayout(box)
            bl.addWidget(lbl)
            splitter.addWidget(box)

        splitter.setSizes([400, 400, 400])
        root.addWidget(splitter, stretch=1)

        # ── Stats bar ────────────────────────────────────────────────────────
        root.addWidget(self._stats_bar())

        # ── Status label ─────────────────────────────────────────────────────
        self._lbl_status = QLabel("Ready." if BACKEND_AVAILABLE
                                  else "⚠ cv_backend not found – build the C++ module first.")
        self._lbl_status.setAlignment(Qt.AlignLeft)
        root.addWidget(self._lbl_status)

    def _param_box(self) -> QGroupBox:
        box = QGroupBox("Parameters")
        grid = QGridLayout(box)
        grid.setSpacing(4)

        def spin_d(val, lo, hi, step, dec=2):
            s = QDoubleSpinBox()
            s.setRange(lo, hi); s.setValue(val)
            s.setSingleStep(step); s.setDecimals(dec)
            return s

        def spin_i(val, lo, hi):
            s = QSpinBox()
            s.setRange(lo, hi); s.setValue(val)
            return s

        self._sp_k         = spin_d(0.04, 0.001, 0.2,  0.005)
        self._sp_block     = spin_i(5, 3, 31)
        self._sp_sigma     = spin_d(1.0, 0.1,  5.0,  0.1)
        self._sp_threshold = spin_d(0.01, 0.001, 0.5, 0.005)
        self._sp_nms       = spin_i(5, 1, 20)

        rows = [
            ("k",         self._sp_k),
            ("Block",     self._sp_block),
            ("σ",         self._sp_sigma),
            ("Threshold", self._sp_threshold),
            ("NMS r",     self._sp_nms),
        ]
        for i, (name, w) in enumerate(rows):
            grid.addWidget(QLabel(name + ":"), 0, 2 * i)
            grid.addWidget(w,                 0, 2 * i + 1)

        return box

    def _stats_bar(self) -> QGroupBox:
        box = QGroupBox("Results")
        h   = QHBoxLayout(box)
        h.setSpacing(20)

        def stat(prefix):
            h.addWidget(QLabel(prefix + ":"))
            lbl = QLabel("—")
            lbl.setObjectName("stat")
            h.addWidget(lbl)
            return lbl

        self._stat_h_count = stat("Harris pts")
        self._stat_h_time  = stat("Harris time")
        self._stat_l_count = stat("λ- pts")
        self._stat_l_time  = stat("λ- time")
        h.addStretch()
        return box

    # ── Backend check ────────────────────────────────────────────────────────

    def _check_backend(self):
        if not BACKEND_AVAILABLE:
            self._btn_run.setEnabled(False)
            self._btn_load.setEnabled(True)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        if not path:
            return

        with open(path, "rb") as f:
            self._img_bytes = f.read()

        # Show original
        px = QPixmap(path)
        self._show_pixmap(self._lbl_orig, px)
        self._lbl_harris.setText("Not yet computed")
        self._lbl_lambda.setText("Not yet computed")
        self._lbl_status.setText(f"Loaded: {os.path.basename(path)}")

        if BACKEND_AVAILABLE:
            self._btn_run.setEnabled(True)

    def _on_run(self):
        if self._img_bytes is None:
            return

        params = {
            "k":          self._sp_k.value(),
            "block_size": self._sp_block.value(),
            "sigma":      self._sp_sigma.value(),
            "threshold":  self._sp_threshold.value(),
            "nms_radius": self._sp_nms.value(),
        }

        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._lbl_status.setText("Running extraction…")

        self._worker = ExtractionWorker(self._img_bytes, params)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)

        # Update stat labels
        self._stat_h_count.setText(str(result.harris_count))
        self._stat_h_time.setText(f"{result.harris_time_ms:.1f} ms")
        self._stat_l_count.setText(str(result.lambda_count))
        self._stat_l_time.setText(f"{result.lambda_time_ms:.1f} ms")

        # Update images
        self._show_pixmap(self._lbl_harris,
                          _bytes_to_pixmap(bytes(result.harris_vis)))
        self._show_pixmap(self._lbl_lambda,
                          _bytes_to_pixmap(bytes(result.lambda_vis)))

        self._lbl_status.setText(
            f"Done.  Harris: {result.harris_count} pts "
            f"({result.harris_time_ms:.1f} ms)   "
            f"λ-: {result.lambda_count} pts "
            f"({result.lambda_time_ms:.1f} ms)"
        )

    def _on_error(self, msg: str):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._lbl_status.setText(f"Error: {msg}")

    # ── Display helper ───────────────────────────────────────────────────────

    def _show_pixmap(self, label: QLabel, px: QPixmap):
        label.setPixmap(
            px.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event):
        """Re-scale displayed pixmaps when the window is resized."""
        super().resizeEvent(event)
        for lbl in (self._lbl_orig, self._lbl_harris, self._lbl_lambda):
            if lbl.pixmap() and not lbl.pixmap().isNull():
                lbl.setPixmap(
                    lbl.pixmap().scaled(
                        lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )