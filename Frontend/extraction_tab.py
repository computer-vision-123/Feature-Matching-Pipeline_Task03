import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QDoubleSpinBox, QSpinBox, QGroupBox, QGridLayout,
    QSizePolicy, QSplitter, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

try:
    import cv_backend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Worker
# ─────────────────────────────────────────────────────────────────────────────
class DescriptionWorker(QThread):
    finished = pyqtSignal(object, int)
    error    = pyqtSignal(str)

    def __init__(self, img_bytes, params, index):
        super().__init__()
        self.img_bytes = img_bytes
        self.params    = params
        self.index     = index

    def run(self):
        try:
            result = cv_backend.run_description(
                self.img_bytes,
                k           = self.params["k"],
                block_size  = self.params["block_size"],
                sigma       = self.params["sigma"],
                threshold   = self.params["threshold"],
                nms_radius  = self.params["nms_radius"],
                num_octaves = self.params["num_octaves"],
            )
            self.finished.emit(result, self.index)
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _bytes_to_pixmap(png_bytes: bytes) -> QPixmap:
    return QPixmap.fromImage(QImage.fromData(png_bytes, "PNG"))

def _image_label() -> QLabel:
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

    STYLE = """
    QWidget { background-color: #0d0d1a; color: #e0e0f0;
              font-family: 'Segoe UI', sans-serif; font-size: 13px; }
    QGroupBox { border: 1px solid #2a2a4a; border-radius: 8px;
                margin-top: 10px; padding: 8px; background: #12122a; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px;
                       color: #7b7bff; font-weight: bold; }
    QPushButton { background: #2a2a6e; color: #c8c8ff; border: 1px solid #4a4aaa;
                  border-radius: 6px; padding: 6px 14px; min-height: 28px; }
    QPushButton:hover    { background: #3a3a9e; }
    QPushButton:pressed  { background: #1a1a5e; }
    QPushButton:disabled { background: #1a1a3a; color: #555; }
    QPushButton#active   { background: #5555cc; border: 1px solid #9999ff; }
    QDoubleSpinBox, QSpinBox { background: #1a1a3a; color: #c8c8ff;
                               border: 1px solid #3a3a6a; border-radius: 4px;
                               padding: 2px 6px; }
    QLabel#stat { color: #a0ffa0; font-weight: bold; }
    QProgressBar { border: 1px solid #3a3a6a; border-radius: 4px;
                   background: #1a1a3a; text-align: center; color: #c8c8ff; }
    QProgressBar::chunk { background: #5555ff; border-radius: 3px; }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(self.STYLE)

        self._img_bytes = [None, None]
        self._img_px    = [None, None]
        self._results   = [None, None]
        self._workers   = [None, None]
        self._pending   = 0
        self._view_mode = ["harris", "harris"]

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        root.addWidget(self._param_box())

        splitter = QSplitter(Qt.Horizontal)
        self._lbl_img    = [_image_label(), _image_label()]
        self._btn_load   = [QPushButton("📂  Load Image A"),
                            QPushButton("📂  Load Image B")]
        self._btn_harris = [QPushButton("Harris"), QPushButton("Harris")]
        self._btn_lambda = [QPushButton("λ-"),     QPushButton("λ-")]

        for i in range(2):
            box = QGroupBox(f"Image {'AB'[i]}")
            bl  = QVBoxLayout(box)
            bl.addWidget(self._lbl_img[i])

            toggle_row = QHBoxLayout()
            self._btn_harris[i].setEnabled(False)
            self._btn_lambda[i].setEnabled(False)
            self._btn_harris[i].setObjectName("active")
            self._btn_harris[i].clicked.connect(lambda _, idx=i: self._set_view(idx, "harris"))
            self._btn_lambda[i].clicked.connect(lambda _, idx=i: self._set_view(idx, "lambda"))
            toggle_row.addWidget(self._btn_harris[i])
            toggle_row.addWidget(self._btn_lambda[i])
            bl.addLayout(toggle_row)

            bl.addWidget(self._btn_load[i])
            self._btn_load[i].clicked.connect(lambda _, idx=i: self._on_load(idx))
            splitter.addWidget(box)

        splitter.setSizes([600, 600])
        root.addWidget(splitter, stretch=1)

        self._btn_run = QPushButton("▶  Run")
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run)
        root.addWidget(self._btn_run)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._progress.setFixedHeight(6)
        root.addWidget(self._progress)

        root.addWidget(self._stats_bar())

        self._lbl_status = QLabel(
            "Ready." if BACKEND_AVAILABLE
            else "⚠ cv_backend not found – build the C++ module first."
        )
        root.addWidget(self._lbl_status)

    def _param_box(self) -> QGroupBox:
        box  = QGroupBox("Parameters")
        grid = QGridLayout(box)
        grid.setSpacing(4)

        def spin_d(val, lo, hi, step, dec=2):
            s = QDoubleSpinBox()
            s.setRange(lo, hi); s.setValue(val)
            s.setSingleStep(step); s.setDecimals(dec)
            return s

        def spin_i(val, lo, hi):
            s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); return s

        self._sp_k         = spin_d(0.04, 0.001, 0.2,  0.005)
        self._sp_block     = spin_i(5, 3, 31)
        self._sp_sigma     = spin_d(1.0, 0.1,  5.0,  0.1)
        self._sp_threshold = spin_d(0.01, 0.001, 0.5, 0.005)
        self._sp_nms       = spin_i(5, 1, 20)
        self._sp_octaves   = spin_i(3, 1, 8)

        for i, (name, w) in enumerate([
            ("k", self._sp_k), ("Block", self._sp_block),
            ("σ", self._sp_sigma), ("Threshold", self._sp_threshold),
            ("NMS r", self._sp_nms), ("Octaves", self._sp_octaves),
        ]):
            grid.addWidget(QLabel(name + ":"), 0, 2 * i)
            grid.addWidget(w,                  0, 2 * i + 1)

        return box

    def _stats_bar(self) -> QGroupBox:
        box = QGroupBox("Results")
        h   = QHBoxLayout(box)
        h.setSpacing(20)

        def stat(label):
            h.addWidget(QLabel(label + ":"))
            lbl = QLabel("—"); lbl.setObjectName("stat")
            h.addWidget(lbl); return lbl

        self._stat = {
            "A_harris": stat("A Harris"), "A_lambda": stat("A λ-"),
            "B_harris": stat("B Harris"), "B_lambda": stat("B λ-"),
        }
        h.addStretch()
        return box

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_load(self, idx: int):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        if not path:
            return
        with open(path, "rb") as f:
            self._img_bytes[idx] = f.read()
        self._img_px[idx] = QPixmap(path)
        self._view_mode[idx] = "harris"
        self._show_pixmap(self._lbl_img[idx], self._img_px[idx])
        self._lbl_status.setText(f"Image {'AB'[idx]} loaded: {os.path.basename(path)}")
        if BACKEND_AVAILABLE and all(self._img_bytes):
            self._btn_run.setEnabled(True)

    def _on_run(self):
        params = {
            "k":           self._sp_k.value(),
            "block_size":  self._sp_block.value(),
            "sigma":       self._sp_sigma.value(),
            "threshold":   self._sp_threshold.value(),
            "nms_radius":  self._sp_nms.value(),
            "num_octaves": self._sp_octaves.value(),
        }
        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._lbl_status.setText("Running…")
        self._pending = 2

        for i in range(2):
            w = DescriptionWorker(self._img_bytes[i], params, i)
            w.finished.connect(self._on_done)
            w.error.connect(self._on_error)
            self._workers[i] = w
            w.start()

    def _on_done(self, result, idx: int):
        self._results[idx] = result
        self._btn_harris[idx].setEnabled(True)
        self._btn_lambda[idx].setEnabled(True)
        self._view_mode[idx] = "harris"
        self._refresh_view(idx)
        self._update_toggle_style(idx)

        key = "AB"[idx]
        self._stat[f"{key}_harris"].setText(
            f"{result.harris_count} pts ({result.harris_time_ms:.1f} ms)")
        self._stat[f"{key}_lambda"].setText(
            f"{result.lambda_count} pts ({result.lambda_time_ms:.1f} ms)")

        self._pending -= 1
        if self._pending == 0:
            self._progress.setVisible(False)
            self._btn_run.setEnabled(True)
            self._lbl_status.setText("Done. Descriptors ready for matching.")

    def _on_error(self, msg: str):
        self._pending -= 1
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._lbl_status.setText(f"Error: {msg}")

    def _set_view(self, idx: int, mode: str):
        if self._results[idx] is None:
            return
        self._view_mode[idx] = mode
        self._refresh_view(idx)
        self._update_toggle_style(idx)

    # ── Display ──────────────────────────────────────────────────────────────

    def _refresh_view(self, idx: int):
        if self._results[idx] is None:
            if self._img_px[idx]:
                self._show_pixmap(self._lbl_img[idx], self._img_px[idx])
            return
        vis = (self._results[idx].harris_vis if self._view_mode[idx] == "harris"
               else self._results[idx].lambda_vis)
        self._show_pixmap(self._lbl_img[idx], _bytes_to_pixmap(bytes(vis)))

    def _show_pixmap(self, label: QLabel, px: QPixmap):
        label.setPixmap(
            px.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _update_toggle_style(self, idx: int):
        is_harris = self._view_mode[idx] == "harris"
        self._btn_harris[idx].setObjectName("active" if is_harris else "")
        self._btn_lambda[idx].setObjectName("active" if not is_harris else "")
        for btn in (self._btn_harris[idx], self._btn_lambda[idx]):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for i in range(2):
            self._refresh_view(i)