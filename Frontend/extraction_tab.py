# extraction_tab.py

import os
import random

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QDoubleSpinBox, QSpinBox, QGroupBox, QGridLayout,
    QSizePolicy, QSplitter, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint

try:
    import cv_backend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Workers
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


class MatchingWorker(QThread):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, result_a, result_b):
        super().__init__()
        self.result_a = result_a
        self.result_b = result_b

    def run(self):
        try:
            out = cv_backend.run_matching(self.result_a, self.result_b)
            self.finished.emit(out)
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
        "QLabel { background: #f5f5f5; color: #666; border: 1px solid #ddd;"
        " border-radius: 6px; font-size: 12px; }"
    )
    return lbl

def _random_colors(n: int, seed: int = 42):
    rng = random.Random(seed)
    colors = []
    for i in range(n):
        h = int((i * 360 / max(n, 1)) + rng.randint(0, 15)) % 360
        s = rng.randint(180, 255)
        v = rng.randint(180, 255)
        colors.append(QColor.fromHsv(h, s, v))
    return colors

def _draw_keypoint_overlay(px: QPixmap, kpts: list,
                            imgW: int, imgH: int,
                            colors: list) -> QPixmap:
    """Draw coloured squares on a copy of the image at each keypoint."""
    if px is None:
        return QPixmap()

    canvas = px.copy()
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing)

    scaleX = px.width()  / max(imgW, 1)
    scaleY = px.height() / max(imgH, 1)
    sz = 5  # half-size of each square

    for i, (x, y) in enumerate(kpts):
        px_x = int(x * scaleX)
        px_y = int(y * scaleY)
        pen = QPen(colors[i], 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(px_x - sz, px_y - sz, sz * 2, sz * 2)

    painter.end()
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  Main Tab
# ─────────────────────────────────────────────────────────────────────────────
class MainTab(QWidget):

    STYLE = """
    QWidget { background-color: #f8f9fa; color: #212529;
              font-family: 'Segoe UI', sans-serif; font-size: 13px; }
    QGroupBox { border: 1px solid #dee2e6; border-radius: 8px;
                margin-top: 10px; padding: 8px; background: #ffffff; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px;
                       color: #0066cc; font-weight: bold; }
    QPushButton { background: #e9ecef; color: #495057; border: 1px solid #ced4da;
                  border-radius: 6px; padding: 6px 14px; min-height: 28px; }
    QPushButton:hover    { background: #dee2e6; }
    QPushButton:pressed  { background: #ced4da; }
    QPushButton:disabled { background: #f1f3f5; color: #adb5bd; }
    QPushButton#active   { background: #0066cc; color: white; border: 1px solid #0052a3; }
    QDoubleSpinBox, QSpinBox { background: #ffffff; color: #212529;
                               border: 1px solid #ced4da; border-radius: 4px;
                               padding: 2px 6px; }
    QDoubleSpinBox:focus, QSpinBox:focus { border: 1px solid #0066cc; }
    QLabel#stat { color: #28a745; font-weight: bold; }
    QProgressBar { border: 1px solid #dee2e6; border-radius: 4px;
                   background: #ffffff; text-align: center; color: #212529; }
    QProgressBar::chunk { background: #0066cc; border-radius: 3px; }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(self.STYLE)

        self._img_bytes = [None, None]
        self._img_px    = [None, None]
        self._img_sizes = [(0, 0), (0, 0)]
        self._results   = [None, None]
        self._workers   = [None, None]
        self._pending   = 0
        self._view_mode = ["harris", "harris"]

        # Matching state
        self._match_result = None
        self._match_worker = None
        self._match_view   = False       # True when showing match overlay
        self._match_detector = "harris"  # "harris" or "lambda"
        self._match_method   = "ssd"     # "ssd" or "ncc"

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

        # ── Buttons row: Run + Match ─────────────────────────────────────────
        btn_row = QHBoxLayout()

        self._btn_run = QPushButton("▶  Run Description")
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run)
        btn_row.addWidget(self._btn_run)

        self._btn_match = QPushButton("🔗  Match")
        self._btn_match.setEnabled(False)
        self._btn_match.clicked.connect(self._on_match)
        btn_row.addWidget(self._btn_match)

        root.addLayout(btn_row)

        # ── Match overlay controls (hidden until matching done) ──────────────
        self._match_controls = QWidget()
        self._match_controls.setStyleSheet("""
            QPushButton#active { background: #0066cc; color: white; }
            QPushButton { background: #e9ecef; color: #495057; }
        """)
        mc = QHBoxLayout(self._match_controls)
        mc.setContentsMargins(0, 0, 0, 0)
        mc.setSpacing(6)

        mc.addWidget(QLabel("Detector:"))
        self._btn_m_harris = QPushButton("Harris")
        self._btn_m_lambda = QPushButton("λ-")
        self._btn_m_harris.setObjectName("active")
        self._btn_m_harris.clicked.connect(lambda: self._set_match_detector("harris"))
        self._btn_m_lambda.clicked.connect(lambda: self._set_match_detector("lambda"))
        mc.addWidget(self._btn_m_harris)
        mc.addWidget(self._btn_m_lambda)

        mc.addSpacing(20)

        mc.addWidget(QLabel("Method:"))
        self._btn_m_ssd = QPushButton("SSD")
        self._btn_m_ncc = QPushButton("NCC")
        self._btn_m_ssd.setObjectName("active")
        self._btn_m_ssd.clicked.connect(lambda: self._set_match_method("ssd"))
        self._btn_m_ncc.clicked.connect(lambda: self._set_match_method("ncc"))
        mc.addWidget(self._btn_m_ssd)
        mc.addWidget(self._btn_m_ncc)

        mc.addSpacing(20)

        self._btn_back = QPushButton("✕ Back to Keypoints")
        self._btn_back.clicked.connect(self._exit_match_view)
        mc.addWidget(self._btn_back)

        mc.addStretch()
        self._match_controls.setVisible(False)
        root.addWidget(self._match_controls)

        # ── Progress bar ─────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._progress.setFixedHeight(6)
        root.addWidget(self._progress)

        # ── Stats bars ───────────────────────────────────────────────────────
        root.addWidget(self._desc_stats_bar())
        root.addWidget(self._match_stats_bar())

        # ── Status ───────────────────────────────────────────────────────────
        self._lbl_status = QLabel(
            "Ready." if BACKEND_AVAILABLE
            else "⚠ cv_backend not found – build the C++ module first."
        )
        self._lbl_status.setStyleSheet("color: #6c757d; padding: 5px;")
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
            label = QLabel(name + ":")
            label.setStyleSheet("color: #495057;")
            grid.addWidget(label, 0, 2 * i)
            grid.addWidget(w,     0, 2 * i + 1)

        return box

    def _desc_stats_bar(self) -> QGroupBox:
        box = QGroupBox("Description Results")
        h   = QHBoxLayout(box)
        h.setSpacing(20)

        def stat(label):
            lbl_label = QLabel(label + ":")
            lbl_label.setStyleSheet("color: #495057;")
            h.addWidget(lbl_label)
            lbl = QLabel("—"); lbl.setObjectName("stat")
            h.addWidget(lbl); return lbl

        self._stat = {
            "A_harris": stat("A Harris"), "A_lambda": stat("A λ-"),
            "B_harris": stat("B Harris"), "B_lambda": stat("B λ-"),
        }
        h.addStretch()
        return box

    def _match_stats_bar(self) -> QGroupBox:
        self._match_stats_box = QGroupBox("Matching Results")
        g = QGridLayout(self._match_stats_box)
        g.setSpacing(6)

        def stat_label():
            lbl = QLabel("—")
            lbl.setObjectName("stat")
            return lbl

        headers = ["", "SSD Matches", "SSD Time", "NCC Matches", "NCC Time"]
        for c, h in enumerate(headers):
            lbl = QLabel(h)
            lbl.setStyleSheet("color: #0066cc; font-weight: bold; font-size: 11px;")
            g.addWidget(lbl, 0, c)

        self._mstat_ssd_count_h = stat_label()
        self._mstat_ssd_time_h  = stat_label()
        self._mstat_ncc_count_h = stat_label()
        self._mstat_ncc_time_h  = stat_label()
        
        harris_label = QLabel("Harris")
        harris_label.setStyleSheet("color: #495057; font-weight: bold;")
        g.addWidget(harris_label, 1, 0)
        g.addWidget(self._mstat_ssd_count_h, 1, 1)
        g.addWidget(self._mstat_ssd_time_h,  1, 2)
        g.addWidget(self._mstat_ncc_count_h, 1, 3)
        g.addWidget(self._mstat_ncc_time_h,  1, 4)

        self._mstat_ssd_count_l = stat_label()
        self._mstat_ssd_time_l  = stat_label()
        self._mstat_ncc_count_l = stat_label()
        self._mstat_ncc_time_l  = stat_label()
        
        lambda_label = QLabel("λ-")
        lambda_label.setStyleSheet("color: #495057; font-weight: bold;")
        g.addWidget(lambda_label, 2, 0)
        g.addWidget(self._mstat_ssd_count_l, 2, 1)
        g.addWidget(self._mstat_ssd_time_l,  2, 2)
        g.addWidget(self._mstat_ncc_count_l, 2, 3)
        g.addWidget(self._mstat_ncc_time_l,  2, 4)

        self._match_stats_box.setVisible(False)
        return self._match_stats_box

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
        px = QPixmap(path)
        self._img_px[idx] = px
        self._img_sizes[idx] = (px.width(), px.height())
        self._view_mode[idx] = "harris"
        self._show_pixmap(self._lbl_img[idx], px)
        self._lbl_status.setText(f"Image {'AB'[idx]} loaded: {os.path.basename(path)}")

        # Reset stale results
        self._results[idx] = None
        self._match_result = None
        self._match_view = False
        self._match_controls.setVisible(False)
        self._match_stats_box.setVisible(False)
        self._btn_match.setEnabled(False)

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
        self._btn_match.setEnabled(False)
        self._progress.setVisible(True)
        self._lbl_status.setText("Running description…")
        self._pending = 2
        self._match_result = None
        self._match_view = False
        self._match_controls.setVisible(False)
        self._match_stats_box.setVisible(False)

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
            self._btn_match.setEnabled(True)
            self._lbl_status.setText("Done. Click Match to match features.")

    def _on_error(self, msg: str):
        self._pending = 0
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._lbl_status.setText(f"Error: {msg}")

    # ── Matching ─────────────────────────────────────────────────────────────

    def _on_match(self):
        if self._results[0] is None or self._results[1] is None:
            return
        self._btn_match.setEnabled(False)
        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._lbl_status.setText("Running matching…")

        w = MatchingWorker(self._results[0], self._results[1])
        w.finished.connect(self._on_match_done)
        w.error.connect(self._on_match_error)
        self._match_worker = w
        w.start()

    def _on_match_done(self, match_result):
        self._match_result = match_result
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._btn_match.setEnabled(True)

        # Update match stats
        r = match_result
        self._mstat_ssd_count_h.setText(str(r.harris_ssd_match_count))
        self._mstat_ssd_time_h.setText(f"{r.harris_ssd_time_ms:.2f} ms")
        self._mstat_ncc_count_h.setText(str(r.harris_ncc_match_count))
        self._mstat_ncc_time_h.setText(f"{r.harris_ncc_time_ms:.2f} ms")

        self._mstat_ssd_count_l.setText(str(r.lambda_ssd_match_count))
        self._mstat_ssd_time_l.setText(f"{r.lambda_ssd_time_ms:.2f} ms")
        self._mstat_ncc_count_l.setText(str(r.lambda_ncc_match_count))
        self._mstat_ncc_time_l.setText(f"{r.lambda_ncc_time_ms:.2f} ms")

        self._match_stats_box.setVisible(True)

        # Enter match overlay view
        self._match_view = True
        self._match_detector = "harris"
        self._match_method   = "ssd"
        self._match_controls.setVisible(True)
        self._refresh_match_toggles()
        self._refresh_match_overlay()

        self._lbl_status.setText("Matching complete. Toggle detector/method to compare.")

    def _on_match_error(self, msg: str):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._btn_match.setEnabled(True)
        self._lbl_status.setText(f"Matching error: {msg}")

    def _set_match_detector(self, det: str):
        self._match_detector = det
        self._refresh_match_toggles()
        self._refresh_match_overlay()

    def _set_match_method(self, method: str):
        self._match_method = method
        self._refresh_match_toggles()
        self._refresh_match_overlay()

    def _exit_match_view(self):
        self._match_view = False
        self._match_controls.setVisible(False)
        for i in range(2):
            self._refresh_view(i)
        self._lbl_status.setText("Keypoint view restored. Click Match to see matches again.")

    def _refresh_match_toggles(self):
        is_harris = self._match_detector == "harris"
        self._btn_m_harris.setObjectName("active" if is_harris else "")
        self._btn_m_lambda.setObjectName("active" if not is_harris else "")
        for btn in (self._btn_m_harris, self._btn_m_lambda):
            btn.style().unpolish(btn); btn.style().polish(btn)

        is_ssd = self._match_method == "ssd"
        self._btn_m_ssd.setObjectName("active" if is_ssd else "")
        self._btn_m_ncc.setObjectName("active" if not is_ssd else "")
        for btn in (self._btn_m_ssd, self._btn_m_ncc):
            btn.style().unpolish(btn); btn.style().polish(btn)

    def _refresh_match_overlay(self):
        r = self._match_result
        if r is None:
            return

        if self._match_detector == "harris":
            if self._match_method == "ssd":
                kptsA, kptsB = r.harris_kpts_a, r.harris_kpts_b
            else:
                kptsA, kptsB = r.harris_ncc_kpts_a, r.harris_ncc_kpts_b
        else:
            if self._match_method == "ssd":
                kptsA, kptsB = r.lambda_kpts_a, r.lambda_kpts_b
            else:
                kptsA, kptsB = r.lambda_ncc_kpts_a, r.lambda_ncc_kpts_b

        n = min(len(kptsA), len(kptsB))
        colors = _random_colors(n)

        wA, hA = self._img_sizes[0]
        wB, hB = self._img_sizes[1]

        overlayA = _draw_keypoint_overlay(self._img_px[0], kptsA, wA, hA, colors)
        overlayB = _draw_keypoint_overlay(self._img_px[1], kptsB, wB, hB, colors)

        self._show_pixmap(self._lbl_img[0], overlayA)
        self._show_pixmap(self._lbl_img[1], overlayB)

    # ── Description view ─────────────────────────────────────────────────────

    def _set_view(self, idx: int, mode: str):
        if self._results[idx] is None:
            return
        # Exit match overlay when user clicks a detector toggle on individual images
        if self._match_view:
            self._exit_match_view()
        self._view_mode[idx] = mode
        self._refresh_view(idx)
        self._update_toggle_style(idx)

    def _refresh_view(self, idx: int):
        if self._match_view:
            return  # Don't overwrite match overlay
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
        if self._match_view:
            self._refresh_match_overlay()
        else:
            for i in range(2):
                self._refresh_view(i)