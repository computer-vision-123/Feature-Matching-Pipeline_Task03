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
    """Runs extraction + description on a single image."""
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
    """Runs the matching backend on two description results."""
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, result_a, result_b):
        super().__init__()
        self.result_a = result_a
        self.result_b = result_b

    def run(self):
        try:
            match_out = cv_backend.run_matching(self.result_a, self.result_b)
            self.finished.emit(match_out)
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _bytes_to_pixmap(png_bytes: bytes) -> QPixmap:
    return QPixmap.fromImage(QImage.fromData(png_bytes, "PNG"))


def _image_label(placeholder: str = "No image loaded") -> QLabel:
    lbl = QLabel(placeholder)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setMinimumSize(320, 240)
    lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    lbl.setStyleSheet(
        "QLabel { background: #1a1a2e; color: #888; border: 1px solid #333;"
        " border-radius: 6px; font-size: 12px; }"
    )
    return lbl


def _random_colors(n: int, seed: int = 42):
    """Generate n visually distinct, vibrant colours."""
    rng = random.Random(seed)
    colors = []
    for i in range(n):
        h = int((i * 360 / max(n, 1)) + rng.randint(0, 15)) % 360
        s = rng.randint(180, 255)
        v = rng.randint(180, 255)
        colors.append(QColor.fromHsv(h, s, v))
    return colors


def _draw_matches(pxA: QPixmap, pxB: QPixmap,
                   kptsA: list, kptsB: list,
                   imgW_A: int, imgH_A: int,
                   imgW_B: int, imgH_B: int) -> QPixmap:
    """
    Draw the two images side-by-side with coloured match lines.
    kptsA / kptsB are lists of (x, y) in original image coords.
    """
    if pxA is None or pxB is None:
        return QPixmap()

    # Create side-by-side canvas
    h = max(pxA.height(), pxB.height())
    w = pxA.width() + pxB.width()
    canvas = QPixmap(w, h)
    canvas.fill(QColor("#0d0d1a"))

    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing)

    # Draw images
    painter.drawPixmap(0, 0, pxA)
    offsetX = pxA.width()
    painter.drawPixmap(offsetX, 0, pxB)

    # Compute scale factors (original image → drawn pixmap)
    scaleAx = pxA.width()  / max(imgW_A, 1)
    scaleAy = pxA.height() / max(imgH_A, 1)
    scaleBx = pxB.width()  / max(imgW_B, 1)
    scaleBy = pxB.height() / max(imgH_B, 1)

    n = min(len(kptsA), len(kptsB))
    colors = _random_colors(n)

    for i in range(n):
        ax, ay = kptsA[i]
        bx, by = kptsB[i]

        p1 = QPoint(int(ax * scaleAx), int(ay * scaleAy))
        p2 = QPoint(int(bx * scaleBx) + offsetX, int(by * scaleBy))

        pen = QPen(colors[i], 1.5)
        painter.setPen(pen)
        painter.drawLine(p1, p2)

        # Small circles at keypoints
        painter.setBrush(colors[i])
        painter.drawEllipse(p1, 4, 4)
        painter.drawEllipse(p2, 4, 4)

    painter.end()
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  Matching Tab
# ─────────────────────────────────────────────────────────────────────────────
class MatchingTab(QWidget):

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
    QLabel#stat  { color: #a0ffa0; font-weight: bold; }
    QLabel#title { color: #9999ff; font-size: 15px; font-weight: bold; }
    QProgressBar { border: 1px solid #3a3a6a; border-radius: 4px;
                   background: #1a1a3a; text-align: center; color: #c8c8ff; }
    QProgressBar::chunk { background: #5555ff; border-radius: 3px; }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(self.STYLE)

        self._img_bytes  = [None, None]    # raw file bytes per image
        self._img_px     = [None, None]    # original QPixmap per image
        self._img_sizes  = [(0, 0), (0, 0)]  # (w, h) of original images
        self._desc_results = [None, None]  # DescriptionResult per image
        self._match_result = None          # MatchingOutput
        self._workers    = [None, None]
        self._match_worker = None
        self._pending    = 0

        # Current view state
        self._detector = "harris"   # "harris" or "lambda"
        self._method   = "ssd"      # "ssd" or "ncc"

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        # Parameters
        root.addWidget(self._param_box())

        # Image load row
        load_row = QHBoxLayout()
        self._btn_load = [QPushButton("📂  Load Image A"),
                          QPushButton("📂  Load Image B")]
        self._lbl_file = [QLabel("—"), QLabel("—")]
        for i in range(2):
            self._btn_load[i].clicked.connect(lambda _, idx=i: self._on_load(idx))
            load_row.addWidget(self._btn_load[i])
            self._lbl_file[i].setStyleSheet("color: #888; font-size: 11px;")
            load_row.addWidget(self._lbl_file[i])
        root.addLayout(load_row)

        # Run button
        self._btn_run = QPushButton("▶  Run Matching Pipeline")
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run)
        root.addWidget(self._btn_run)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._progress.setFixedHeight(6)
        root.addWidget(self._progress)

        # Toggle row: detector + method
        toggle_row = QHBoxLayout()

        toggle_row.addWidget(QLabel("Detector:"))
        self._btn_harris = QPushButton("Harris")
        self._btn_lambda = QPushButton("λ-")
        self._btn_harris.setObjectName("active")
        self._btn_harris.setEnabled(False)
        self._btn_lambda.setEnabled(False)
        self._btn_harris.clicked.connect(lambda: self._set_detector("harris"))
        self._btn_lambda.clicked.connect(lambda: self._set_detector("lambda"))
        toggle_row.addWidget(self._btn_harris)
        toggle_row.addWidget(self._btn_lambda)

        toggle_row.addSpacing(30)

        toggle_row.addWidget(QLabel("Method:"))
        self._btn_ssd = QPushButton("SSD")
        self._btn_ncc = QPushButton("NCC")
        self._btn_ssd.setObjectName("active")
        self._btn_ssd.setEnabled(False)
        self._btn_ncc.setEnabled(False)
        self._btn_ssd.clicked.connect(lambda: self._set_method("ssd"))
        self._btn_ncc.clicked.connect(lambda: self._set_method("ncc"))
        toggle_row.addWidget(self._btn_ssd)
        toggle_row.addWidget(self._btn_ncc)
        toggle_row.addStretch()

        root.addLayout(toggle_row)

        # Match visualisation
        self._lbl_match = _image_label("Run matching to see results")
        self._lbl_match.setMinimumSize(640, 320)
        root.addWidget(self._lbl_match, stretch=1)

        # Stats bar
        root.addWidget(self._stats_bar())

        # Status
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
        box = QGroupBox("Matching Results")
        g   = QGridLayout(box)
        g.setSpacing(6)

        def stat_label():
            lbl = QLabel("—")
            lbl.setObjectName("stat")
            return lbl

        headers = ["", "SSD Matches", "SSD Time", "NCC Matches", "NCC Time"]
        for c, h in enumerate(headers):
            lbl = QLabel(h)
            lbl.setStyleSheet("color: #7b7bff; font-weight: bold; font-size: 11px;")
            g.addWidget(lbl, 0, c)

        self._stat_ssd_count_h = stat_label()
        self._stat_ssd_time_h  = stat_label()
        self._stat_ncc_count_h = stat_label()
        self._stat_ncc_time_h  = stat_label()

        g.addWidget(QLabel("Harris"), 1, 0)
        g.addWidget(self._stat_ssd_count_h, 1, 1)
        g.addWidget(self._stat_ssd_time_h,  1, 2)
        g.addWidget(self._stat_ncc_count_h, 1, 3)
        g.addWidget(self._stat_ncc_time_h,  1, 4)

        self._stat_ssd_count_l = stat_label()
        self._stat_ssd_time_l  = stat_label()
        self._stat_ncc_count_l = stat_label()
        self._stat_ncc_time_l  = stat_label()

        g.addWidget(QLabel("λ-"), 2, 0)
        g.addWidget(self._stat_ssd_count_l, 2, 1)
        g.addWidget(self._stat_ssd_time_l,  2, 2)
        g.addWidget(self._stat_ncc_count_l, 2, 3)
        g.addWidget(self._stat_ncc_time_l,  2, 4)

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
        px = QPixmap(path)
        self._img_px[idx] = px
        self._img_sizes[idx] = (px.width(), px.height())
        self._lbl_file[idx].setText(os.path.basename(path))
        self._lbl_status.setText(f"Image {'AB'[idx]} loaded: {os.path.basename(path)}")

        # Reset stale results
        self._desc_results[idx] = None
        self._match_result = None

        if BACKEND_AVAILABLE and all(self._img_bytes):
            self._btn_run.setEnabled(True)

    def _on_run(self):
        """Run the full pipeline: describe both images, then match."""
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
        self._lbl_status.setText("Running description on both images…")
        self._pending = 2
        self._desc_results = [None, None]
        self._match_result = None

        for i in range(2):
            w = DescriptionWorker(self._img_bytes[i], params, i)
            w.finished.connect(self._on_desc_done)
            w.error.connect(self._on_error)
            self._workers[i] = w
            w.start()

    def _on_desc_done(self, result, idx: int):
        self._desc_results[idx] = result
        self._pending -= 1
        if self._pending == 0:
            self._lbl_status.setText("Description done. Running matching…")
            self._run_matching()

    def _run_matching(self):
        w = MatchingWorker(self._desc_results[0], self._desc_results[1])
        w.finished.connect(self._on_match_done)
        w.error.connect(self._on_error)
        self._match_worker = w
        w.start()

    def _on_match_done(self, match_result):
        self._match_result = match_result
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)

        # Enable toggles
        for btn in (self._btn_harris, self._btn_lambda,
                     self._btn_ssd, self._btn_ncc):
            btn.setEnabled(True)

        # Update stats
        r = match_result
        self._stat_ssd_count_h.setText(str(r.harris_ssd_match_count))
        self._stat_ssd_time_h.setText(f"{r.harris_ssd_time_ms:.2f} ms")
        self._stat_ncc_count_h.setText(str(r.harris_ncc_match_count))
        self._stat_ncc_time_h.setText(f"{r.harris_ncc_time_ms:.2f} ms")

        self._stat_ssd_count_l.setText(str(r.lambda_ssd_match_count))
        self._stat_ssd_time_l.setText(f"{r.lambda_ssd_time_ms:.2f} ms")
        self._stat_ncc_count_l.setText(str(r.lambda_ncc_match_count))
        self._stat_ncc_time_l.setText(f"{r.lambda_ncc_time_ms:.2f} ms")

        # Default view
        self._detector = "harris"
        self._method   = "ssd"
        self._refresh_toggles()
        self._refresh_visualisation()

        self._lbl_status.setText("Matching complete.")

    def _on_error(self, msg: str):
        self._pending = 0
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._lbl_status.setText(f"Error: {msg}")

    def _set_detector(self, det: str):
        self._detector = det
        self._refresh_toggles()
        self._refresh_visualisation()

    def _set_method(self, method: str):
        self._method = method
        self._refresh_toggles()
        self._refresh_visualisation()

    # ── Display ──────────────────────────────────────────────────────────────

    def _refresh_toggles(self):
        is_harris = self._detector == "harris"
        self._btn_harris.setObjectName("active" if is_harris else "")
        self._btn_lambda.setObjectName("active" if not is_harris else "")
        for btn in (self._btn_harris, self._btn_lambda):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        is_ssd = self._method == "ssd"
        self._btn_ssd.setObjectName("active" if is_ssd else "")
        self._btn_ncc.setObjectName("active" if not is_ssd else "")
        for btn in (self._btn_ssd, self._btn_ncc):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _refresh_visualisation(self):
        r = self._match_result
        if r is None:
            return

        # Pick matched keypoints based on current detector + method
        if self._detector == "harris":
            if self._method == "ssd":
                kptsA = r.harris_kpts_a
                kptsB = r.harris_kpts_b
            else:
                kptsA = r.harris_ncc_kpts_a
                kptsB = r.harris_ncc_kpts_b
        else:
            if self._method == "ssd":
                kptsA = r.lambda_kpts_a
                kptsB = r.lambda_kpts_b
            else:
                kptsA = r.lambda_ncc_kpts_a
                kptsB = r.lambda_ncc_kpts_b

        wA, hA = self._img_sizes[0]
        wB, hB = self._img_sizes[1]

        canvas = _draw_matches(
            self._img_px[0], self._img_px[1],
            kptsA, kptsB,
            wA, hA, wB, hB
        )

        if not canvas.isNull():
            self._show_pixmap(self._lbl_match, canvas)

    def _show_pixmap(self, label: QLabel, px: QPixmap):
        label.setPixmap(
            px.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_visualisation()
