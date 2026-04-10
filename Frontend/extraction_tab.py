import os
import random

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QDoubleSpinBox, QSpinBox, QGroupBox, QGridLayout,
    QSizePolicy, QSplitter, QProgressBar, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QSize

try:
    import cv_backend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Workers  (unchanged)
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

# Fixed pixel size for image display panels
IMAGE_PANEL_W = 480
IMAGE_PANEL_H = 340

def _image_label() -> QLabel:
    lbl = QLabel()
    lbl.setAlignment(Qt.AlignCenter)
    # Fixed size – never grows or shrinks with window
    lbl.setFixedSize(IMAGE_PANEL_W, IMAGE_PANEL_H)
    lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    # Placeholder text drawn via a helper property we set later
    lbl.setText("Drop or load an image")
    lbl.setStyleSheet("""
        QLabel {
            background-color: #F3F4F6;
            color: #9CA3AF;
            border: 2px dashed #D1D5DB;
            border-radius: 10px;
            font-size: 13px;
            font-family: 'Segoe UI', sans-serif;
        }
    """)
    return lbl

def _random_colors(n: int, seed: int = 42):
    rng = random.Random(seed)
    colors = []
    for i in range(n):
        h = int((i * 360 / max(n, 1)) + rng.randint(0, 15)) % 360
        s = rng.randint(180, 240)
        v = rng.randint(140, 210)
        colors.append(QColor.fromHsv(h, s, v))
    return colors

def _draw_keypoint_overlay(px: QPixmap, kpts: list,
                            imgW: int, imgH: int,
                            colors: list) -> QPixmap:
    if px is None:
        return QPixmap()
    canvas = px.copy()
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing)
    scaleX = px.width()  / max(imgW, 1)
    scaleY = px.height() / max(imgH, 1)
    sz = 5
    for i, (x, y) in enumerate(kpts):
        px_x = int(x * scaleX)
        px_y = int(y * scaleY)
        pen = QPen(colors[i], 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(px_x - sz, px_y - sz, sz * 2, sz * 2)
    painter.end()
    return canvas


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setStyleSheet("color: #E5E7EB;")
    return line


# ─────────────────────────────────────────────────────────────────────────────
#  Main Tab
# ─────────────────────────────────────────────────────────────────────────────
class MainTab(QWidget):

    STYLE = """
    /* ── Base ── */
    QWidget {
        background-color: #FFFFFF;
        color: #1F2937;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        font-size: 13px;
    }

    /* ── Group boxes ── */
    QGroupBox {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        margin-top: 14px;
        padding: 12px 10px 10px 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        top: 2px;
        padding: 0 6px;
        color: #374151;
        font-weight: 600;
        font-size: 12px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* ── Buttons – default ── */
    QPushButton {
        background-color: #F9FAFB;
        color: #374151;
        border: 1px solid #D1D5DB;
        border-radius: 7px;
        padding: 6px 16px;
        min-height: 30px;
        font-weight: 500;
    }
    QPushButton:hover {
        background-color: #F3F4F6;
        border-color: #9CA3AF;
    }
    QPushButton:pressed {
        background-color: #E5E7EB;
    }
    QPushButton:disabled {
        background-color: #F9FAFB;
        color: #D1D5DB;
        border-color: #E5E7EB;
    }

    /* ── Active / selected toggle ── */
    QPushButton#active {
        background-color: #2563EB;
        color: #FFFFFF;
        border: 1px solid #1D4ED8;
        font-weight: 600;
    }
    QPushButton#active:hover {
        background-color: #1D4ED8;
    }

    /* ── Primary action buttons ── */
    QPushButton#primary {
        background-color: #111827;
        color: #FFFFFF;
        border: none;
        border-radius: 7px;
        padding: 7px 20px;
        min-height: 32px;
        font-weight: 600;
    }
    QPushButton#primary:hover  { background-color: #1F2937; }
    QPushButton#primary:pressed { background-color: #374151; }
    QPushButton#primary:disabled {
        background-color: #E5E7EB;
        color: #9CA3AF;
    }

    /* ── Load image buttons ── */
    QPushButton#load {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        border-radius: 7px;
        padding: 6px 16px;
        min-height: 30px;
        font-weight: 500;
    }
    QPushButton#load:hover {
        background-color: #DBEAFE;
        border-color: #93C5FD;
    }

    /* ── Spin boxes ── */
    QDoubleSpinBox, QSpinBox {
        background-color: #F9FAFB;
        color: #1F2937;
        border: 1px solid #D1D5DB;
        border-radius: 6px;
        padding: 3px 8px;
        min-width: 72px;
    }
    QDoubleSpinBox:focus, QSpinBox:focus {
        border-color: #2563EB;
        background-color: #FFFFFF;
    }
    QDoubleSpinBox::up-button, QSpinBox::up-button,
    QDoubleSpinBox::down-button, QSpinBox::down-button {
        width: 16px;
    }

    /* ── Stat labels ── */
    QLabel#stat {
        color: #059669;
        font-weight: 600;
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 5px;
        padding: 2px 8px;
    }

    /* ── Status bar label ── */
    QLabel#status {
        color: #6B7280;
        font-size: 12px;
        padding: 4px 2px;
    }
    QLabel#statusError {
        color: #DC2626;
        font-size: 12px;
        padding: 4px 2px;
    }
    QLabel#statusOk {
        color: #059669;
        font-size: 12px;
        padding: 4px 2px;
    }

    /* ── Progress bar ── */
    QProgressBar {
        border: none;
        border-radius: 3px;
        background-color: #E5E7EB;
        text-align: center;
        color: transparent;
    }
    QProgressBar::chunk {
        background-color: #2563EB;
        border-radius: 3px;
    }

    /* ── Splitter handle ── */
    QSplitter::handle {
        background-color: #E5E7EB;
        width: 1px;
    }

    /* ── Table/grid header labels ── */
    QLabel#th {
        color: #6B7280;
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(self.STYLE)

        self._img_bytes  = [None, None]
        self._img_px     = [None, None]
        self._img_sizes  = [(0, 0), (0, 0)]
        self._results    = [None, None]
        self._workers    = [None, None]
        self._pending    = 0
        self._view_mode  = ["harris", "harris"]

        self._match_result   = None
        self._match_worker   = None
        self._match_view     = False
        self._match_detector = "harris"
        self._match_method   = "ssd"

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    # ── Stable show/hide helpers (keep widget space reserved) ────────────────
    @staticmethod
    def _stable_hide(widget: QWidget):
        """Hide content but preserve layout space so nothing shifts."""
        widget.setMaximumHeight(0)
        widget.setVisible(False)

    @staticmethod
    def _stable_show(widget: QWidget, fixed_height: int = None):
        """Restore widget to its natural size (or a fixed height)."""
        if fixed_height is not None:
            widget.setFixedHeight(fixed_height)
        else:
            widget.setMaximumHeight(16777215)   # Qt's QWIDGETSIZE_MAX
        widget.setVisible(True)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 14, 16, 14)

        # Parameters
        root.addWidget(self._param_box())

        # ── Image panels ─────────────────────────────────────────────────────
        img_row = QHBoxLayout()
        img_row.setSpacing(16)

        self._lbl_img    = [_image_label(), _image_label()]
        self._btn_load   = [QPushButton("📂  Load Image A"),
                            QPushButton("📂  Load Image B")]
        self._btn_harris = [QPushButton("Harris"), QPushButton("Harris")]
        self._btn_lambda = [QPushButton("λ−"),     QPushButton("λ−")]

        for i in range(2):
            card = QGroupBox(f"Image {'AB'[i]}")
            cl   = QVBoxLayout(card)
            cl.setSpacing(8)
            cl.setContentsMargins(10, 14, 10, 10)

            # Fixed-size image holder wrapped in a centering widget
            holder = QWidget()
            holder.setFixedSize(IMAGE_PANEL_W, IMAGE_PANEL_H)
            hl = QVBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(self._lbl_img[i])
            cl.addWidget(holder, alignment=Qt.AlignHCenter)

            # Toggle row
            toggle_row = QHBoxLayout()
            toggle_row.setSpacing(6)
            self._btn_harris[i].setEnabled(False)
            self._btn_lambda[i].setEnabled(False)
            self._btn_harris[i].setObjectName("active")
            self._btn_harris[i].clicked.connect(lambda _, idx=i: self._set_view(idx, "harris"))
            self._btn_lambda[i].clicked.connect(lambda _, idx=i: self._set_view(idx, "lambda"))
            toggle_row.addStretch()
            toggle_row.addWidget(self._btn_harris[i])
            toggle_row.addWidget(self._btn_lambda[i])
            toggle_row.addStretch()
            cl.addLayout(toggle_row)

            # Load button
            self._btn_load[i].setObjectName("load")
            self._btn_load[i].clicked.connect(lambda _, idx=i: self._on_load(idx))
            cl.addWidget(self._btn_load[i])

            img_row.addWidget(card)

        root.addLayout(img_row)

        # ── Action row: Run + Match ───────────────────────────────────────────
        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        self._btn_run = QPushButton("▶  Run Description")
        self._btn_run.setObjectName("primary")
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run)
        action_row.addWidget(self._btn_run)

        self._btn_match = QPushButton("⇌  Match Features")
        self._btn_match.setObjectName("primary")
        self._btn_match.setEnabled(False)
        self._btn_match.clicked.connect(self._on_match)
        action_row.addWidget(self._btn_match)

        action_row.addStretch()
        root.addLayout(action_row)

        # ── Match overlay controls ────────────────────────────────────────────
        self._match_controls = QWidget()
        mc = QHBoxLayout(self._match_controls)
        mc.setContentsMargins(0, 0, 0, 0)
        mc.setSpacing(6)

        mc.addWidget(_pill_label("Detector"))
        self._btn_m_harris = QPushButton("Harris")
        self._btn_m_lambda = QPushButton("λ−")
        self._btn_m_harris.setObjectName("active")
        self._btn_m_harris.clicked.connect(lambda: self._set_match_detector("harris"))
        self._btn_m_lambda.clicked.connect(lambda: self._set_match_detector("lambda"))
        mc.addWidget(self._btn_m_harris)
        mc.addWidget(self._btn_m_lambda)

        mc.addSpacing(18)

        mc.addWidget(_pill_label("Method"))
        self._btn_m_ssd = QPushButton("SSD")
        self._btn_m_ncc = QPushButton("NCC")
        self._btn_m_ssd.setObjectName("active")
        self._btn_m_ssd.clicked.connect(lambda: self._set_match_method("ssd"))
        self._btn_m_ncc.clicked.connect(lambda: self._set_match_method("ncc"))
        mc.addWidget(self._btn_m_ssd)
        mc.addWidget(self._btn_m_ncc)

        mc.addSpacing(18)

        self._btn_back = QPushButton("✕  Back to Keypoints")
        self._btn_back.clicked.connect(self._exit_match_view)
        mc.addWidget(self._btn_back)
        mc.addStretch()

        self._stable_hide(self._match_controls)
        root.addWidget(self._match_controls)

        # ── Progress bar ──────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setFixedHeight(4)
        self._progress.setVisible(False)
        self._progress.setMaximumHeight(0)   # collapse without shifting layout
        root.addWidget(self._progress)

        # ── Stats bars ────────────────────────────────────────────────────────
        root.addWidget(self._desc_stats_bar())
        root.addWidget(self._match_stats_bar())

        # ── Status ────────────────────────────────────────────────────────────
        self._lbl_status = QLabel(
            "Ready — load two images to begin." if BACKEND_AVAILABLE
            else "⚠  cv_backend not found – build the C++ module first."
        )
        self._lbl_status.setObjectName("status" if BACKEND_AVAILABLE else "statusError")
        root.addWidget(self._lbl_status)

    # ── Parameter box ────────────────────────────────────────────────────────

    def _param_box(self) -> QGroupBox:
        box  = QGroupBox("Parameters")
        grid = QGridLayout(box)
        grid.setSpacing(8)
        grid.setContentsMargins(12, 16, 12, 10)

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

        params = [
            ("k",          self._sp_k),
            ("Block Size", self._sp_block),
            ("Sigma (σ)",  self._sp_sigma),
            ("Threshold",  self._sp_threshold),
            ("NMS Radius", self._sp_nms),
            ("Octaves",    self._sp_octaves),
        ]
        for i, (name, widget) in enumerate(params):
            lbl = QLabel(name)
            lbl.setStyleSheet("color: #6B7280; font-size: 12px;")
            grid.addWidget(lbl,    0, 2 * i,     Qt.AlignRight)
            grid.addWidget(widget, 0, 2 * i + 1)

        return box

    # ── Description stats bar ────────────────────────────────────────────────

    def _desc_stats_bar(self) -> QGroupBox:
        box = QGroupBox("Description Results")
        h   = QHBoxLayout(box)
        h.setSpacing(14)
        h.setContentsMargins(12, 14, 12, 10)

        def stat(label):
            lbl_txt = QLabel(label + ":")
            lbl_txt.setStyleSheet("color: #6B7280; font-size: 12px;")
            h.addWidget(lbl_txt)
            lbl = QLabel("—")
            lbl.setObjectName("stat")
            h.addWidget(lbl)
            return lbl

        self._stat = {
            "A_harris": stat("A · Harris"),
            "A_lambda": stat("A · λ−"),
            "B_harris": stat("B · Harris"),
            "B_lambda": stat("B · λ−"),
        }
        h.addStretch()
        return box

    # ── Match stats bar ──────────────────────────────────────────────────────

    def _match_stats_bar(self) -> QGroupBox:
        self._match_stats_box = QGroupBox("Matching Results")
        g = QGridLayout(self._match_stats_box)
        g.setSpacing(8)
        g.setContentsMargins(12, 14, 12, 10)

        def stat_label():
            lbl = QLabel("—")
            lbl.setObjectName("stat")
            return lbl

        headers = ["", "SSD Matches", "SSD Time", "NCC Matches", "NCC Time"]
        for c, h in enumerate(headers):
            lbl = QLabel(h)
            lbl.setObjectName("th")
            g.addWidget(lbl, 0, c)

        self._mstat_ssd_count_h = stat_label()
        self._mstat_ssd_time_h  = stat_label()
        self._mstat_ncc_count_h = stat_label()
        self._mstat_ncc_time_h  = stat_label()
        row_lbl = QLabel("Harris")
        row_lbl.setStyleSheet("color: #374151; font-weight: 600;")
        g.addWidget(row_lbl, 1, 0)
        g.addWidget(self._mstat_ssd_count_h, 1, 1)
        g.addWidget(self._mstat_ssd_time_h,  1, 2)
        g.addWidget(self._mstat_ncc_count_h, 1, 3)
        g.addWidget(self._mstat_ncc_time_h,  1, 4)

        self._mstat_ssd_count_l = stat_label()
        self._mstat_ssd_time_l  = stat_label()
        self._mstat_ncc_count_l = stat_label()
        self._mstat_ncc_time_l  = stat_label()
        row_lbl2 = QLabel("λ−")
        row_lbl2.setStyleSheet("color: #374151; font-weight: 600;")
        g.addWidget(row_lbl2, 2, 0)
        g.addWidget(self._mstat_ssd_count_l, 2, 1)
        g.addWidget(self._mstat_ssd_time_l,  2, 2)
        g.addWidget(self._mstat_ncc_count_l, 2, 3)
        g.addWidget(self._mstat_ncc_time_l,  2, 4)

        self._stable_hide(self._match_stats_box)
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
        self._lbl_status.setObjectName("status")
        self._lbl_status.style().unpolish(self._lbl_status)
        self._lbl_status.style().polish(self._lbl_status)

        self._results[idx]   = None
        self._match_result   = None
        self._match_view     = False
        self._stable_hide(self._match_controls)
        self._stable_hide(self._match_stats_box)
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
        self._stable_show(self._progress, fixed_height=4)
        self._set_status("Running description…", "status")
        self._pending      = 2
        self._match_result = None
        self._match_view   = False
        self._stable_hide(self._match_controls)
        self._stable_hide(self._match_stats_box)

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
            f"{result.harris_count} pts  {result.harris_time_ms:.1f} ms")
        self._stat[f"{key}_lambda"].setText(
            f"{result.lambda_count} pts  {result.lambda_time_ms:.1f} ms")

        self._pending -= 1
        if self._pending == 0:
            self._stable_hide(self._progress)
            self._btn_run.setEnabled(True)
            self._btn_match.setEnabled(True)
            self._set_status("Done — click Match Features to compare.", "statusOk")

    def _on_error(self, msg: str):
        self._pending = 0
        self._stable_hide(self._progress)
        self._btn_run.setEnabled(True)
        self._set_status(f"Error: {msg}", "statusError")

    # ── Matching ─────────────────────────────────────────────────────────────

    def _on_match(self):
        if self._results[0] is None or self._results[1] is None:
            return
        self._btn_match.setEnabled(False)
        self._btn_run.setEnabled(False)
        self._stable_show(self._progress, fixed_height=4)
        self._set_status("Running matching…", "status")

        w = MatchingWorker(self._results[0], self._results[1])
        w.finished.connect(self._on_match_done)
        w.error.connect(self._on_match_error)
        self._match_worker = w
        w.start()

    def _on_match_done(self, match_result):
        self._match_result = match_result
        self._stable_hide(self._progress)
        self._btn_run.setEnabled(True)
        self._btn_match.setEnabled(True)

        r = match_result
        self._mstat_ssd_count_h.setText(str(r.harris_ssd_match_count))
        self._mstat_ssd_time_h.setText(f"{r.harris_ssd_time_ms:.2f} ms")
        self._mstat_ncc_count_h.setText(str(r.harris_ncc_match_count))
        self._mstat_ncc_time_h.setText(f"{r.harris_ncc_time_ms:.2f} ms")

        self._mstat_ssd_count_l.setText(str(r.lambda_ssd_match_count))
        self._mstat_ssd_time_l.setText(f"{r.lambda_ssd_time_ms:.2f} ms")
        self._mstat_ncc_count_l.setText(str(r.lambda_ncc_match_count))
        self._mstat_ncc_time_l.setText(f"{r.lambda_ncc_time_ms:.2f} ms")

        self._stable_show(self._match_stats_box)

        self._match_view     = True
        self._match_detector = "harris"
        self._match_method   = "ssd"
        self._stable_show(self._match_controls)
        self._refresh_match_toggles()
        self._refresh_match_overlay()
        self._set_status("Matching complete — toggle detector / method to compare.", "statusOk")

    def _on_match_error(self, msg: str):
        self._stable_hide(self._progress)
        self._btn_run.setEnabled(True)
        self._btn_match.setEnabled(True)
        self._set_status(f"Matching error: {msg}", "statusError")

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
        self._stable_hide(self._match_controls)
        for i in range(2):
            self._refresh_view(i)
        self._set_status("Keypoint view restored.", "status")

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

        n      = min(len(kptsA), len(kptsB))
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
        if self._match_view:
            self._exit_match_view()
        self._view_mode[idx] = mode
        self._refresh_view(idx)
        self._update_toggle_style(idx)

    def _refresh_view(self, idx: int):
        if self._match_view:
            return
        if self._results[idx] is None:
            if self._img_px[idx]:
                self._show_pixmap(self._lbl_img[idx], self._img_px[idx])
            return
        vis = (self._results[idx].harris_vis if self._view_mode[idx] == "harris"
               else self._results[idx].lambda_vis)
        self._show_pixmap(self._lbl_img[idx], _bytes_to_pixmap(bytes(vis)))

    def _show_pixmap(self, label: QLabel, px: QPixmap):
        """Scale to fit fixed-size label, keeping aspect ratio."""
        label.setText("")
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

    def _set_status(self, text: str, style_name: str):
        self._lbl_status.setText(text)
        self._lbl_status.setObjectName(style_name)
        self._lbl_status.style().unpolish(self._lbl_status)
        self._lbl_status.style().polish(self._lbl_status)

    # resizeEvent not needed anymore since image panels are fixed size


# ─────────────────────────────────────────────────────────────────────────────
#  Tiny helper – pill-shaped section label
# ─────────────────────────────────────────────────────────────────────────────
def _pill_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        "QLabel { background-color: #F3F4F6; color: #6B7280; "
        "border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 600; "
        "letter-spacing: 0.04em; }"
    )
    return lbl