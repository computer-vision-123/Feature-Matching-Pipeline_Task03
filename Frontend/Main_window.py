import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.abspath(os.path.join(current_dir, '..', 'build'))
sys.path.append(build_dir)
sys.path.append(os.path.join(build_dir, 'Release'))

from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt

from extraction_tab import MainTab

# ── Global application style ──────────────────────────────────────────────────
APP_STYLE = """
QMainWindow {
    background-color: #F9FAFB;
}

QTabWidget::pane {
    border: none;
    background-color: #F9FAFB;
}

QTabBar::tab {
    background-color: transparent;
    color: #9CA3AF;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
    font-weight: 500;
    padding: 10px 22px;
    border: none;
    border-bottom: 2px solid transparent;
    margin-right: 2px;
}

QTabBar::tab:selected {
    color: #111827;
    border-bottom: 2px solid #2563EB;
    font-weight: 600;
}

QTabBar::tab:hover:!selected {
    color: #374151;
    background-color: #F3F4F6;
    border-radius: 6px 6px 0 0;
}

QTabWidget > QTabBar {
    background-color: #FFFFFF;
    border-bottom: 1px solid #E5E7EB;
}

/* Scroll bars – subtle */
QScrollBar:vertical {
    background: #F9FAFB;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #D1D5DB;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #9CA3AF; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal {
    background: #F9FAFB;
    height: 8px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #D1D5DB;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover { background: #9CA3AF; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feature Matching Pipeline")
        self.resize(1100, 860)
        self.setMinimumSize(900, 700)
        self.setStyleSheet(APP_STYLE)

        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)   # cleaner tab bar on macOS / modern look

        self.main_tab = MainTab()
        self.tabs.addTab(self.main_tab, "Main")

        root.addWidget(self.tabs)


if __name__ == "__main__":
    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Use a clean default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())