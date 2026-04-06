import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.abspath(os.path.join(current_dir, '..', 'build'))
sys.path.append(build_dir)
sys.path.append(os.path.join(build_dir, 'Release')) 

from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout

from extraction_tab import MainTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pipeline Test")
        self.resize(400, 200)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.tabs = QTabWidget()
        
        self.tabs.addTab(MainTab(), "Main")


        root.addWidget(self.tabs)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())