from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit
import cv_backend

class MainTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # Inputs layout
        h_layout = QHBoxLayout()
        self.input1 = QLineEdit()
        self.input2 = QLineEdit()
        
        h_layout.addWidget(self.input1)
        h_layout.addWidget(QLabel("+"))
        h_layout.addWidget(self.input2)
        layout.addLayout(h_layout)

        # Button
        self.btn = QPushButton("Calculate in C++")
        self.btn.clicked.connect(self.run_test)
        layout.addWidget(self.btn)

        # Result Label
        self.result_label = QLabel("Result: ")
        layout.addWidget(self.result_label)

    def run_test(self):
        try:
            # Grab the numbers from UI
            a = int(self.input1.text())
            b = int(self.input2.text())
            
            # Call the C++ function
            res = cv_backend.add_numbers(a, b)
            
            # Update the UI
            self.result_label.setText(f"Result: {res}")
        except ValueError:
            self.result_label.setText("Result: Please enter valid integers.")