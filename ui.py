from PyQt6 import QtWidgets, QtGui, QtCore
from PIL import ImageQt
import pygetwindow as gw
import sys
from lib import describe_ui  # Import the describe_ui function

class Window(QtWidgets.QWidget):
    screenshot_captured = QtCore.pyqtSignal(QtGui.QPixmap)
    def __init__(self):
        super().__init__()

        self.initUI()

        self.screenshot_captured.connect(self.set_screenshot)

    def initUI(self):
        self.setWindowTitle('Window Capture')

        # Create a dropdown for active windows
        self.dropdown = QtWidgets.QComboBox(self)
        self.dropdown.addItems([win.title for win in gw.getAllWindows() if win.title])

        # Create a button to capture the screenshot
        self.button = QtWidgets.QPushButton('Screenshot', self)
        self.button.clicked.connect(self.capture_screenshot)

        # Create a QLabel to display the screenshot
        self.image_label = QtWidgets.QLabel(self)

        # Create a layout and add the dropdown, button, and QLabel to it
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.dropdown)
        layout.addWidget(self.button)
        layout.addWidget(self.image_label)

    def capture_screenshot(self):
        try:
            results = describe_ui(self.dropdown.currentText())

            # Process results list
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Probs object for classification outputs
                obb = result.obb  # Oriented boxes object for OBB outputs
                result.show()  # display to screen
                result.save(filename="result.jpg")  # save to disk
        except Exception as e:
            print(f"An error occurred: {e}")

    @QtCore.pyqtSlot(QtGui.QPixmap)
    def set_screenshot(self, pixmap):
        # Set the pixmap of the QLabel
        self.image_label.setPixmap(pixmap)

def main():
    app = QtWidgets.QApplication(sys.argv)

    win = Window()
    win.show()

    sys.exit(app.exec())

if __name__ == '__main__':
    main()