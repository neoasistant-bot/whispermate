import sys
from PyQt6.QtWidgets import QApplication
from src.ui.tray import TrayApp
from src.config import Config


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # no cerrar al cerrar settings

    config = Config()
    tray = TrayApp(config)
    tray.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
