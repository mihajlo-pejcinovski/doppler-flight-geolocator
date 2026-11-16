"""``dopplergeo-gui [media-or-project]``: start the desktop app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dopplergeo-gui", description="Doppler Flight Geolocator desktop app")
    ap.add_argument("path", nargs="?", help="media file or project folder to open")
    ap.add_argument("--cache", help="flight cache folder (default ~/.cache/dopplergeo/flights)")
    ap.add_argument("--no-basemap", action="store_true", help="plain background instead of online map tiles")
    args, qt_args = ap.parse_known_args(argv)

    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtWidgets import QApplication

    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)  # before QtWebEngine starts
    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("dopplergeo")
    app.setOrganizationName("dopplergeo")

    from .main_window import FROM_SETTINGS, MainWindow
    from .settings import apply_saved_credentials

    apply_saved_credentials()  # keys saved in the Settings dialog -> the providers' environment variables

    w = MainWindow(style_url=None if args.no_basemap else FROM_SETTINGS, cache_dir=args.cache)
    # Destroy the window (and its QtWebEngine page) inside the event loop on close:
    # left to Python at exit it may outlive the QApplication and crash on the way out.
    w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    w.show()
    if args.path:
        p = Path(args.path)
        if p.is_dir():
            w.open_project(p)
        else:
            w.open_media(p)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
