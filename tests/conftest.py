import os

import pytest

from scenario import make_recording

# GUI tests run headless; QtWebEngine needs shared GL contexts before the
# QApplication (created by pytest-qt's qapp fixture) exists.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtCore import QCoreApplication, Qt

    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
except ImportError:  # GUI extra not installed: GUI tests skip themselves
    pass


@pytest.fixture(scope="session")
def recording(tmp_path_factory):
    """(media file, flight cache dir, tmp dir) of the shared two-aircraft scenario."""
    return make_recording(tmp_path_factory.mktemp("locate"))
