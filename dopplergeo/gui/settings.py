"""Settings dialog (spec "GUI / Panels / Settings"): data-source credentials in the
OS keychain, map tile style and flight cache folder in QSettings.

Credentials reach the providers through the environment variables they already
read (``ADSBLOL_API_KEY``, ``OPENSKY_CLIENT_ID``, ``OPENSKY_CLIENT_SECRET``):
``apply_saved_credentials`` fills them from the keychain at start-up, where an
environment variable set by the user wins; saving the dialog sets them.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

SERVICE = "dopplergeo"
# keychain entry -> environment variable the provider reads
SECRETS = {
    "adsblol_api_key": "ADSBLOL_API_KEY",
    "opensky_client_id": "OPENSKY_CLIENT_ID",
    "opensky_client_secret": "OPENSKY_CLIENT_SECRET",
}


def _keyring():
    try:
        import keyring
        keyring.get_keyring()
        return keyring
    except Exception:  # not installed, or no usable backend (e.g. headless without a secret service)
        return None


def load_secret(name: str) -> str:
    kr = _keyring()
    if kr is None:
        return ""
    try:
        return kr.get_password(SERVICE, name) or ""
    except Exception:
        return ""


def save_secret(name: str, value: str) -> bool:
    """Store (or, for an empty value, delete) one secret. False if no keychain works."""
    kr = _keyring()
    if kr is None:
        return False
    try:
        if value:
            kr.set_password(SERVICE, name, value)
        else:
            try:
                kr.delete_password(SERVICE, name)
            except Exception:
                pass  # nothing stored
        return True
    except Exception:
        return False


def apply_saved_credentials(environ=os.environ) -> list[str]:
    """Fill the providers' environment variables from the keychain, keeping any the
    user set. Returns the variables filled."""
    filled = []
    for name, env in SECRETS.items():
        if environ.get(env):
            continue
        v = load_secret(name)
        if v:
            environ[env] = v
            filled.append(env)
    return filled


class SettingsDialog(QDialog):
    def __init__(self, qsettings: QSettings, default_style: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.qsettings = qsettings
        self.keychain_ok = _keyring() is not None

        self.adsblol_key = QLineEdit(load_secret("adsblol_api_key") or os.environ.get("ADSBLOL_API_KEY", ""))
        self.adsblol_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.adsblol_key.setPlaceholderText("optional: without a key adsb.lol allows 500 requests/day per IP")
        self.os_id = QLineEdit(load_secret("opensky_client_id") or os.environ.get("OPENSKY_CLIENT_ID", ""))
        self.os_secret = QLineEdit(load_secret("opensky_client_secret") or os.environ.get("OPENSKY_CLIENT_SECRET", ""))
        self.os_secret.setEchoMode(QLineEdit.EchoMode.Password)
        self.style = QLineEdit(str(qsettings.value("map/style_url", default_style) or ""))
        self.style.setPlaceholderText("empty: plain background, no online tiles")
        self.cache = QLineEdit(str(qsettings.value("cache/dir", "") or ""))
        self.cache.setPlaceholderText("default: ~/.cache/dopplergeo/flights")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        cache_row = QHBoxLayout()
        cache_row.addWidget(self.cache, 1)
        cache_row.addWidget(browse)
        show = QPushButton("Show keys")
        show.setCheckable(True)
        show.toggled.connect(self._show_keys)

        form = QFormLayout()
        form.addRow(QLabel("<b>Flight data</b>"))
        form.addRow("adsb.lol API key", self.adsblol_key)
        form.addRow("OpenSky client id", self.os_id)
        form.addRow("OpenSky client secret", self.os_secret)
        form.addRow("", show)
        form.addRow(QLabel("<b>Map and cache</b>"))
        form.addRow("Map tile style URL", self.style)
        form.addRow("Flight cache folder", cache_row)
        note = QLabel("Keys are stored in the system keychain." if self.keychain_ok else
                      "<span style='color:#c53030'>No system keychain available: keys apply to this session "
                      "only (or set the environment variables).</span>")
        note.setWordWrap(True)
        self.restart_note = QLabel("")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(note)
        lay.addWidget(self.restart_note)
        lay.addWidget(buttons)
        self.resize(560, self.sizeHint().height())

    def _show_keys(self, on: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password
        self.adsblol_key.setEchoMode(mode)
        self.os_secret.setEchoMode(mode)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Flight cache folder", self.cache.text())
        if d:
            self.cache.setText(d)

    def save(self, environ=os.environ) -> dict[str, bool]:
        """Store everything; returns which settings changed that need a restart."""
        values = {"adsblol_api_key": self.adsblol_key.text().strip(),
                  "opensky_client_id": self.os_id.text().strip(),
                  "opensky_client_secret": self.os_secret.text().strip()}
        for name, v in values.items():
            save_secret(name, v)
            env = SECRETS[name]
            if v:
                environ[env] = v
            else:
                environ.pop(env, None)
        changed = {}
        old_style = str(self.qsettings.value("map/style_url", "") or "")
        self.qsettings.setValue("map/style_url", self.style.text().strip())
        changed["style"] = old_style != self.style.text().strip()
        old_cache = str(self.qsettings.value("cache/dir", "") or "")
        self.qsettings.setValue("cache/dir", self.cache.text().strip())
        changed["cache"] = old_cache != self.cache.text().strip()
        return changed
