"""Video panel: QMediaPlayer + QVideoWidget. Its position drives every other
panel's cursor (spec "GUI / Panels")."""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QStyle, QVBoxLayout, QWidget


def fmt_time(s: float) -> str:
    s = max(0.0, s)
    return f"{int(s // 60):d}:{s % 60:05.2f}"


class VideoPanel(QWidget):
    position_changed = Signal(float)   # seconds of media time
    mute_marked = Signal(float, float)  # a mute range marked while watching (start, end s)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.video = QVideoWidget(self)
        self.video.setMinimumSize(240, 135)
        self.player.setVideoOutput(self.video)

        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_btn.setEnabled(False)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.time = QLabel("0:00.00 / 0:00.00")
        # Mark speech, cars, ... while watching: first click starts a mute range at the
        # playhead, the second ends it.
        self.mute_btn = QPushButton("Mute from here")
        self.mute_btn.setToolTip("Start a mute range at the playhead (speech, traffic); click again to end it")
        self.mute_btn.setEnabled(False)
        self._mute_start: float | None = None
        bar = QHBoxLayout()
        bar.addWidget(self.play_btn)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.time)
        bar.addWidget(self.mute_btn)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.video, 1)
        lay.addLayout(bar)

        self.play_btn.clicked.connect(self.toggle)
        self.mute_btn.clicked.connect(self._mute_clicked)
        self.slider.sliderMoved.connect(lambda ms: self.seek(ms / 1000.0))
        self.player.positionChanged.connect(self._pos)
        self.player.durationChanged.connect(lambda ms: self.slider.setRange(0, int(ms)))
        self.player.playbackStateChanged.connect(self._state)

    def load(self, path: str) -> None:
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.play_btn.setEnabled(True)
        self.slider.setEnabled(True)
        self.mute_btn.setEnabled(True)
        self._mute_start = None
        self.mute_btn.setText("Mute from here")

    def toggle(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def position_s(self) -> float:
        return self.player.position() / 1000.0

    def _mute_clicked(self) -> None:
        t = self.position_s()
        if self._mute_start is None:
            self._mute_start = t
            self.mute_btn.setText(f"End mute here (from {fmt_time(t)})")
            return
        a, b = sorted((self._mute_start, t))
        self._mute_start = None
        self.mute_btn.setText("Mute from here")
        if b - a >= 0.2:
            self.mute_marked.emit(a, b)

    def seek(self, t: float) -> None:
        self.player.setPosition(int(max(0.0, t) * 1000))

    def _pos(self, ms: int) -> None:
        if not self.slider.isSliderDown():
            self.slider.setValue(ms)
        self.time.setText(f"{fmt_time(ms / 1000)} / {fmt_time(self.player.duration() / 1000)}")
        self.position_changed.emit(ms / 1000.0)

    def _state(self, st) -> None:
        icon = (QStyle.StandardPixmap.SP_MediaPause if st == QMediaPlayer.PlaybackState.PlayingState
                else QStyle.StandardPixmap.SP_MediaPlay)
        self.play_btn.setIcon(self.style().standardIcon(icon))
