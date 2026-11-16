"""Background jobs (spec "Threading"): heavy work runs in ``QThreadPool``
workers that post progress and log signals; the UI never blocks and a job can
be cancelled. Questions a job must ask the user (e.g. "download 3 GB?") are
answered on the UI thread while the worker waits."""

from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class Cancelled(Exception):
    """Raised inside a job when the user cancels it."""


class JobContext:
    """Handed to the job function: logging, progress, cancellation, questions."""

    def __init__(self, signals: WorkerSignals):
        self._signals = signals
        self._cancel = threading.Event()
        self._answer: bool = False
        self._answered = threading.Event()

    # -- called from the job thread --------------------------------------
    def log(self, message: str) -> None:
        self.check()
        self._signals.log.emit(str(message))

    def progress(self, stage: str, done: int, total: int) -> None:
        self.check()
        self._signals.progress.emit(stage, int(done), int(total))

    def check(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

    def ask(self, question: str) -> bool:
        """Block the job until the UI answers yes/no (False if cancelled)."""
        self._answered.clear()
        self._signals.question.emit(question)
        while not self._answered.wait(0.1):
            if self._cancel.is_set():
                return False
        return self._answer

    # -- called from the UI thread ---------------------------------------
    def cancel(self) -> None:
        self._cancel.set()

    def answer(self, yes: bool) -> None:
        self._answer = bool(yes)
        self._answered.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()


class WorkerSignals(QObject):
    log = Signal(str)
    progress = Signal(str, object, object)  # object: byte counts of multi-GB downloads overflow a C++ int
    question = Signal(str)
    finished = Signal(object)        # the job's return value
    failed = Signal(str, str)        # message, traceback
    cancelled = Signal()
    done = Signal()                  # always last


class Worker(QRunnable):
    def __init__(self, name: str, fn: Callable[[JobContext], Any]):
        super().__init__()
        self.name = name
        self.fn = fn
        self.signals = WorkerSignals()
        self.ctx = JobContext(self.signals)
        self.exited = threading.Event()  # set when run() has fully returned
        self.setAutoDelete(False)  # the Python side keeps the reference (JobRunner._alive)

    def run(self) -> None:
        try:
            result = self.fn(self.ctx)
        except Cancelled:
            self.signals.cancelled.emit()
        except Exception as e:  # report every failure to the UI, never crash the pool
            self.signals.failed.emit(f"{type(e).__name__}: {e}", traceback.format_exc())
        else:
            if self.ctx.cancelled:
                self.signals.cancelled.emit()
            else:
                self.signals.finished.emit(result)
        finally:
            self.signals.done.emit()
            self.exited.set()


class _Relay(QObject):
    """Receives a worker's signals on the UI thread (it lives there, so Qt queues
    the cross-thread calls) and forwards them to the callbacks."""

    def __init__(self, runner: JobRunner, w: Worker, on_result, on_error, on_log, on_progress, on_question):
        super().__init__(runner)
        self.runner, self.w = runner, w
        self.on_result, self.on_error = on_result, on_error
        self.on_log, self.on_progress, self.on_question = on_log, on_progress, on_question
        sig = w.signals
        sig.finished.connect(self.finished)
        sig.failed.connect(self.failed)
        sig.log.connect(self.log)
        sig.progress.connect(self.progress)
        sig.question.connect(self.question)
        sig.done.connect(self.done)

    def _current(self) -> bool:
        return self.runner.active.get(self.w.name) is self.w and not self.w.ctx.cancelled

    @Slot(object)
    def finished(self, result) -> None:
        if self._current():
            self.on_result(result)

    @Slot(str, str)
    def failed(self, message: str, tb: str) -> None:
        if self._current():
            self.on_error(message, tb)

    @Slot(str)
    def log(self, message: str) -> None:
        if self.on_log:
            self.on_log(message)

    @Slot(str, object, object)
    def progress(self, stage: str, done: int, total: int) -> None:
        if self.on_progress:
            self.on_progress(stage, done, total)

    @Slot(str)
    def question(self, text: str) -> None:
        self.w.ctx.answer(bool(self.on_question(text)) if self.on_question else False)

    @Slot()
    def done(self) -> None:
        self.runner._finish(self.w)
        self.deleteLater()


class JobRunner(QObject):
    """Runs one job per name at a time; starting a job with the same name
    cancels the previous one (its result is dropped)."""

    started = Signal(str)
    stopped = Signal(str)

    def __init__(self, pool: QThreadPool | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self.pool = pool or QThreadPool.globalInstance()
        self.active: dict[str, Worker] = {}
        # Every started worker stays referenced until its thread has left run():
        # freeing a QRunnable (or its signals object) while the pool thread is still
        # returning from it crashes the process.
        self._alive: list[Worker] = []

    def start(self, name: str, fn: Callable[[JobContext], Any], *, on_result: Callable[[Any], None],
              on_error: Callable[[str, str], None], on_log: Callable[[str], None] | None = None,
              on_progress: Callable[[str, int, int], None] | None = None,
              on_question: Callable[[str], bool] | None = None) -> Worker:
        if name in self.active:
            self.active[name].ctx.cancel()
        self._alive = [x for x in self._alive if not x.exited.is_set()]
        w = Worker(name, fn)
        self._alive.append(w)
        _Relay(self, w, on_result, on_error, on_log, on_progress, on_question)
        self.active[name] = w
        self.started.emit(name)
        self.pool.start(w)
        return w

    def _finish(self, w: Worker) -> None:
        if self.active.get(w.name) is w:
            del self.active[w.name]
            self.stopped.emit(w.name)

    def cancel(self, name: str | None = None) -> None:
        for n, w in list(self.active.items()):
            if name is None or n == name:
                w.ctx.cancel()

    def busy(self, name: str | None = None) -> bool:
        return bool(self.active) if name is None else name in self.active

    def wait(self, msecs: int = 30_000) -> bool:
        """Block until every job this runner started has left its thread."""
        deadline = time.monotonic() + msecs / 1000
        for w in list(self._alive):
            if not w.exited.wait(max(0.0, deadline - time.monotonic())):
                return False
        self._alive = [x for x in self._alive if not x.exited.is_set()]
        return True

    def shutdown(self, msecs: int = 30_000) -> bool:
        """Cancel everything and wait (window close, tests)."""
        self.cancel()
        return self.wait(msecs)
