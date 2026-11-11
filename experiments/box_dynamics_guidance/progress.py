"""Small dependency-free terminal progress display, also readable in log files."""
import sys
import time


def duration(seconds):
    if seconds is None:
        return "--:--"
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return "{}:{:02d}:{:02d}".format(hours, minutes, seconds) if hours else "{:02d}:{:02d}".format(minutes, seconds)


class Progress:
    def __init__(self, total, label, unit, stream=None):
        if total <= 0:
            raise ValueError("Progress total must be positive")
        self.total, self.label, self.unit = total, label, unit
        self.stream = stream if stream is not None else sys.stderr
        self.terminal = bool(getattr(self.stream, "isatty", lambda: False)())
        self.started = time.monotonic()
        self.initial = None
        self.completed = 0
        self.closed = False
        self.width = 0

    def update(self, completed, detail=""):
        if self.closed:
            return
        self.completed = min(self.total, max(0, completed))
        now = time.monotonic()
        if self.initial is None:
            self.initial = self.completed
            self.started = now
        elapsed = now - self.started
        newly_completed = self.completed - self.initial
        eta = ((self.total - self.completed) * elapsed / newly_completed
               if newly_completed > 0 else None)
        if self.completed >= self.total:
            eta = 0
        fraction = self.completed / self.total
        filled = min(24, int(24 * fraction))
        bar = "#" * filled + "-" * (24 - filled)
        amount = "{:,}/{:,}".format(self.completed, self.total)
        line = "{} [{}] {:5.1f}% {} {} | elapsed {} | ETA {}".format(
            self.label, bar, fraction * 100, amount, self.unit, duration(elapsed), duration(eta))
        if detail:
            line += " | " + detail
        if self.terminal:
            self.stream.write("\r" + line.ljust(self.width))
            self.width = max(self.width, len(line))
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def close(self, detail=""):
        if self.closed:
            return
        # An early stop or interruption must not masquerade as 100% completed.
        if detail:
            self.update(self.completed, detail)
        if self.terminal:
            self.stream.write("\n")
            self.stream.flush()
        self.closed = True
