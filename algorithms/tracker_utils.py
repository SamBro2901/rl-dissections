"""Shared CodeCarbon task-timing helpers used by every algorithm's train()
loop (see algorithms/sac.py and algorithms/mbpo.py). Kept in one place so
adding a new algorithm doesn't mean re-deriving the unique-task-name-per-call
workaround described in README.md ("Why epoch-level tagging..." section)."""
from __future__ import annotations

from typing import Dict


class NullTask:
    """No-op context manager used when tracker is None (e.g. dry-run/debug)."""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TrackerTask:
    """Wraps tracker.start_task/stop_task as a context manager and stores the
    returned EmissionsData's energy_consumed (kWh) into a results dict keyed
    by a stable prefix (unique task *names* per call avoid a CodeCarbon bug
    where reusing a task name corrupts internal accounting -- see README)."""

    def __init__(self, tracker, prefix: str, counter: int, energy_log: Dict[str, float]):
        self.tracker = tracker
        self.name = f"{prefix}_{counter}"
        self.prefix = prefix
        self.energy_log = energy_log

    def __enter__(self):
        if self.tracker is not None:
            self.tracker.start_task(self.name)
        return self

    def __exit__(self, *a):
        if self.tracker is not None:
            data = self.tracker.stop_task(self.name)
            self.energy_log[self.prefix] = self.energy_log.get(self.prefix, 0.0) + (
                data.energy_consumed if data is not None else 0.0
            )
        return False
