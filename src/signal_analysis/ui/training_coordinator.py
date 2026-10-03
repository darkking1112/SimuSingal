"""Window-level serialization for training-page operations."""
from dataclasses import dataclass
import uuid

from PySide6 import QtCore


@dataclass
class _Reservation:
    token: str
    owner: str
    kind: str
    task_id: str | None = None


class TrainingCoordinator(QtCore.QObject):
    """Allow one training run at a time across both training pages."""

    slot_changed = QtCore.Signal(object)

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._slot = None

    @property
    def slot(self):
        if self._slot is None:
            return None
        return {"owner": self._slot.owner, "kind": self._slot.kind,
                "task_id": self._slot.task_id}

    def owner_busy(self, owner):
        return self._slot is not None and self._slot.owner == owner

    def acquire(self, owner, kind):
        """占槽；返回 ``(token, None)``，被占用时返回 ``(None, 拒绝原因)``。"""
        if self._slot is not None:
            return None, (f"「{self._slot.owner}」正在执行"
                          f"{self._kind_text(self._slot.kind)}，请等待结束")
        reservation = _Reservation(uuid.uuid4().hex, owner, kind)
        self._slot = reservation
        self.slot_changed.emit(self.slot)
        return reservation.token, None

    def bind_task(self, token, task):
        reservation = self._reservation(token)
        if reservation is None:
            raise RuntimeError("训练运行槽已失效，无法关联任务")
        if task.owner != reservation.owner:
            raise ValueError("任务归属与训练运行槽不一致")
        reservation.task_id = task.id

    def release(self, token):
        reservation = self._reservation(token)
        if reservation is None:
            return False
        self._slot = None
        self.slot_changed.emit(None)
        return True

    @staticmethod
    def _kind_text(kind):
        return {"training": "模型训练"}.get(kind, "训练任务")

    def _reservation(self, token):
        if self._slot is not None and self._slot.token == token:
            return self._slot
        return None
