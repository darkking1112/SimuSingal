"""复用控件：Hz 单位自适应输入框与带外置单位标签的 Spin 工厂。"""

from PySide6 import QtWidgets


class UnitSpinBox(QtWidgets.QDoubleSpinBox):
    """Stores Hz internally; displays the value in kHz / MHz / GHz.

    The unit is shown in a QLabel beside the box (``unit_label``), not inside
    the input; it switches automatically when the value crosses a boundary.
    """

    _UNITS = (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3), ("Hz", 1.0))

    def __init__(self, minimum, maximum, value, decimals):
        self._scale = 1.0  # 必须先于 setRange/setDecimals：Qt 内部会调用 textFromValue
        super().__init__()
        self.setRange(minimum, maximum)
        self.setDecimals(decimals)
        self.unit_label = QtWidgets.QLabel("Hz")
        self.setValue(value)
        self._sync_unit()
        self.setAccelerated(True)
        self.valueChanged.connect(self._sync_unit)

    def _sync_unit(self, *_):
        unit, scale = self._UNITS[-1]
        for candidate, factor in self._UNITS:
            if abs(self.value()) >= factor:
                unit, scale = candidate, factor
                break
        if scale != self._scale:
            self._scale = scale
            self.unit_label.setText(unit)
            self.setSingleStep(scale)
            self.update()

    def textFromValue(self, value):
        return f"{value / self._scale:.{self.decimals()}f}"

    def valueFromText(self, text):
        return float(text.strip()) * self._scale


def _unit_row(spin):
    """Place a unit label beside a spin box (unit outside the input)."""
    container = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(spin)
    row.addWidget(spin.unit_label)
    return container


def _plain_spin(minimum, maximum, value, decimals, unit=None):
    """Plain double spin; with ``unit`` the label is placed outside the box."""
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setValue(value)
    if unit is None:
        return spin
    spin.unit_label = QtWidgets.QLabel(unit)
    return _unit_row(spin), spin


def _freq_spin(minimum, maximum, value, decimals):
    """Frequency spin in Hz with a kHz / MHz / GHz unit label outside."""
    spin = UnitSpinBox(minimum, maximum, value, decimals)
    return _unit_row(spin), spin
