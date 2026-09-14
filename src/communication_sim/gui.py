"""Independent generic event simulation desktop."""
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from common.gui import DesktopWindow
from .storage import Workspace
from .tasks import run_job

class MainWindow(DesktopWindow):
    run_task = staticmethod(run_job)

    def __init__(self, workspace):
        self.events = []
        super().__init__(Workspace(workspace), "通信仿真实验 · CommunicationSim",
                         "通用队列模型 · 场景参数 · 事件回放")
        # 标签页由本项目统一注册；公共外壳 DesktopWindow 不添加任何页面。
        self.tabs.addTab(self.build_simulation(), "事件仿真")
        self.tabs.addTab(self.build_history(), "运行记录")
        self.refresh_history()

    def build_history(self):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(QtWidgets.QLabel("最近 100 次成功运行 · 双击查看或回放"))
        self.history = QtWidgets.QListWidget()
        self.history.itemDoubleClicked.connect(self.open_history)
        layout.addWidget(self.history)
        return box

    def refresh_history(self):
        self.history.clear()
        for result in self.workspace.list_runs():
            item = QtWidgets.QListWidgetItem(
                f"{result['created_at'][:19]}  ·  {result['kind']}  ·  {result['id'][:8]}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, result["id"])
            self.history.addItem(item)

    def open_history(self, item):
        try:
            self.display_result(self.workspace.get_run(item.data(QtCore.Qt.ItemDataRole.UserRole)))
            self.status.setText("已读取历史结果")
        except (ValueError, OSError) as exc:
            self.status.setText(f"无法读取历史结果：{exc}")

    def job_buttons(self):
        return (self.sim_button,)

    def result_ready(self, result):
        self.refresh_history()
        self.display_result(result)

    def build_simulation(self):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(QtWidgets.QLabel("终端 A → 中继队列 → 终端 B  /  通用事件演示，未接入星地协议模型"))
        form = QtWidgets.QHBoxLayout()
        self.message_count = QtWidgets.QSpinBox()
        self.message_count.setRange(1, 1000)
        self.message_count.setValue(12)
        self.sim_fields = {}
        form.addWidget(QtWidgets.QLabel("消息数"))
        form.addWidget(self.message_count)
        for key, label, value in (("interval_s", "生成间隔", .1), ("transit_s", "单段传递时间", .02),
                                  ("service_s", "处理时间", .15), ("duration_s", "仿真时长", 3.0)):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(.001, 3600)
            spin.setDecimals(3)
            spin.setValue(value)
            spin.setSuffix(" s")
            form.addWidget(QtWidgets.QLabel(label))
            form.addWidget(spin)
            self.sim_fields[key] = spin
        self.sim_button = QtWidgets.QPushButton("运行实验")
        self.sim_button.setObjectName("primary")
        self.sim_button.clicked.connect(self.run_simulation)
        form.addWidget(self.sim_button)
        layout.addLayout(form)
        self.timeline = pg.PlotWidget(title="消息事件时间轴")
        self.timeline.setLabel("bottom", "模型时间", units="s")
        self.timeline.getAxis("left").setTicks([[(0, "终端 A"), (1, "中继"), (2, "终端 B")]])
        self.timeline.setYRange(-.5, 2.5)
        layout.addWidget(self.timeline, 1)
        self.replay = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.replay.setRange(0, 0)
        self.replay.valueChanged.connect(self.show_events)
        layout.addWidget(self.replay)
        self.sim_summary = QtWidgets.QLabel("运行后可拖动时间轴回放；尚未完成的消息单独计数。")
        layout.addWidget(self.sim_summary)
        self.event_table = QtWidgets.QTableWidget(0, 4)
        self.event_table.setHorizontalHeaderLabels(["时间 / s", "消息 ID", "节点", "阶段"])
        self.event_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.event_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.event_table, 1)
        export = QtWidgets.QPushButton("导出当前实验报告…")
        export.clicked.connect(self.export_current)
        layout.addWidget(export)
        return box


    def run_simulation(self):
        scenario = {key: spin.value() for key, spin in self.sim_fields.items()}
        scenario["messages"] = self.message_count.value()
        self.start_job("simulate", scenario=scenario)


    def show_events(self, count):
        events = self.events[:count]
        self.timeline.clear()
        positions = {"A": 0, "relay": 1, "B": 2}
        for node, y in positions.items():
            times = [event["time_s"] for event in events if event["node"] == node]
            self.timeline.plot(times, [y] * len(times), pen=None, symbol="o", symbolSize=7,
                               symbolBrush={"A": "#2365b3", "relay": "#e39b35", "B": "#29947d"}[node])
        visible = events[-200:]
        self.event_table.setRowCount(len(visible))
        for row, event in enumerate(visible):
            for column, value in enumerate((f"{event['time_s']:.6f}", event["message_id"], event["node"], event["stage"])):
                self.event_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))


    def display_result(self, result):
        self.last_result = result
        if result["kind"] == "simulation":
            self.tab_results[1] = result
            self.events = result["events"]
            self.replay.setRange(0, len(self.events))
            self.replay.setValue(len(self.events))
            self.show_events(len(self.events))
            s = result["summary"]
            self.sim_summary.setText(f"计划 {s['planned']}  |  已发送 {s['sent']}  |  已接收 {s['received']}  |  "
                                     f"在途/排队 {s['pending']}  |  未启动 {s['not_started']}  |  "
                                     f"模型时间 {s['simulated_duration_s']:g} s / 实际耗时 {s['wall_duration_s']:.4f} s")
            self.tabs.setCurrentIndex(0)

def launch(workspace):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(workspace)
    window.show()
    return app.exec()
