"""跳频脊线追踪：贪心追踪 vs 带状 DP。

模拟场景特意做成「非平凡」的，否则对比毫无意义：

* 若脊线在每一帧都是全列最大值，贪心永不失误，两种算法的偏差都是 0；
* 这里在正弦脊线上叠加两段 **深度衰落**（脊线能量掉到与噪声同量级）与一条
  **恒定 +4 个频点的邻道干扰**（能量恒强于衰落后的脊线、但弱于满幅脊线）。
  衰落期间贪心会跳到干扰上；衰落结束后干扰仍跟着脊线平行运动，贪心的
  ``±max_jump`` 搜索窗以**自己上一步（已错）的位置**为中心，于是真实脊线被
  甩在窗外，贪心就一直被干扰拖到扫频减速的波谷附近才回来。
  带状 DP 因为带平滑代价，不会为这点短期收益绕道，所以偏差维持在 ~0。
"""
import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

# ========== 1. 生成模拟时频矩阵 ==========
rng = np.random.default_rng(42)
n_freq, n_time = 64, 140            # 140 帧：正弦轨迹跑完 3.5 个周期

noise_amp = 0.30                    # 背景噪声上限（均匀分布），均值约 0.15
ridge_amp = 2.0                     # 脊线满幅能量
leak_amp = 0.25                     # 相邻频点泄漏系数（真实信号不会只占一个 bin）
fade_windows = ((48, 8), (88, 8))   # 两段深度衰落：(起始帧, 持续帧数)
fade_gain = 0.20                    # 衰落期间的幅度增益
lure_offset = 4                     # 邻道干扰相对真实脊线的频点偏移（< max_jump）
lure_amp = 0.70                     # 邻道干扰能量：强于衰落脊线，弱于满幅脊线

# 真实跳频脊线：正弦调制的频率轨迹
true_ridge = np.array([int(20 + 15 * np.sin(2 * np.pi * t / 40)) for t in range(n_time)])

tf = rng.random((n_freq, n_time)) * noise_amp   # 背景噪声

# 衰落增益曲线：非衰落帧为 1，衰落窗内降为 fade_gain
fade = np.ones(n_time)
for start, length in fade_windows:
    fade[start:start + length] = fade_gain

for t in range(n_time):
    tf[true_ridge[t], t] += ridge_amp * fade[t]
    # 加一点相邻频率泄漏，模拟真实信号
    for delta in (-1, 1):
        f = true_ridge[t] + delta
        if 0 <= f < n_freq:
            tf[f, t] += leak_amp * ridge_amp * fade[t]

# 邻道干扰：与脊线平行、偏移 lure_offset 的镜像分量，全程存在
for t in range(n_time):
    f = true_ridge[t] + lure_offset
    if 0 <= f < n_freq:
        tf[f, t] += lure_amp


# ========== 2. 贪心追踪 ==========
def greedy_tracking(tf_matrix, start_time, max_jump):
    n_freq, n_time = tf_matrix.shape
    ridge = np.zeros(n_time, dtype=int)
    current_freq = np.argmax(tf_matrix[:, start_time])
    ridge[start_time] = current_freq

    for t in range(start_time + 1, n_time):
        low = max(0, current_freq - max_jump)
        high = min(n_freq - 1, current_freq + max_jump)
        candidates = tf_matrix[low:high + 1, t]
        best_freq = low + np.argmax(candidates)
        ridge[t] = best_freq
        current_freq = best_freq

    return ridge


# ========== 3. 带状 DP ==========
def banded_dp_tracking(tf_matrix, ref_ridge, W=16, lam=0.5):
    n_freq, n_time = tf_matrix.shape
    INF = -1e18

    dp = {}
    parent = {}

    t = 0
    center = ref_ridge[t]
    low = max(0, center - W)
    high = min(n_freq - 1, center + W)

    for f in range(low, high + 1):
        dp[(t, f)] = tf_matrix[f, t]
        parent[(t, f)] = None

    for t in range(1, n_time):
        center = ref_ridge[t]
        low = max(0, center - W)
        high = min(n_freq - 1, center + W)

        prev_center = ref_ridge[t - 1]
        prev_low = max(0, prev_center - W)
        prev_high = min(n_freq - 1, prev_center + W)

        for f in range(low, high + 1):
            best_score = INF
            best_prev = None

            for f_prev in range(prev_low, prev_high + 1):
                if (t - 1, f_prev) not in dp:
                    continue
                score = dp[(t - 1, f_prev)] - lam * abs(f - f_prev)
                if score > best_score:
                    best_score = score
                    best_prev = f_prev

            if best_prev is not None:
                dp[(t, f)] = tf_matrix[f, t] + best_score
                parent[(t, f)] = best_prev

    t_last = n_time - 1
    center = ref_ridge[t_last]
    low = max(0, center - W)
    high = min(n_freq - 1, center + W)

    best_f = max(range(low, high + 1), key=lambda f: dp.get((t_last, f), INF))

    path = np.zeros(n_time, dtype=int)
    f = best_f
    for t in range(t_last, -1, -1):
        path[t] = f
        f = parent[(t, f)]
        if f is None:
            break

    return path


# ========== 4. 运行算法 ==========
max_jump = 5        # 贪心每帧允许的最大频点跳变
band_width = 10     # 带状 DP 相对参考脊线的搜索带宽
penalty = 0.5       # 频点跳变的平滑代价系数 λ

ref = greedy_tracking(tf, start_time=0, max_jump=max_jump)
optimal = banded_dp_tracking(tf, ref, W=band_width, lam=penalty)

error_ref = np.abs(ref - true_ridge)
error_dp = np.abs(optimal - true_ridge)
print("贪心与真值平均偏差：", np.mean(error_ref))
print("带状DP与真值平均偏差：", np.mean(error_dp))
print("偏差非零的帧数：贪心 {} / {}，带状DP {} / {}".format(
    int(np.count_nonzero(error_ref)), n_time,
    int(np.count_nonzero(error_dp)), n_time))


# ========== 5. 绘图（PySide6 + pyqtgraph） ==========
def build_window():
    """构建「时频图 + 轨迹图 + 文本摘要」三行窗口，返回 ``GraphicsLayoutWidget``。"""
    # 与 common/gui.py 的 DesktopWindow 保持同一套主题
    pg.setConfigOptions(background="#ffffff", foreground="#34465a", antialias=True)

    widget = pg.GraphicsLayoutWidget(title="跳频脊线对比（贪心 vs 带状DP）")
    widget.resize(1300, 1000)
    frames = np.arange(n_time)
    viridis = pg.colormap.get("viridis")
    lure_ridge = true_ridge + lure_offset      # 邻道干扰的频点轨迹

    def fade_regions():
        """深度衰落窗的阴影：沿用 gui.py 里 LinearRegionItem 的写法。"""
        return [pg.LinearRegionItem(values=(start, start + length), movable=False,
                                    brush=pg.mkBrush(226, 86, 74, 45),
                                    pen=pg.mkPen("#e2564a"))
                for start, length in fade_windows]

    # --- 子图 1：时频图 + 四条脊线 ---
    ax1 = widget.addPlot(row=0, col=0, title="时频图上的脊线对比")
    ax1.setLabel("bottom", "时间帧")
    ax1.setLabel("left", "频率索引")
    ax1.addLegend()

    image = pg.ImageItem(axisOrder="row-major")
    # 注意：row-major 下数组第 0 行本来就落在纵轴下端（等价于 matplotlib 的
    # origin='lower'），这里**不能再 flipud**，否则时频图上下镜像、看上去和
    # 脊线曲线对不上。
    image.setImage(tf, autoLevels=True)
    image.setRect(QtCore.QRectF(0, 0, n_time, n_freq))
    image.setLookupTable(viridis.getLookupTable())
    ax1.addItem(image)
    for item in fade_regions():                # 标出贪心失手的诱因
        ax1.addItem(item)

    # 色条（pyqtgraph >= 0.13，已由 pyproject 钉住）
    bar = pg.ColorBarItem(values=(float(tf.min()), float(tf.max())),
                          colorMap=viridis, label="能量", width=14)
    bar.setImageItem([image], insert_in=ax1)

    ax1.plot(frames, lure_ridge, pen=pg.mkPen("#7d8fa1", width=1.5,
                                              style=QtCore.Qt.PenStyle.DotLine),
             name="邻道干扰")
    ax1.plot(frames, true_ridge,
             pen=pg.mkPen("w", width=2.5, style=QtCore.Qt.PenStyle.DashLine), name="真实脊线")
    ax1.plot(frames, ref, pen=pg.mkPen("r", width=2), name="贪心追踪")
    ax1.plot(frames, optimal, pen=pg.mkPen("c", width=2), name="带状DP")

    # --- 子图 2：三条曲线直接对比 ---
    ax2 = widget.addPlot(row=1, col=0, title="脊线轨迹对比（贪心 vs 带状DP vs 真值）")
    ax2.setLabel("bottom", "时间帧")
    ax2.setLabel("left", "频率索引")
    ax2.addLegend()
    ax2.showGrid(x=True, y=True, alpha=0.3)
    for item in fade_regions():
        ax2.addItem(item)

    ax2.plot(frames, lure_ridge, pen=pg.mkPen("#7d8fa1", width=1.5,
                                              style=QtCore.Qt.PenStyle.DotLine),
             name="邻道干扰")
    ax2.plot(frames, true_ridge,
             pen=pg.mkPen("k", width=2.5, style=QtCore.Qt.PenStyle.DashLine), name="真实脊线")
    ax2.plot(frames, ref, pen=pg.mkPen("r", width=1.8), name="贪心追踪")
    ax2.plot(frames, optimal, pen=pg.mkPen("c", width=1.8), name="带状DP")

    # 真值 ±1 容差带：pyqtgraph 用 FillBetweenItem 代替 matplotlib 的 fill_between
    upper = ax2.plot(frames, true_ridge + 1, pen=None)
    lower = ax2.plot(frames, true_ridge - 1, pen=None)
    ax2.addItem(pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(128, 128, 128, 60)))

    # 文本摘要单独占一行：TextItem 会参与自动量程，塞进曲线图会把纵轴拉到 64、
    # 曲线被压扁，所以这里用 addLabel。
    widget.addLabel(
        "贪心平均偏差 <b>{:.3f}</b>（{} 帧偏离）　|　"
        "带状DP平均偏差 <b>{:.3f}</b>（{} 帧偏离）　|　红色阴影 = 深度衰落窗".format(
            float(np.mean(error_ref)), int(np.count_nonzero(error_ref)),
            float(np.mean(error_dp)), int(np.count_nonzero(error_dp))),
        row=2, col=0, justify="left", color="#183044", size="11pt")

    return widget


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = build_window()
    widget.show()
    # 先让事件循环跑一轮，否则布局未生效，grab() 可能截到空白
    app.processEvents()
    # 保存 PNG：QWidget.grab() 纯 Qt 实现，无需 matplotlib 的 savefig
    widget.grab().save("ridge_comparison.png")
    print("已保存 ridge_comparison.png")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
