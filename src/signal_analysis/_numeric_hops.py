"""跳频逐跳参数估计：时频脊线跟踪、细网格量测和发射机会话归并。

仅依赖标准库、NumPy 与其他底层数值模块，可随数值核心进行 Cython 编译。
"""

import numpy as np

from ._numeric_common import (
    DETECT_SNR_DEFINITION,
    _SESSION_GAP_RATIO,
    _SNR_FLOOR_DB,
    _binary_close,
    _finite,
    _noise_floor_db,
    _occupied_span,
    _stft_psd,
    _true_runs,
    validate_rate,
    validate_samples,
)

from ._numeric_energy import (
    detect_signals,
)


HOP_ALGORITHM = "hop_track_v1"


HOP_CONTRACT = "fh_hops_v1"


#: 一跳至少占这么多帧，否则驻留过短、无法从时频面上可靠分辨
_HOP_MIN_DWELL_FRAMES = 4


_HOP_DEFAULTS = {
    "nfft": 512,
    # 单帧周期图只有约 2 个自由度，逐点起伏约 5.6 dB，远大于时间平均后的
    # 0.5 dB。故先按 smooth_frames 帧滑动平均把起伏压到约 2.8 dB，再取
    # 6 dB 门限，使单点虚警概率降到 1e-6 量级（否则 512 点 × 数百帧上会
    # 长出大量噪声游程）。逐跳的频域闭运算半径也相应取小（见下）。
    "threshold_db": 6.0,
    "smooth_frames": 4,
    "min_bandwidth_hz": 0.0,
    "min_dwell_s": 0.0,
    "merge_bins": 0,
    "max_gap_frames": 1,
    "max_gap_bins": 0,
    "transition_ratio": 1.5,
    "max_hops": 256,
}


def _hop_config(rate, duration, config):
    """校验并展开逐跳跟踪配置。

    参数：rate 为 Hz，duration 为秒，config 为字典或 None。

    返回：有效配置字典，包括频率分辨率、最小驻留秒数与帧数。

    算法与边界：拒绝未知键；默认 512 点、6 dB 门限、4 帧平滑，最小带宽 3 点，自动闭运算半径 max(1,nfft//512)。最小驻留秒数至少
    4·nfft/rate，帧数按 ceil(min_dwell·rate/nfft) 计算且至少为 4；保留此既有换算。非法配置抛出 ValueError。
    """
    if config is None:
        settings = {}
    elif isinstance(config, dict):
        settings = dict(config)
    else:
        raise ValueError("逐跳配置必须是字典")
    unknown = set(settings) - set(_HOP_DEFAULTS)
    if unknown:
        raise ValueError(f"未知的逐跳参数：{', '.join(sorted(unknown))}")
    nfft = settings.get("nfft", _HOP_DEFAULTS["nfft"])
    if isinstance(nfft, bool) or int(nfft) != nfft or not 16 <= int(nfft) <= 4096:
        raise ValueError("FFT 点数必须为 16～4096 的整数")
    nfft = int(nfft)
    resolution = rate / nfft
    threshold_db = _finite(settings.get("threshold_db", _HOP_DEFAULTS["threshold_db"]),
                           "检测门限", 0.0, 80.0)
    smooth_frames = settings.get("smooth_frames", _HOP_DEFAULTS["smooth_frames"])
    if isinstance(smooth_frames, bool) or int(smooth_frames) != smooth_frames \
            or not 1 <= int(smooth_frames) <= 64:
        raise ValueError("时间平滑帧数必须为 1～64 的整数")
    min_bandwidth = _finite(settings.get("min_bandwidth_hz", 0.0), "最小带宽", 0.0, rate)
    if min_bandwidth <= 0:
        # 同一默认：至少 3 个频点宽，抑制单点毛刺
        min_bandwidth = 3.0 * resolution
    min_dwell = _finite(settings.get("min_dwell_s", 0.0), "最短驻留时间", 0.0, duration)
    min_dwell = max(min_dwell, _HOP_MIN_DWELL_FRAMES * nfft / rate)
    merge_bins = settings.get("merge_bins", 0)
    if isinstance(merge_bins, bool) or int(merge_bins) != merge_bins \
            or not 0 <= int(merge_bins) <= nfft // 4:
        raise ValueError(f"谱合并宽度必须为 0～{nfft // 4} 的整数")
    # 逐跳的闭运算半径取 1 个频点（桥接 2 点）：单帧上信道内部已经平坦，
    # 半径取大只会把相邻信道在跳变帧里焊成一条，掩盖真正的跳变。
    merge_bins = int(merge_bins) or max(1, nfft // 512)
    max_gap_frames = settings.get("max_gap_frames", _HOP_DEFAULTS["max_gap_frames"])
    if isinstance(max_gap_frames, bool) or int(max_gap_frames) != max_gap_frames \
            or not 0 <= int(max_gap_frames) <= 16:
        raise ValueError("允许的跟踪间断帧数必须为 0～16 的整数")
    max_gap_bins = settings.get("max_gap_bins", 0)
    if isinstance(max_gap_bins, bool) or int(max_gap_bins) != max_gap_bins \
            or not 0 <= int(max_gap_bins) <= nfft // 4:
        raise ValueError(f"频点跳变门限必须为 0～{nfft // 4} 的整数")
    max_gap_bins = int(max_gap_bins) or merge_bins
    transition_ratio = _finite(settings.get("transition_ratio", _HOP_DEFAULTS["transition_ratio"]),
                               "过渡帧宽度比", 1.1, 4.0)
    max_hops = settings.get("max_hops", _HOP_DEFAULTS["max_hops"])
    if isinstance(max_hops, bool) or int(max_hops) != max_hops or not 1 <= int(max_hops) <= 256:
        raise ValueError("最大跳数必须为 1～256 的整数")
    return {
        "nfft": nfft,
        "threshold_db": threshold_db,
        "smooth_frames": int(smooth_frames),
        "min_bandwidth_hz": min_bandwidth,
        "min_dwell_s": min_dwell,
        "merge_bins": merge_bins,
        "max_gap_frames": int(max_gap_frames),
        "max_gap_bins": max_gap_bins,
        "transition_ratio": transition_ratio,
        "max_hops": int(max_hops),
        "freq_resolution_hz": resolution,
        "min_bandwidth_bins": max(1, int(np.ceil(min_bandwidth / resolution))),
        "min_dwell_frames": max(_HOP_MIN_DWELL_FRAMES, int(np.ceil(min_dwell * rate / nfft))),
    }


def _smooth_psd(psd, window):
    """沿帧维滑动平均线性 PSD，保持时间网格。

    参数：psd 为（帧数,频点数）线性 PSD；window 为平滑窗口帧数。

    返回：与 psd 同形状的平均数组；window<=1 时返回原对象。

    算法与边界：两端复制边界帧，左侧补 window//2 帧、右侧补 window-1-window//2
    帧，再沿滑窗新增的末维求均值。平滑降低单帧噪声起伏，驻留量测仍使用未平滑 PSD。
    """
    if window <= 1:
        return psd
    half = window // 2
    padded = np.pad(psd, ((half, window - 1 - half), (0, 0)), mode="edge")
    # ``sliding_window_view`` 把窗口维度追加在**末尾**（不是挂在 axis 位置上），
    # 所以这里按最后一维求均值。
    frames = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    return frames.mean(axis=-1)


def _frame_runs(psd_db, threshold_dbfs, merge_bins, min_bins):
    """逐帧提取门限以上的频谱游程。

    参数：psd_db 为（帧数,频点数）dB PSD；threshold_dbfs 为同口径绝对谱密度门限；merge_bins 为闭运算半径，min_bins
    为最小游程宽度。

    返回：每帧一个游程列表，元素为包含两端的 (first_bin,last_bin)。

    算法与边界：逐帧比较门限，频域闭运算填凹口后去除过窄游程；保留空帧列表以维持原时间索引。
    """
    rows = []
    for index in range(psd_db.shape[0]):
        mask = _binary_close(psd_db[index] > threshold_dbfs, merge_bins)
        rows.append([(start, stop) for start, stop in _true_runs(mask)
                     if stop - start + 1 >= min_bins])
    return rows


def _track_band(track):
    """取轨迹中已接受游程的频带并集。

    参数：track 为轨迹字典，runs 为非空的包含两端的频点索引列表。

    返回：(first_bin,last_bin)，均包含端点。

    算法与边界：取游程起点最小值和终点最大值；过渡帧只有被跟踪器接受为几何证据时才进入 runs。
    """
    runs = track["runs"]
    return min(start for start, _ in runs), max(stop for _, stop in runs)


def _best_track(tracks, frame, first, last, max_gap_frames, max_gap_bins, excluded,
                allow_gap=True):
    """为当前游程选择最匹配的有效轨迹。

    参数：tracks 为轨迹列表；frame 为当前帧号；first、last 为频点边界；max_gap_frames、max_gap_bins
    为容许间隔；excluded 为本帧已占用轨迹编号；allow_gap 控制能否跨频隙。

    返回：轨迹字典原对象；无匹配时返回 None。

    算法与边界：过滤已占用及过期轨迹；优先频域重叠最大者，其次选择容许范围内最小频隙，最后按轨迹编号排序稳定决胜。多游程帧禁用频隙接续，避免焊接相邻信道。
    """
    ranked = []
    for track in tracks:
        if track["id"] in excluded:
            continue
        if frame - track["last_frame"] - 1 > max_gap_frames:
            continue
        low, high = _track_band(track)
        overlap = min(last, high) - max(first, low) + 1
        if overlap > 0:
            ranked.append((0, -overlap, track["id"], track))
            continue
        if not allow_gap or track["last_frame"] == frame:
            # 同帧内已经有一只游程续上了这条轨道，再靠频隙衔接会把
            # 相邻信道（跳变帧）焊在一起，因此只允许严格重叠。
            continue
        distance = max(low - last, first - high)
        if distance <= max_gap_bins:
            ranked.append((1, distance, track["id"], track))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranked[0][3]


def _overlaps_any(tracks, frame, first, last, max_gap_frames):
    """检查当前游程是否已被仍有效的轨迹覆盖。

    参数：tracks 为轨迹列表；frame 为帧号；first、last 为频点边界；max_gap_frames 为容许缺帧数。

    返回：布尔值。

    算法与边界：仅考察时间上仍可跟踪的轨迹，存在频带重叠即返回 True；过期轨迹不阻止同一频点后续访问新建轨迹，避免将频谱凹口误当新跳。
    """
    for track in tracks:
        if frame - track["last_frame"] - 1 > max_gap_frames:
            continue
        low, high = _track_band(track)
        if min(last, high) - max(first, low) + 1 > 0:
            return True
    return False


def _link_tracks(rows, frequencies, resolved, transition_frames):
    """沿时间把逐帧游程连接为跳频轨迹。

    参数：rows 为逐帧游程；frequencies 为 Hz 频率轴；resolved 为有效配置；transition_frames 为单元素可变计数列表。

    返回：轨迹字典列表，含帧范围、帧索引、几何游程和过渡帧计数；同步更新 transition_frames[0]。

    算法与边界：单游程允许小频隙，多游程只按重叠接续，未匹配游程新建轨迹。多游程帧或超过典型宽度 transition_ratio
    倍的宽游程只延续时间、不扩展已有轨迹几何，避免跳变泄漏把两跳频带合并。
    """
    ratio = resolved["transition_ratio"]
    max_gap_frames = resolved["max_gap_frames"]
    max_gap_bins = resolved["max_gap_bins"]
    tracks = []
    counter = 0
    for frame, runs in enumerate(rows):
        if not runs:
            continue
        if len(runs) > 1:
            transition_frames[0] += 1
        claimed = set()
        for first, last in runs:
            track = _best_track(tracks, frame, first, last, max_gap_frames,
                                max_gap_bins, claimed, allow_gap=len(runs) == 1)
            if track is None and len(runs) > 1 and _overlaps_any(
                    tracks, frame, first, last, max_gap_frames):
                continue
            if track is None:
                counter += 1
                tracks.append({"id": counter, "first_frame": frame, "last_frame": frame,
                               "frames": [frame], "runs": [(first, last)],
                               "transition_frames": 0})
                claimed.add(counter)
                continue
            claimed.add(track["id"])
            width = last - first + 1
            widths = sorted(stop - start + 1 for start, stop in track["runs"])
            median = widths[len(widths) // 2]
            track["last_frame"] = frame
            track["frames"].append(frame)
            if len(runs) > 1 or (len(track["runs"]) >= 2 and width > ratio * median):
                track["transition_frames"] += 1
                continue
            track["runs"].append((first, last))
    return tracks


def _finalise_hops(tracks, psd, frequencies, starts, nfft, rate, duration,
                   noise_linear, resolved):
    """将轨迹转换为单跳频带、驻留、功率和 SNR 量测。

    参数：tracks 为轨迹；psd 为原始线性 PSD；frequencies 为 Hz；starts 为帧起点采样索引；nfft、rate、duration 为
    FFT 点数、Hz、秒；noise_linear 为每 Hz 噪声功率，resolved 为配置。

    返回：单跳量测字典列表；同时供传统跟踪与 AI 候选框复用。

    算法与边界：几何游程给搜索带，原始 PSD 带内功率越门限给时间支撑，选与轨迹重叠最多的一段以区分重复访频。丢弃驻留不足项；扣噪后按 99% 功率占用带测宽，SNR
    使用扣噪信号功率/(N0·B_hop)，下限 -20 dB。时间裁到记录边界。
    """
    resolution = resolved["freq_resolution_hz"]
    threshold_db = resolved["threshold_db"]
    min_dwell_frames = resolved["min_dwell_frames"]
    hops = []
    for track in tracks:
        first_bin, last_bin = _track_band(track)
        bandwidth = (last_bin - first_bin + 1) * resolution
        f_low = float(frequencies[first_bin] - resolution / 2.0)
        f_high = float(frequencies[last_bin] + resolution / 2.0)
        band_power = np.asarray(psd[:, first_bin:last_bin + 1].sum(axis=1),
                                dtype=np.float64) * resolution
        active = band_power > noise_linear * bandwidth * 10.0 ** (threshold_db / 10.0)
        if not active.any():
            continue
        best, best_overlap = None, 0
        for start, stop in _true_runs(active):
            overlap = min(stop, track["last_frame"]) - max(start, track["first_frame"]) + 1
            if overlap > best_overlap:
                best, best_overlap = (start, stop), overlap
        if best is None:
            best = (track["first_frame"], track["last_frame"])
        start_frame, stop_frame = best
        if stop_frame - start_frame + 1 < min_dwell_frames:
            continue
        t_start = float(starts[start_frame]) / rate
        t_end = float(min((starts[stop_frame] + nfft) / rate, duration))
        dwell = t_end - t_start
        if dwell < resolved["min_dwell_s"]:
            continue
        segment = band_power[start_frame:stop_frame + 1]
        power_linear = float(segment.mean())
        band_frequencies = frequencies[first_bin:last_bin + 1]
        profile = psd[start_frame:stop_frame + 1, first_bin:last_bin + 1].mean(axis=0)
        weight_sum = float(profile.sum())
        centroid = (float((profile * band_frequencies).sum() / weight_sum)
                    if weight_sum > 0 else float(band_frequencies.mean()))
        # 单跳带宽取「扣噪声后的 99% 功率占用带宽」：检测掩膜会被 OFDM/2FSK 的
        # 频谱旁瓣撑宽数个频点（实测约 1.3～2.4 个频点/侧），只有功率占用带宽才代表
        # 信号真正占用的频带。
        corrected = np.maximum(profile.astype(np.float64) - noise_linear, 0.0)
        occupancy_low, occupancy_high = _occupied_span(corrected, 0, corrected.size - 1)
        occupied_low = float(frequencies[first_bin + occupancy_low] - resolution / 2.0)
        occupied_high = float(frequencies[first_bin + occupancy_high] + resolution / 2.0)
        occupied_bandwidth = occupied_high - occupied_low
        # inband_snr_v1：信号功率按检测掩膜频带内的实测功率扣掉该频带噪声，
        # 噪声参考功率按信号自身占用带宽计算（N0·B_hop）。
        signal_power = max(power_linear - noise_linear * bandwidth, 1e-30)
        noise_band = max(noise_linear * occupied_bandwidth, 1e-30)
        snr_db = float(max(10.0 * np.log10(signal_power / noise_band), _SNR_FLOOR_DB))
        hops.append({
            "center_hz": 0.5 * (occupied_low + occupied_high),
            "centroid_hz": centroid,
            "bandwidth_hz": occupied_bandwidth,
            "f_low_hz": occupied_low,
            "f_high_hz": occupied_high,
            "mask_bandwidth_hz": bandwidth,
            "mask_f_low_hz": f_low,
            "mask_f_high_hz": f_high,
            "t_start_s": t_start,
            "t_end_s": t_end,
            "dwell_s": dwell,
            "power_linear": power_linear,
            "snr_db": snr_db,
            "confidence": float(np.clip((snr_db + 5.0) / 25.0, 0.0, 1.0)),
            "frame_count": int(stop_frame - start_frame + 1),
            "bin_count": int(sum(stop - start + 1 for start, stop in track["runs"])),
            "band_nfft": resolved["nfft"],
        })
    return hops


#: 频带精测的分析点数上限。跟踪需要时间分辨率、带宽需要频率分辨率，一个网格
#: 满足不了两者，所以跟踪用粗网格、频带测量再在细网格上复算一遍。
_HOP_REFINE_NFFT = 2048


def _refine_hop_bands(hops, x, rate, noise_linear, resolved, threshold_db, coarse_nfft):
    """在更细 FFT 网格上原位复测各跳频带。

    参数：hops 为可变单跳列表；x 为 IQ；rate 为 Hz；noise_linear 为每 Hz 噪声功率；resolved 为配置；threshold_db 为
    dB；coarse_nfft 为粗网格点数。

    返回：返回 None；更新 hops 内频带、功率、质心、SNR、置信度和 band_nfft 字段。

    算法与边界：细网格上限 2048 点，按最短驻留限制点数；搜索范围取粗掩膜加两格保护带，噪声底沿用粗网格。重算扣噪 99%
    占用带并保持粗网格时间不变；空输入、细网格无增益或无有效频段时跳过。
    """
    if not hops:
        return
    shortest = min(item["dwell_s"] for item in hops) * rate
    limit = int(2.0 ** np.floor(np.log2(max(shortest / _HOP_MIN_DWELL_FRAMES, 16.0))))
    refine_nfft = int(min(_HOP_REFINE_NFFT, limit))
    if refine_nfft <= coarse_nfft:
        return
    psd, frequencies, starts, _ = _stft_psd(x, rate, refine_nfft)
    resolution = rate / refine_nfft
    centers = (starts + (refine_nfft - 1) / 2.0) / rate
    threshold_linear = 10.0 ** (threshold_db / 10.0)
    guard = 2.0 * resolved["freq_resolution_hz"]
    for item in hops:
        low_bin = int(np.searchsorted(frequencies, item["mask_f_low_hz"] - guard, side="left"))
        high_bin = int(np.searchsorted(frequencies, item["mask_f_high_hz"] + guard, side="right"))
        low_bin = max(0, min(low_bin, frequencies.size - 2))
        high_bin = min(frequencies.size - 1, max(high_bin, low_bin + 1))
        inside = np.flatnonzero((centers >= item["t_start_s"]) & (centers <= item["t_end_s"]))
        if inside.size == 0:
            continue
        profile = psd[inside, low_bin:high_bin + 1].mean(axis=0).astype(np.float64)
        band = frequencies[low_bin:high_bin + 1]
        mask = profile > noise_linear * threshold_linear
        if not mask.any():
            continue
        occupied_low, occupied_high = _occupied_span(
            np.maximum(profile - noise_linear, 0.0), 0, profile.size - 1)
        f_low = float(band[occupied_low]) - resolution / 2.0
        f_high = float(band[occupied_high]) + resolution / 2.0
        mask_bins = np.flatnonzero(mask)
        mask_low = float(band[int(mask_bins[0])]) - resolution / 2.0
        mask_high = float(band[int(mask_bins[-1])]) + resolution / 2.0
        power_linear = float(profile[mask].sum() * resolution)
        weight_sum = float(profile.sum())
        centroid = (float((profile * band).sum() / weight_sum)
                    if weight_sum > 0 else float(band.mean()))
        signal_power = max(power_linear - noise_linear * (mask_high - mask_low), 1e-30)
        noise_band = max(noise_linear * (f_high - f_low), 1e-30)
        snr_db = float(max(10.0 * np.log10(signal_power / noise_band), _SNR_FLOOR_DB))
        item.update({
            "f_low_hz": f_low,
            "f_high_hz": f_high,
            "bandwidth_hz": f_high - f_low,
            "center_hz": 0.5 * (f_low + f_high),
            "centroid_hz": centroid,
            "mask_f_low_hz": mask_low,
            "mask_f_high_hz": mask_high,
            "mask_bandwidth_hz": mask_high - mask_low,
            "power_linear": power_linear,
            "snr_db": snr_db,
            "confidence": float(np.clip((snr_db + 5.0) / 25.0, 0.0, 1.0)),
            "band_nfft": refine_nfft,
        })


#: 同一会话内频点间隔的最大/最小比：超过它说明这个时间组里混进了另一部
#: 发射机（两部发射机的频带之间会留下一道明显更宽的缝）。
_HOP_SESSION_SPLIT_RATIO = 3.0


_HOP_SESSION_OVERLAP_TOL_FRAMES = 1.5


def _cluster_centres(values, resolution):
    """将重复访问测得的频点聚合为信道集合。

    参数：values 为中心频率可迭代序列，resolution 为分辨率，单位均为 Hz。

    返回：升序 float 信道中心列表。

    算法与边界：排序后把与当前簇最后一点相距不超过一格的值连接成簇，每簇取均值；采用相邻点链式归并，空输入返回空列表。
    """
    clusters = []
    for value in sorted(values):
        if not clusters or value - clusters[-1][-1] > resolution:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [float(np.mean(group)) for group in clusters]


def _split_session(members, resolution):
    """按异常信道间隔拆分混入多个发射机的时间组。

    参数：members 为单跳字典列表；resolution 为频率分辨率 Hz。

    返回：由单跳列表组成的分组列表，复用原单跳对象。

    算法与边界：聚合信道后，当最大间隔超过 max(间隔中位数,resolution) 的 3 倍时从该间隔递归拆分；少于 3
    个独立信道或没有异常间隔时保留整组。使用中位数避免误拆不规则单发射机信道表。
    """
    if len(members) < 2:
        return [members]
    channels = _cluster_centres((item["center_hz"] for item in members), resolution)
    if len(channels) < 3:
        return [members]
    gaps = [(channels[index + 1] - channels[index], index)
            for index in range(len(channels) - 1)]
    widest, index = max(gaps)
    values = sorted(gap for gap, _ in gaps)
    median = values[len(values) // 2]
    if widest <= _HOP_SESSION_SPLIT_RATIO * max(median, resolution):
        return [members]
    limit = 0.5 * (channels[index] + channels[index + 1])
    lower = [item for item in members if item["center_hz"] <= limit]
    upper = [item for item in members if item["center_hz"] > limit]
    return _split_session(lower, resolution) + _split_session(upper, resolution)


def _group_hop_sessions(hops, noise_linear, duration, resolution, frame_interval_s):
    """按时间连续性与信道分布归并发射机会话。

    参数：hops 为单跳量测列表；noise_linear 为每 Hz 噪声功率；duration 为记录秒数；resolution 为
    Hz；frame_interval_s 为帧间隔秒数。

    返回：会话字典列表，并原位写入每个单跳的 session_id。

    算法与边界：按起点和中心排序，以时间间隔作可用性门限、频距最近者优先接续，再按异常信道间隔拆组。计算访问信道、驻留统计和起点间隔中位数倒数作为跳速；单跳会话跳速为
    None。功率按驻留加权摊到整段记录后扣除会话频带噪声，不能用能量检测的整带归并替代。
    """
    overlap_tol = _HOP_SESSION_OVERLAP_TOL_FRAMES * frame_interval_s
    ordered = sorted(hops, key=lambda item: (item["t_start_s"], item["center_hz"]))
    groups = []
    for hop in ordered:
        chosen, chosen_score = None, None
        for index, session in enumerate(groups):
            pause = hop["t_start_s"] - session["t_end_s"]
            if pause < -overlap_tol:
                # 与已在进行的会话明显重叠：只能是另一部发射机（不足
                # 两帧的重叠是跳变帧本身的展宽，不算证据）
                continue
            if pause > _SESSION_GAP_RATIO * session["dwell_median_s"]:
                continue
            centres = [item["center_hz"] for item in session["hops"]]
            distance = max(min(centres) - hop["center_hz"],
                           hop["center_hz"] - max(centres), 0.0)
            # 先看频距再看时间：同一部发射机的跳频点落在一段固定的跨度里，
            # 因此「最近的会话」是稳定的判据；两部交错的发射机也由此分开，
            # 而时间只作为可用性门限（跳变帧会带来不足两帧的重叠）。
            score = (distance, pause)
            if chosen_score is None or score < chosen_score:
                chosen, chosen_score = index, score
        if chosen is None:
            groups.append({"t_end_s": hop["t_end_s"],
                           "dwell_median_s": hop["dwell_s"],
                           "hops": [hop]})
            continue
        session = groups[chosen]
        session["hops"].append(hop)
        session["t_end_s"] = max(session["t_end_s"], hop["t_end_s"])
        dwells = sorted(item["dwell_s"] for item in session["hops"])
        session["dwell_median_s"] = dwells[len(dwells) // 2]
    sessions = [{"hops": members}
                for group in groups
                for members in _split_session(group["hops"], resolution)]
    sessions.sort(key=lambda session: session["hops"][0]["t_start_s"])
    for identifier, session in enumerate(sessions, start=1):
        members = session["hops"]
        for item in members:
            item["session_id"] = identifier
        dwells = sorted(item["dwell_s"] for item in members)
        widths = sorted(item["bandwidth_hz"] for item in members)
        # 跳频点聚合：同一信道被多次访问时测得的中心会有不到一个频点的抖动，
        # 先按一个频点宽度聚类再去重，得到「访问过的跳频点集合」。
        channels = _cluster_centres((item["center_hz"] for item in members), resolution)
        spacings = [channels[index + 1] - channels[index]
                    for index in range(len(channels) - 1)]
        # 跳速取「跳起点间隔中位数」的倒数：它是发射机的跳频周期，与驻留时间
        # （信号真正存在的时间）不同——本工程 OFDM 图传样式每跳尾部有空闲，两者
        # 相差正好是一个占空比。
        starts_sorted = sorted(item["t_start_s"] for item in members)
        periods = [starts_sorted[index + 1] - starts_sorted[index]
                   for index in range(len(starts_sorted) - 1)]
        hop_period = sorted(periods)[len(periods) // 2] if periods else None
        dwell_median = dwells[len(dwells) // 2]
        f_low = min(item["f_low_hz"] for item in members)
        f_high = max(item["f_high_hz"] for item in members)
        # 会话级功率：按驻留时长加权摊到整段记录，与生成器的 power_dbfs_actual 同口径
        power_linear = sum(item["power_linear"] * item["dwell_s"]
                           for item in members) / max(duration, 1e-30)
        noise_band = max(noise_linear * (f_high - f_low), 1e-30)
        signal_power = max(power_linear - noise_band, 1e-30)
        session.update({
            "session_id": identifier,
            "hop_count": len(members),
            "sequence": [item["center_hz"] for item in members],
            "hop_frequencies_hz": channels,
            "channel_spacing_hz": (min(spacings) if spacings else None),
            "hop_span_hz": (channels[-1] - channels[0]) if len(channels) > 1 else 0.0,
            "hop_bandwidth_hz": widths[len(widths) // 2],
            "hop_period_s": hop_period,
            "hop_rate_hz": (1.0 / hop_period) if hop_period else None,
            "duty_cycle": ((dwell_median / hop_period) if hop_period else None),
            "dwell_median_s": dwell_median,
            "dwell_min_s": dwells[0],
            "dwell_max_s": dwells[-1],
            "center_hz": 0.5 * (f_low + f_high),
            "bandwidth_hz": f_high - f_low,
            "f_low_hz": f_low,
            "f_high_hz": f_high,
            "t_start_s": members[0]["t_start_s"],
            "t_end_s": max(item["t_end_s"] for item in members),
            "power_linear": power_linear,
            "snr_db": float(max(10.0 * np.log10(signal_power / noise_band), _SNR_FLOOR_DB)),
            "session_detection_id": None,
        })
    return sessions


def detect_hops(samples, sample_rate, config=None, with_sessions=True):
    """执行逐跳跟踪、参数估计和会话汇总。

    参数：samples 为一维 IQ；sample_rate 为 Hz；config 为逐跳配置或 None；with_sessions 默认
    True，控制是否附加会话级能量基线。

    返回：(summary, arrays)：fh_hops_v1 摘要及频谱、原始／平滑时频图、逐跳框和量测数组；所有频率均为基带偏移 Hz，时间为秒。

    算法与边界：STFT 保留帧维，本底从时间平均谱估计；平滑、逐帧门限游程、脊线连接后在原始功率上复测驻留，细 FFT 复测频带，再归并会话。超 max_hops
    先按功率截取，再按时间和频率稳定编号。无可辨跳时通过 resolvable/reason 说明；同频连续跳可能表现为一次长驻留。with_sessions=True
    时调用 detect_signals 并按最大频带重叠互链；不改变 detect_result_v1。见跳频逐跳参数估计设计。
    """
    x = validate_samples(samples)
    rate = validate_rate(sample_rate)
    duration = float(x.size / rate)
    resolved = _hop_config(rate, duration, config)
    nfft = resolved["nfft"]
    resolution = resolved["freq_resolution_hz"]
    threshold_db = resolved["threshold_db"]

    psd, frequencies, starts, hop = _stft_psd(x, rate, nfft)
    psd_average = psd.mean(axis=0)
    psd_average_db = 10.0 * np.log10(np.maximum(psd_average, 1e-30))
    noise_floor_db = _noise_floor_db(psd_average_db, threshold_db)
    threshold_dbfs = noise_floor_db + threshold_db
    noise_linear = 10.0 ** (noise_floor_db / 10.0)

    smoothed = _smooth_psd(psd, resolved["smooth_frames"])
    smoothed_db = 10.0 * np.log10(np.maximum(smoothed, 1e-30))
    rows = _frame_runs(smoothed_db, threshold_dbfs, resolved["merge_bins"],
                       resolved["min_bandwidth_bins"])
    transition_frames = [0]
    tracks = _link_tracks(rows, frequencies, resolved, transition_frames)
    hops = _finalise_hops(tracks, psd, frequencies, starts, nfft, rate, duration,
                          noise_linear, resolved)
    _refine_hop_bands(hops, x, rate, noise_linear, resolved, threshold_db, nfft)
    if len(hops) > resolved["max_hops"]:
        hops.sort(key=lambda item: -item["power_linear"])
        hops = hops[:resolved["max_hops"]]
    hops.sort(key=lambda item: (item["t_start_s"], item["center_hz"]))
    for index, item in enumerate(hops, start=1):
        item["id"] = index
        item["session_id"] = None
    sessions = _group_hop_sessions(hops, noise_linear, duration, resolution,
                                   hop / rate)

    dwell_limit_s = resolved["min_dwell_frames"] * hop / rate
    resolvable = bool(hops) and min(item["dwell_s"] for item in hops) >= dwell_limit_s
    if not hops:
        reason = (f"没有任何游程同时满足最小带宽 {resolved['min_bandwidth_hz']:.0f} Hz 与"
                  f"最短驻留 {resolved['min_dwell_frames']} 帧；请降低门限或增大分析点数")
    elif not resolvable:
        reason = (f"STFT 帧间距 {hop / rate * 1000.0:.3f} ms、一跳至少"
                  f"{resolved['min_dwell_frames']} 帧，跳速高于"
                  f"{rate / hop / resolved['min_dwell_frames']:.1f} Hz 时驻留不足、逐跳不可分辨")
    else:
        reason = None

    hop_entries = []
    for item in hops:
        hop_entries.append({
            "id": item["id"],
            "session_id": item["session_id"],
            "center_hz": round(item["center_hz"], 3),
            "centroid_hz": round(item["centroid_hz"], 3),
            "bandwidth_hz": round(item["bandwidth_hz"], 3),
            "f_low_hz": round(item["f_low_hz"], 3),
            "f_high_hz": round(item["f_high_hz"], 3),
            "t_start_s": round(item["t_start_s"], 6),
            "t_end_s": round(item["t_end_s"], 6),
            "dwell_s": round(item["dwell_s"], 6),
            "power_dbfs": round(float(10.0 * np.log10(max(item["power_linear"], 1e-30))), 3),
            "snr_db": round(item["snr_db"], 3),
            "confidence": round(item["confidence"], 3),
            "frame_count": int(item["frame_count"]),
            "bin_count": int(item["bin_count"]),
            "band_nfft": int(item["band_nfft"]),
            "mask_f_low_hz": round(item["mask_f_low_hz"], 3),
            "mask_f_high_hz": round(item["mask_f_high_hz"], 3),
            "mask_bandwidth_hz": round(item["mask_bandwidth_hz"], 3),
        })

    session_entries = []
    for session in sessions:
        session_entries.append({
            "session_id": session["session_id"],
            "hop_count": int(session["hop_count"]),
            "sequence": [round(value, 3) for value in session["sequence"]],
            "hop_frequencies_hz": [round(value, 3) for value in session["hop_frequencies_hz"]],
            "channel_spacing_hz": (None if session["channel_spacing_hz"] is None
                                   else round(session["channel_spacing_hz"], 3)),
            "hop_span_hz": round(session["hop_span_hz"], 3),
            "hop_bandwidth_hz": round(session["hop_bandwidth_hz"], 3),
            "hop_period_s": (None if session["hop_period_s"] is None
                             else round(session["hop_period_s"], 6)),
            "hop_rate_hz": (None if session["hop_rate_hz"] is None
                            else round(session["hop_rate_hz"], 3)),
            "duty_cycle": (None if session["duty_cycle"] is None
                           else round(session["duty_cycle"], 4)),
            "dwell_median_s": round(session["dwell_median_s"], 6),
            "dwell_min_s": round(session["dwell_min_s"], 6),
            "dwell_max_s": round(session["dwell_max_s"], 6),
            "center_hz": round(session["center_hz"], 3),
            "bandwidth_hz": round(session["bandwidth_hz"], 3),
            "f_low_hz": round(session["f_low_hz"], 3),
            "f_high_hz": round(session["f_high_hz"], 3),
            "t_start_s": round(session["t_start_s"], 6),
            "t_end_s": round(session["t_end_s"], 6),
            "power_dbfs": round(float(10.0 * np.log10(max(session["power_linear"], 1e-30))), 3),
            "snr_db": round(session["snr_db"], 3),
            "session_detection_id": session["session_detection_id"],
        })

    summary = {
        "contract": HOP_CONTRACT,
        "algorithm": HOP_ALGORITHM,
        "snr_definition": DETECT_SNR_DEFINITION,
        "frequency_reference": "baseband_offset",
        "sample_rate_hz": rate,
        "sample_count": int(x.size),
        "duration_s": duration,
        "nfft": nfft,
        "hop_samples": hop,
        "frame_count": int(psd.shape[0]),
        "config": {
            "nfft": nfft,
            "threshold_db": threshold_db,
            "smooth_frames": resolved["smooth_frames"],
            "min_bandwidth_hz": round(resolved["min_bandwidth_hz"], 6),
            "min_dwell_s": round(resolved["min_dwell_s"], 9),
            "merge_bins": resolved["merge_bins"],
            "max_gap_frames": resolved["max_gap_frames"],
            "max_gap_bins": resolved["max_gap_bins"],
            "transition_ratio": resolved["transition_ratio"],
            "max_hops": resolved["max_hops"],
        },
        "freq_resolution_hz": resolution,
        "frame_interval_s": hop / rate,
        "noise_floor_dbfs_per_hz": round(noise_floor_db, 3),
        "threshold_dbfs_per_hz": round(threshold_dbfs, 3),
        "dwell_limit_s": dwell_limit_s,
        "hop_rate_limit_hz": 1.0 / dwell_limit_s,
        "resolvable": resolvable,
        "reason": reason,
        "transition_frames": int(transition_frames[0]),
        "hops": hop_entries,
        "sessions": session_entries,
    }
    if with_sessions:
        baseline, _ = detect_signals(x, rate, {
            "nfft": nfft,
            "threshold_db": threshold_db,
            "min_bandwidth_hz": resolved["min_bandwidth_hz"],
            "merge_bins": resolved["merge_bins"],
        })
        summary["baseline"] = {
            "contract": baseline["contract"],
            "algorithm": baseline["algorithm"],
            "threshold_dbfs_per_hz": baseline["threshold_dbfs_per_hz"],
            "detections": baseline["detections"],
        }
        # 会话与会话级检测互链：取频带重叠最多的一条（跳频会话必然是 hopping 实例）
        for session in summary["sessions"]:
            best, best_overlap = None, 0.0
            for detection in baseline["detections"]:
                overlap = min(session["f_high_hz"], detection["f_high_hz"]) \
                    - max(session["f_low_hz"], detection["f_low_hz"])
                if overlap > best_overlap:
                    best, best_overlap = detection["id"], overlap
            session["session_detection_id"] = best

    hop_boxes = [[item["f_low_hz"], item["f_high_hz"], item["t_start_s"], item["t_end_s"]]
                 for item in hop_entries]
    arrays = {
        "frequency": frequencies,
        "frame_time": (starts + (nfft - 1) / 2.0) / rate,
        "spectrogram_db": (smoothed_db).astype(np.float32),
        "spectrogram_raw_db": (10.0 * np.log10(np.maximum(psd, 1e-30))).astype(np.float32),
        "spectrum_db": psd_average_db,
        "spectrum_median_db": 10.0 * np.log10(np.maximum(np.median(psd, axis=0), 1e-30)),
        "threshold_db": np.array([threshold_dbfs], dtype=np.float64),
        "noise_floor_db": np.array([noise_floor_db], dtype=np.float64),
        "hop_boxes": np.array(hop_boxes, dtype=np.float64).reshape(-1, 4),
        "hop_id": np.array([item["id"] for item in hop_entries], dtype=np.int64),
        "hop_session_id": np.array([item["session_id"] for item in hop_entries], dtype=np.int64),
        "hop_snr_db": np.array([item["snr_db"] for item in hop_entries], dtype=np.float64),
        "hop_power_dbfs": np.array([item["power_dbfs"] for item in hop_entries], dtype=np.float64),
    }
    return summary, arrays
