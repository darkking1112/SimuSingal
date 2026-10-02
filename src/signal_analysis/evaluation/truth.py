"""真值构造：生成器摘要 → 会话级（``detect_result_v1``）与逐跳（``fh_hops_v1``）真值。"""

import numpy as np

from ..core_api import occupied_interval
from ..algorithms.generation.iqgen import (
    _fh_hop_boundaries,
)


def _finite_or_none(value):
    """Float value or ``None``; results are serialised with allow_nan=False."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _rounded(value, digits=6):
    number = _finite_or_none(value)
    return None if number is None else round(number, digits)


def signal_truth(summary):
    """Truth list for one IQ generator summary (:func:`plan_signal` output).

    One entry per generated signal. Hopping signals follow the agreed
    "one session, one instance" semantics: the reported band is the session
    band of the channels that were **actually visited** (the generator draws
    the hop sequence randomly, so in a short record the outer channels may
    never be used). Falling back to ``bandwidth_actual`` (``span +
    hop_bandwidth``) keeps the entry well defined when the hop set is only
    described nominally.
    """
    if not isinstance(summary, dict):
        return []
    duration = _finite_or_none(summary.get("duration_s"))
    truth = []
    for index, entry in enumerate(summary.get("signals") or []):
        if not isinstance(entry, dict):
            continue
        offset = _finite_or_none(entry.get("offset"))
        width = _finite_or_none(entry.get("bandwidth_actual", entry.get("bandwidth")))
        if offset is None or width is None or width <= 0:
            continue
        mode = str(entry.get("mode", ""))
        hopping = mode.startswith("fh")
        low, high = occupied_interval(offset, width, mode, entry.get("side"))
        visited_channels = None
        if hopping:
            hop_bw = _finite_or_none(entry.get("hop_bandwidth"))
            visited = [value for value in (_finite_or_none(point)
                                          for point in (entry.get("hop_points") or []))
                       if value is not None]
            if hop_bw and hop_bw > 0 and visited:
                channels = sorted(set(visited))
                low = channels[0] - hop_bw / 2.0
                high = channels[-1] + hop_bw / 2.0
                visited_channels = len(channels)
        truth.append({
            "index": index,
            "mode": mode,
            "hopping": hopping,
            "nominal_offset_hz": _rounded(offset),
            "nominal_bandwidth_hz": _rounded(width),
            "center_hz": _rounded((low + high) / 2.0),
            "bandwidth_hz": _rounded(high - low),
            "f_low_hz": _rounded(low),
            "f_high_hz": _rounded(high),
            "t_start_s": 0.0,
            "t_end_s": duration,
            "power_dbfs": _finite_or_none(entry.get("power_dbfs_actual")),
            "snr_inband_db": _finite_or_none(entry.get("snr_inband_db")),
            "session_id": index,
            "visited_channels": visited_channels,
        })
    return truth


def hop_truth(summary):
    """Per-dwell truth for the hopping contract (``fh_hops_v1``).

    One entry per hop the generator actually emitted, i.e. per entry of
    ``hop_points`` — the hop sequence is drawn randomly, so consecutive hops
    may reuse a channel (indistinguishable from a longer dwell), and the
    record may end mid-sequence. Only ``fh*`` modes have this granularity,
    every other mode returns ``[]``: their truth is the whole transmission,
    which :func:`signal_truth` already describes.

    The **band** of every entry is ``hop_bandwidth`` around the visited hop
    point, not the generator's ``occupied_bandwidth`` (which is the measured
    occupancy of the whole signal). With that convention the union of all
    entries is exactly the session band reported by :func:`signal_truth`,
    so the two granularities can be scored side by side.

    ``snr_inband_db`` is converted to the per-hop bandwidth:
    ``snr_inband_db + 10*log10(bandwidth_actual / hop_bandwidth)``. The
    generator reports in-band SNR over its full actual bandwidth, while a
    hop detector measures the power of a single dwell inside one hop
    bandwidth, so the conversion subtracts ``N0·(B_actual - B_hop)``.
    """
    if not isinstance(summary, dict):
        return []
    rate = _finite_or_none(summary.get("sample_rate_hz"))
    duration = _finite_or_none(summary.get("duration_s"))
    truth = []
    for session_index, entry in enumerate(summary.get("signals") or []):
        if not isinstance(entry, dict):
            continue
        mode = str(entry.get("mode", ""))
        if not mode.startswith("fh"):
            continue
        hop_bw = _finite_or_none(entry.get("hop_bandwidth"))
        points = [value for value in (_finite_or_none(point)
                                      for point in (entry.get("hop_points") or []))
                  if value is not None]
        if not hop_bw or hop_bw <= 0 or not points or not rate:
            continue
        count = _finite_or_none(entry.get("sample_count")) or (
            rate * duration if duration else None)
        if not count:
            continue
        width = _finite_or_none(entry.get("bandwidth_actual", entry.get("bandwidth")))
        hop_rate = _finite_or_none(entry.get("hop_rate"))
        power_dbfs = _finite_or_none(entry.get("power_dbfs_actual"))
        session_snr = _finite_or_none(entry.get("snr_inband_db"))
        correction = (10.0 * float(np.log10(width / hop_bw))
                      if width and width > 0 else 0.0)
        boundaries = _fh_hop_boundaries(int(round(count)), len(points))
        for hop_index, point in enumerate(points):
            start = float(boundaries[hop_index]) / rate
            stop = float(boundaries[hop_index + 1]) / rate
            truth.append({
                "index": len(truth),
                "mode": mode,
                "session_index": session_index,
                "hop_index": hop_index,
                "hop_count": len(points),
                "center_hz": _rounded(point),
                "bandwidth_hz": _rounded(hop_bw),
                "f_low_hz": _rounded(point - hop_bw / 2.0),
                "f_high_hz": _rounded(point + hop_bw / 2.0),
                "t_start_s": _rounded(start),
                "t_end_s": _rounded(stop),
                "dwell_s": _rounded(stop - start),
                "hop_rate_hz": _rounded(hop_rate),
                "power_dbfs": power_dbfs,
                "snr_inband_db": (None if session_snr is None
                                  else _rounded(session_snr + correction)),
            })
    return truth
