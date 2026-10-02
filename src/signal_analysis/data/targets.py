"""目标与会话/跳的参考参数版本（TargetMixin）。

“目标”= 资产内的信号会话或跳；参考参数追加式版本化（revision/version），
旧版本不改、不删除。由 :class:`signal_analysis.data.workspace.Workspace`
组合使用。
"""

import uuid

from common.storage import utc_now
from ..storage.schema import SCOPES, TASKS, VERSION_SOURCES, _MODULATION_NAMES
from ..storage.utils import (_clean_text, _enum, _integer, _json_safe, _json_text,
                             _number, _optional_text, _target_sort_key)


class TargetMixin:
    """目标的登记、查询与参考参数版本追加。"""

    # ------------------------------------------------------------------ 生成器摘要 → 目标
    def _register_generation_targets(self, conn, asset_id, summary, sample_count, rate):
        """把生成器摘要登记为 targets / target_versions（幂等：已有目标则跳过）。"""
        from ..evaluation import hop_truth, signal_truth

        exists = conn.execute("SELECT 1 FROM targets WHERE asset_id=? LIMIT 1",
                              (asset_id,)).fetchone()
        if exists:
            return
        raw_signals = [entry for entry in (summary.get("signals") or [])
                       if isinstance(entry, dict)]
        truth = signal_truth(summary)
        hopping_hops = hop_truth(summary)
        if not truth:
            return
        for index, entry in enumerate(truth):
            mode = str(entry.get("mode", ""))
            hopping = bool(entry.get("hopping"))
            scope = "session" if (hopping or len(truth) > 1) else "whole_record"
            # AMC 适用标记：有规范调制名的样式置位；跳频/演示样式保持 0。
            target_id = self._insert_target_row(
                conn, asset_id=asset_id, target_key=f"s{index}", scope=scope,
                parent_target_id=None, hop_index=None, for_detection=1,
                for_amc=1 if mode in _MODULATION_NAMES else 0)
            raw = raw_signals[index] if index < len(raw_signals) else {}
            mapped = {key: _json_safe(value) for key, value in raw.items()
                      if key not in ("mode", "offset", "bandwidth", "bandwidth_actual",
                                     "power_dbfs", "power_dbfs_actual", "snr_inband_db")}
            self._insert_target_version_row(
                conn, target_id=target_id, source="generator",
                sample_start=0, sample_end=int(sample_count),
                f_low_hz=entry.get("f_low_hz"), f_high_hz=entry.get("f_high_hz"),
                nominal_center_hz=entry.get("nominal_offset_hz"),
                nominal_bandwidth_hz=entry.get("nominal_bandwidth_hz"),
                signal_type=None, waveform_mode=mode,
                modulation=_MODULATION_NAMES.get(mode),
                symbol_rate_baud=_number(raw.get("symbol_rate"), "符号率", minimum=0.0)
                if raw.get("symbol_rate") is not None else None,
                hop_rate_hz=_number(raw.get("hop_rate"), "跳速", minimum=0.0)
                if raw.get("hop_rate") is not None else None,
                is_hopping=1 if hopping else 0,
                snr_db=entry.get("snr_inband_db"), snr_definition="inband_snr_v1",
                power_dbfs=entry.get("power_dbfs"), params_json=mapped)
            if not hopping:
                continue
            for hop in hopping_hops:
                if hop.get("session_index") != index:
                    continue
                hop_index = int(hop["hop_index"])
                hop_target = self._insert_target_row(
                    conn, asset_id=asset_id, target_key=f"s{index}.h{hop_index}",
                    scope="hop", parent_target_id=target_id, hop_index=hop_index,
                    for_detection=1, for_amc=0)
                self._insert_target_version_row(
                    conn, target_id=hop_target, source="generator",
                    sample_start=int(round(float(hop["t_start_s"]) * rate)),
                    sample_end=int(round(float(hop["t_end_s"]) * rate)),
                    f_low_hz=hop.get("f_low_hz"), f_high_hz=hop.get("f_high_hz"),
                    nominal_center_hz=hop.get("center_hz"),
                    nominal_bandwidth_hz=hop.get("bandwidth_hz"),
                    signal_type=None, waveform_mode=mode, modulation=None,
                    symbol_rate_baud=None, hop_rate_hz=hop.get("hop_rate_hz"),
                    is_hopping=1, snr_db=hop.get("snr_inband_db"),
                    snr_definition="inband_snr_v1", power_dbfs=hop.get("power_dbfs"),
                    params_json={"hop_index": hop_index, "hop_count": hop.get("hop_count")})

    def _register_placeholder_target(self, conn, asset_id, sample_count):
        """无生成摘要的资产：写“未知”占位目标，不猜测任何物理参数。"""
        target_id = self._insert_target_row(
            conn, asset_id=asset_id, target_key="s0", scope="whole_record",
            parent_target_id=None, hop_index=None, for_detection=1, for_amc=1)
        self._insert_target_version_row(
            conn, target_id=target_id, source="import",
            sample_start=0, sample_end=int(sample_count), f_low_hz=None, f_high_hz=None,
            nominal_center_hz=None, nominal_bandwidth_hz=None, signal_type=None,
            waveform_mode=None, modulation=None, symbol_rate_baud=None, hop_rate_hz=None,
            is_hopping=None, snr_db=None, snr_definition=None, power_dbfs=None,
            params_json=None)

    # ------------------------------------------------------------------ 目标
    def _insert_target_row(self, conn, *, asset_id, target_key, scope,
                           parent_target_id, hop_index, for_detection, for_amc):
        target_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO targets (id, asset_id, target_key, scope, parent_target_id, "
            "hop_index, for_detection, for_amc, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (target_id, asset_id, target_key, scope, parent_target_id, hop_index,
             int(bool(for_detection)), int(bool(for_amc)), utc_now()))
        return target_id

    def add_target(self, asset_id, target_key, scope, *, parent_target_id=None,
                   hop_index=None, for_detection=0, for_amc=0):
        asset = self.get_asset(asset_id)
        key = _clean_text(target_key, "目标键", 64)
        scope = _enum(scope, SCOPES, "目标粒度")
        hop_index = _integer(hop_index, "跳序号", minimum=0)
        if scope == "hop":
            if parent_target_id is None:
                raise ValueError("跳目标必须指定父会话目标")
            if hop_index is None:
                raise ValueError("跳目标必须给出 hop_index")
        else:
            hop_index = None
        if parent_target_id is not None:
            parent = self.get_target(parent_target_id)
            if parent["asset_id"] != asset["id"]:
                raise ValueError("父目标必须属于同一资产")
        with self.connect() as conn:
            try:
                target_id = self._insert_target_row(
                    conn, asset_id=asset_id, target_key=key, scope=scope,
                    parent_target_id=parent_target_id, hop_index=hop_index,
                    for_detection=for_detection, for_amc=for_amc)
            except Exception as exc:
                if "UNIQUE" in str(exc):
                    raise ValueError("同一资产内目标键必须唯一") from exc
                raise
        return self.get_target(target_id)

    def get_target(self, target_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM targets WHERE id=?", (target_id,)).fetchone()
        if row is None:
            raise ValueError("目标不存在")
        return dict(row)

    def update_target_flags(self, target_id, *, for_detection=None, for_amc=None):
        self.get_target(target_id)
        with self.connect() as conn:
            if for_detection is not None:
                conn.execute("UPDATE targets SET for_detection=? WHERE id=?",
                             (int(bool(for_detection)), target_id))
            if for_amc is not None:
                conn.execute("UPDATE targets SET for_amc=? WHERE id=?",
                             (int(bool(for_amc)), target_id))
        return self.get_target(target_id)

    def list_targets(self, asset_id, *, with_current=True):
        self.get_asset(asset_id)
        with self.connect() as conn:
            rows = [dict(row) for row in conn.execute(
                "SELECT * FROM targets WHERE asset_id=?", (asset_id,))]
            versions = [dict(row) for row in conn.execute(
                "SELECT * FROM target_versions WHERE target_id IN "
                "(SELECT id FROM targets WHERE asset_id=?)", (asset_id,))]
        current = {}
        for version in versions:
            kept = current.get(version["target_id"])
            if kept is None or version["version_no"] > kept["version_no"]:
                current[version["target_id"]] = version
        rows.sort(key=_target_sort_key)
        if with_current:
            for row in rows:
                row["current"] = current.get(row["id"])
        return rows

    def append_target_version(self, target_id, *, source, note=None, sample_start=None,
                              sample_end=None, f_low_hz=None, f_high_hz=None,
                              center_hz=None, bandwidth_hz=None, nominal_center_hz=None,
                              nominal_bandwidth_hz=None, signal_type=None,
                              waveform_mode=None, modulation=None, symbol_rate_baud=None,
                              hop_rate_hz=None, is_hopping=None, snr_db=None,
                              snr_definition=None, power_dbfs=None, params_json=None):
        target = self.get_target(target_id)
        asset = self.get_asset(target["asset_id"])
        source = _enum(source, VERSION_SOURCES, "来源")
        start = _integer(sample_start, "起始采样点", minimum=0)
        end = _integer(sample_end, "结束采样点", minimum=0)
        if (start is None) != (end is None):
            raise ValueError("时间范围必须同时给出起止采样点")
        if start is not None:
            if end <= start:
                raise ValueError("结束采样点必须大于起始采样点")
            if end > int(asset["sample_count"]):
                raise ValueError("时间范围不能超出资产采样点数")
        low = _number(f_low_hz, "频率下限")
        high = _number(f_high_hz, "频率上限")
        if (low is None) != (high is None):
            raise ValueError("频率范围必须同时给出上下限")
        if low is not None and high <= low:
            raise ValueError("频率上限必须大于下限")
        derived_center = derived_bandwidth = None
        if low is not None:
            derived_center = (low + high) / 2.0
            derived_bandwidth = high - low
            given_center = _number(center_hz, "中心频率")
            given_bandwidth = _number(bandwidth_hz, "带宽", minimum=0.0)
            tolerance = max(1e-6 * max(1.0, abs(derived_center)), 1e-9)
            if given_center is not None and abs(given_center - derived_center) > tolerance:
                raise ValueError("中心频率与频率边界不一致（应为边界派生值）")
            if given_bandwidth is not None and \
                    abs(given_bandwidth - derived_bandwidth) > tolerance:
                raise ValueError("带宽与频率边界不一致（应为边界派生值）")
        note = _optional_text(note, "备注", 500)
        nominal_bandwidth_hz = _number(nominal_bandwidth_hz, "名义带宽", minimum=0.0)
        with self.connect() as conn:
            row = conn.execute("SELECT MAX(version_no) FROM target_versions WHERE target_id=?",
                               (target_id,)).fetchone()
            previous = conn.execute(
                "SELECT id FROM target_versions WHERE target_id=? ORDER BY version_no DESC "
                "LIMIT 1", (target_id,)).fetchone()
            return self._insert_target_version_row(
                conn, target_id=target_id, source=source, note=note,
                sample_start=start, sample_end=end, f_low_hz=low, f_high_hz=high,
                nominal_center_hz=_number(nominal_center_hz, "名义中心频率"),
                nominal_bandwidth_hz=nominal_bandwidth_hz, signal_type=signal_type,
                waveform_mode=waveform_mode, modulation=modulation,
                symbol_rate_baud=_number(symbol_rate_baud, "符号率", minimum=0.0),
                hop_rate_hz=_number(hop_rate_hz, "跳速", minimum=0.0),
                is_hopping=None if is_hopping is None else int(bool(is_hopping)),
                snr_db=_number(snr_db, "SNR"), snr_definition=snr_definition,
                power_dbfs=_number(power_dbfs, "功率"), params_json=params_json,
                version_no=int(row[0] or 0) + 1,
                supersedes_id=previous["id"] if previous else None)

    def _insert_target_version_row(self, conn, *, target_id, source, version_no=None,
                                   supersedes_id=None, note=None, sample_start=None,
                                   sample_end=None, f_low_hz=None, f_high_hz=None,
                                   nominal_center_hz=None, nominal_bandwidth_hz=None,
                                   signal_type=None, waveform_mode=None, modulation=None,
                                   symbol_rate_baud=None, hop_rate_hz=None, is_hopping=None,
                                   snr_db=None, snr_definition=None, power_dbfs=None,
                                   params_json=None):
        if version_no is None:
            row = conn.execute("SELECT MAX(version_no) FROM target_versions "
                               "WHERE target_id=?", (target_id,)).fetchone()
            version_no = int(row[0] or 0) + 1
        # 派生值与 evaluation._rounded 同为 6 位小数：保证目标真值与生成器摘要
        # 在逐位口径上可比（避免 (high-low) 的浮点噪声污染比较与报表）
        derived_center = (round((f_low_hz + f_high_hz) / 2.0, 6)
                          if f_low_hz is not None and f_high_hz is not None else None)
        derived_bandwidth = (round(f_high_hz - f_low_hz, 6)
                             if f_low_hz is not None and f_high_hz is not None else None)
        version_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO target_versions (id, target_id, version_no, supersedes_id, source, "
            "created_at, note, sample_start, sample_end, f_low_hz, f_high_hz, center_hz, "
            "bandwidth_hz, nominal_center_hz, nominal_bandwidth_hz, signal_type, "
            "waveform_mode, modulation, symbol_rate_baud, hop_rate_hz, is_hopping, snr_db, "
            "snr_definition, power_dbfs, params_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (version_id, target_id, version_no, supersedes_id, source, utc_now(), note,
             sample_start, sample_end, f_low_hz, f_high_hz, derived_center,
             derived_bandwidth, nominal_center_hz, nominal_bandwidth_hz, signal_type,
             waveform_mode, modulation, symbol_rate_baud, hop_rate_hz, is_hopping,
             snr_db, snr_definition, power_dbfs, _json_text(params_json, "样式参数")))
        row = conn.execute("SELECT * FROM target_versions WHERE id=?",
                           (version_id,)).fetchone()
        return dict(row)

    def get_target_version(self, version_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM target_versions WHERE id=?",
                               (version_id,)).fetchone()
        if row is None:
            raise ValueError("目标参数版本不存在")
        return dict(row)

    def current_target_version(self, target_id):
        self.get_target(target_id)
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM target_versions WHERE target_id=? "
                               "ORDER BY version_no DESC LIMIT 1", (target_id,)).fetchone()
        return dict(row) if row else None

    def list_target_versions(self, target_id):
        self.get_target(target_id)
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM target_versions WHERE target_id=? "
                                "ORDER BY version_no", (target_id,)).fetchall()
        return [dict(row) for row in rows]

    def collection_targets(self, collection_id, *, task=None, with_current=True):
        """集合内全部目标（可按任务适用标记过滤），附资产来源与当前参数版本。"""
        self.get_collection(collection_id)
        flag = None
        if task is not None:
            flag = {"detection": "for_detection", "amc": "for_amc"}.get(
                _enum(task, TASKS, "任务"))
        clause = f"AND t.{flag}=1" if flag else ""
        with self.connect() as conn:
            rows = [dict(row) for row in conn.execute(
                "SELECT t.*, a.name AS asset_name, a.source_kind AS asset_source_kind, "
                "a.origin_group_id AS origin_group_id, a.sample_rate AS asset_sample_rate, "
                "a.sample_count AS asset_sample_count, m.position AS collection_position "
                "FROM targets t JOIN collection_members m ON m.asset_id=t.asset_id "
                "JOIN assets a ON a.id=t.asset_id WHERE m.collection_id=? "
                f"{clause}", (collection_id,))]
            versions = [dict(row) for row in conn.execute(
                "SELECT v.* FROM target_versions v JOIN targets t ON t.id=v.target_id "
                "JOIN collection_members m ON m.asset_id=t.asset_id "
                "WHERE m.collection_id=?", (collection_id,))]
        current = {}
        for version in versions:
            kept = current.get(version["target_id"])
            if kept is None or version["version_no"] > kept["version_no"]:
                current[version["target_id"]] = version
        rows.sort(key=lambda row: (row["collection_position"],
                                   _target_sort_key(row), row["id"]))
        if with_current:
            for row in rows:
                row["current"] = current.get(row["id"])
        return rows
