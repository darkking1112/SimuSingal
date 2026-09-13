"""工作区数据盘点与安全清理的单元测试（不需要 GUI，也不写运行记录）。"""
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from common.storage import utc_now
from signal_analysis.maintenance import (DEFAULT_JOB_RETENTION_DAYS, SETTINGS_NAME,
                                         apply_cleanup, build_report, format_bytes,
                                         preview_cleanup, read_settings, write_settings)
from signal_analysis.storage import Workspace


def make_asset(workspace, name="参考样本", source="generated:test"):
    return workspace.add_samples(np.ones(64, dtype=np.complex64), 48_000.0, name, source)


def make_job(workspace, action="analysis", state="success", age_days=90, payload=b"x" * 128):
    job_id = f"{action}{age_days}".replace(".", "_")
    folder = workspace.root / "jobs" / job_id
    folder.mkdir(parents=True, exist_ok=True)
    finished = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
    (folder / "status.json").write_text(json.dumps(
        {"job_id": job_id, "action": action, "state": state,
         "started": finished, "finished": finished}), encoding="utf-8")
    (folder / "worker.log").write_bytes(payload)
    return folder


def test_format_bytes_handles_edges():
    assert format_bytes(0) == "0 B"
    assert format_bytes(-5) == "0 B"
    assert format_bytes(float("nan")) == "0 B"
    assert format_bytes(float("inf")) == "0 B"
    assert format_bytes(None) == "0 B"
    assert format_bytes("bad") == "0 B"
    assert format_bytes(999) == "999 B"
    assert format_bytes(2048) == "2.0 KiB"
    assert format_bytes(3 * 1024 ** 2) == "3.0 MiB"


def test_settings_round_trip_and_validation(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    assert read_settings(workspace) == {"extra_dirs": [], "job_retention_days":
                                        DEFAULT_JOB_RETENTION_DAYS}
    write_settings(workspace, extra_dirs=[str(tmp_path), str(tmp_path)], job_retention_days=7)
    settings = read_settings(workspace)
    assert settings["job_retention_days"] == 7
    assert settings["extra_dirs"] == [str(tmp_path)]
    with pytest.raises(ValueError):
        write_settings(workspace, job_retention_days=4000)
    # 工作区自身不能作为额外目录（否则容量会被重复计算）
    write_settings(workspace, extra_dirs=[str(workspace.root)])
    assert read_settings(workspace)["extra_dirs"] == []
    # 损坏的设置文件回退默认值而不是抛异常
    (tmp_path / "ws" / SETTINGS_NAME).write_text("{ not json", encoding="utf-8")
    assert read_settings(workspace)["job_retention_days"] == DEFAULT_JOB_RETENTION_DAYS


def test_report_counts_match_disk(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    asset = make_asset(workspace)
    workspace.save_run("analysis", {"mean": 1.0}, asset_id=asset["id"])
    make_job(workspace, action="detect", state="success", age_days=100)

    report = build_report(workspace, job_retention_days=30)
    categories = {item["key"]: item for item in report["categories"]}
    assert categories["assets"]["bytes"] == (workspace.root / asset["path"]).stat().st_size
    assert categories["assets"]["files"] == 1
    assert categories["catalog"]["bytes"] == (workspace.root / "catalog.sqlite3").stat().st_size
    assert categories["jobs"]["files"] == 2  # status.json + worker.log
    assert report["totals"]["bytes"] == report["totals"]["workspace_bytes"]
    assert report["totals"]["files"] == sum(item["files"] for item in report["categories"])
    # JSON 必须可安全序列化（无 NaN/Infinity），否则 result.json 写入会失败
    json.dumps(report, ensure_ascii=False, allow_nan=False)


def test_report_groups_runs_by_kind_and_index_redundancy(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    asset = make_asset(workspace)
    workspace.save_run("analysis", {"a": 1}, asset_id=asset["id"])
    workspace.save_run("detect", {"b": 2}, asset_id=asset["id"])
    workspace.save_run("detect", {"c": 3}, asset_id=asset["id"])

    report = build_report(workspace)
    by_kind = {item["kind"]: item for item in report["runs_by_kind"]}
    assert by_kind["detect"]["count"] == 2
    assert by_kind["detect"]["label"] == "能量检测"
    assert by_kind["analysis"]["count"] == 1
    # 索引冗余 = runs.result_json + asset_metadata.metadata_json，必然小于磁盘占用
    assert report["catalog"]["index_bytes"] > 0
    assert report["catalog"]["index_bytes"] < report["totals"]["bytes"]
    run_rows = report["tables"]["runs"]
    assert len(run_rows) == 3 and all(row["exists"] for row in run_rows)
    assert all(row["bytes"] >= row["sqlite_bytes"] for row in run_rows)


def test_report_detects_consistency_problems(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    asset = make_asset(workspace)
    run = workspace.save_run("analysis", {"a": 1}, asset_id=asset["id"])

    # 未入库文件 + 原子写残留
    (workspace.root / "assets" / "orphan.npy").write_bytes(b"0" * 40)
    (workspace.root / "assets" / "halfwritten.npy.tmp").write_bytes(b"0" * 20)
    # 未入库运行目录
    (workspace.root / "runs" / "ghost").mkdir()
    (workspace.root / "runs" / "ghost" / "result.json").write_text("{}", encoding="utf-8")
    # 入库但文件缺失
    (workspace.root / asset["path"]).unlink()

    report = build_report(workspace)
    consistency = report["consistency"]
    assert consistency["orphan_files_count"] == 1
    assert consistency["orphan_files"][0]["path"] == "assets/orphan.npy"
    assert consistency["tmp_files_count"] == 1
    assert consistency["orphan_run_dirs_count"] == 1
    assert consistency["missing_files_count"] == 1
    assert report["tables"]["assets"][0]["exists"] is False
    assert any("样本文件缺失" in item for item in report["warnings"])
    # 源资产已删除的运行记录单独指出（而不仅是“未入库目录”）
    assert report["consistency"]["orphan_runs_count"] == 0
    assert report["catalog"]["runs"] == 1
    assert run["run_id"]


def test_cleanup_preview_excludes_unfinished_jobs(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    make_job(workspace, action="detect", state="success", age_days=100)
    make_job(workspace, action="analysis", state="running", age_days=100)

    report = build_report(workspace, job_retention_days=30)
    targets = report["cleanup"]["targets"]
    assert [item["path"] for item in targets] == ["jobs/detect100"]
    assert report["cleanup"]["skipped_running"] == 1
    # 保留天数内不产生候选
    assert build_report(workspace, job_retention_days=365)["cleanup"]["targets"] == []
    preview = preview_cleanup(workspace, job_retention_days=30)
    assert preview["kind"] == "storage_preview"
    assert preview["bytes"] == report["cleanup"]["bytes"]


def test_apply_cleanup_deletes_only_candidates(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    asset = make_asset(workspace)
    run = workspace.save_run("analysis", {"a": 1}, asset_id=asset["id"])
    old_job = make_job(workspace, action="detect", state="success", age_days=100)
    fresh_job = make_job(workspace, action="detect", state="success", age_days=1)
    orphan = workspace.root / "assets" / "orphan.npy"
    orphan.write_bytes(b"0" * 40)

    result = apply_cleanup(workspace, ["jobs/detect100", "assets/orphan.npy"],
                           job_retention_days=30)
    assert result["kind"] == "storage_cleanup"
    assert {item["path"] for item in result["removed"]} == {"jobs/detect100", "assets/orphan.npy"}
    assert result["bytes"] > 0 and result["requested"] == 2
    assert not old_job.exists() and not orphan.exists()
    assert fresh_job.exists()
    # 已入库的资产与运行目录永不可删
    assert (workspace.root / asset["path"]).is_file()
    assert (workspace.root / "runs" / run["run_id"]).is_dir()
    assert workspace.list_assets()[0]["id"] == asset["id"]
    # 审计日志每次清理追加一行 JSON
    entries = [json.loads(line) for line
               in (workspace.root / "maintenance.log").read_text(encoding="utf-8").splitlines()]
    assert entries[-1]["removed"] and entries[-1]["bytes"] == result["bytes"]


@pytest.mark.parametrize("bad", ["../secret", "/etc/passwd", "exports/keep.bin",
                                "runs", "assets/../runs", "  ", "C:/tmp/x"])
def test_apply_cleanup_rejects_unsafe_targets(tmp_path, bad):
    workspace = Workspace(tmp_path / "ws")
    make_asset(workspace)
    exported = workspace.root / "exports" / "keep.bin"
    exported.parent.mkdir()
    exported.write_bytes(b"0" * 32)

    result = apply_cleanup(workspace, [bad], job_retention_days=0)
    assert result["removed"] == []
    assert result["skipped"]
    assert exported.is_file()
    assert (workspace.root / "assets").is_dir()


def test_apply_cleanup_ignores_blank_requests(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    make_asset(workspace)

    result = apply_cleanup(workspace, ["", None, "   ", {"path": ""}], job_retention_days=0)
    # 空路径与缺省的 dict 直接忽略；只有纯空白算“被请求但被拒绝”
    assert result["requested"] == 1
    assert result["removed"] == []
    assert [item["path"] for item in result["skipped"]] == ["   "]
    assert (workspace.root / "assets").is_dir()


def test_apply_cleanup_still_skips_running_and_keeps_catalog_runs(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    asset = make_asset(workspace)
    run = workspace.save_run("analysis", {"a": 1}, asset_id=asset["id"])
    make_job(workspace, action="analysis", state="running", age_days=100)

    # 即便是显式请求，运行中的任务与已入库运行目录也不会被删除
    result = apply_cleanup(workspace, ["jobs/analysis100", f"runs/{run['run_id']}"],
                           job_retention_days=0)
    assert result["removed"] == []
    assert len(result["skipped"]) == 2
    assert (workspace.root / "runs" / run["run_id"]).is_dir()
    assert (workspace.root / "jobs" / "analysis100").is_dir()


def test_extra_dirs_are_statistics_only(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    outside = tmp_path / "training" / "data"
    outside.mkdir(parents=True)
    (outside / "amc.json").write_bytes(b"y" * 512)
    missing = tmp_path / "training" / "runs"

    report = build_report(workspace, extra_dirs=[str(outside), str(missing), str(workspace.root)])
    # 工作区自身被去重，不计入 extra_bytes
    assert [row["path"] for row in report["extra_dirs"]] == [str(outside.resolve()),
                                                            str(missing.resolve())]
    assert report["totals"]["extra_bytes"] == 512
    assert report["totals"]["bytes"] == report["totals"]["workspace_bytes"] + 512
    assert report["extra_dirs"][0]["files"] == 1
    assert report["extra_dirs"][1]["exists"] is False
    assert any("额外目录不存在" in item for item in report["warnings"])
    # 额外目录不参与一致性检查，也不产生清理候选
    assert report["consistency"]["orphan_files_count"] == 0
    assert report["cleanup"]["targets"] == []


def test_report_is_stable_on_empty_workspace(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    report = build_report(workspace)
    assert report["kind"] == "storage_report"
    assert report["totals"]["bytes"] >= 0
    assert report["runs_by_kind"] == []
    assert report["cleanup"]["targets"] == []
    assert report["consistency"]["assets"] == 0
    json.dumps(report, ensure_ascii=False, allow_nan=False)


def test_report_marks_jobs_without_status_file(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    folder = workspace.root / "jobs" / "halfdone"
    folder.mkdir(parents=True)
    (folder / "worker.log").write_bytes(b"z" * 16)
    time.sleep(0.001)

    report = build_report(workspace, job_retention_days=0)
    row = report["tables"]["jobs"][0]
    assert row["state"] == "未知" and row["action"] == "未知"
    assert row["has_status"] is False
    # 状态未知的任务视为“未结束”，不会被误删
    assert report["cleanup"]["targets"] == []
    assert report["cleanup"]["skipped_running"] == 1
    # 用目录 mtime 估算年龄，保证 0 天保留期下也不会得到负数
    assert row["age_days"] >= 0
    assert Path(report["root"]) == workspace.root
    assert report["generated_at"] <= utc_now()
