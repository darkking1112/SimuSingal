"""集合、分片、分页与统计的存储层测试（方案文档 §3）。"""
import json

import numpy as np
import pytest

from signal_analysis.core_api import generate_iq
from signal_analysis.storage import Workspace


def add_generated(workspace, mode="fm", seed=1, name=None, rate=200_000.0, duration=0.05):
    samples, summary = generate_iq(
        rate, duration,
        [{"mode": mode, "offset": 0.0, "bandwidth": 50_000.0, "power_dbfs": -6.0}],
        {"enabled": True, "snr_db": 15.0}, seed)
    return workspace.add_samples(samples, rate, name or f"样本{seed}",
                                 f"generated:iq_{mode}_v1",
                                 metadata={"generation": summary})


def test_collection_membership_order_and_scattered(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    alpha = workspace.create_collection("集合甲", description="甲")
    beta = workspace.create_collection("集合乙")
    first = add_generated(workspace, seed=1, name="一号")
    second = add_generated(workspace, seed=2, name="二号")
    third = add_generated(workspace, seed=3, name="三号")
    assert workspace.add_collection_member(alpha["id"], second["id"]) is True
    assert workspace.add_collection_member(alpha["id"], first["id"]) is True
    assert workspace.add_collection_member(alpha["id"], first["id"]) is False  # 幂等
    assert workspace.add_collection_member(beta["id"], second["id"]) is True
    # 成员位置从 0 递增（历史缺陷：0 被当成假值导致两个成员同为 0，排序随机）
    with workspace.connect() as conn:
        positions = {row["asset_id"]: row["position"] for row in conn.execute(
            "SELECT asset_id, position FROM collection_members WHERE collection_id=?",
            (alpha["id"],))}
    assert positions == {second["id"]: 0, first["id"]: 1}
    # 集合内按加入顺序（position），不是创建时间
    ordered = workspace.list_assets(limit=10, collection_id=alpha["id"])
    assert [asset["id"] for asset in ordered] == [second["id"], first["id"]]
    assert workspace.count_assets(collection_id=alpha["id"]) == 2
    # 零散资产 = 不在任何未归档集合中
    scattered = workspace.list_assets(limit=10, scattered=True)
    assert [asset["id"] for asset in scattered] == [third["id"]]
    assert workspace.count_assets(scattered=True) == 1
    # 同一资产属于多个集合：关系表多对多
    assert {item["name"] for item in workspace.collections_of_asset(second["id"])} == \
        {"集合甲", "集合乙"}
    # 移出集合只删成员行，资产本身仍在
    assert workspace.remove_collection_member(alpha["id"], second["id"]) is True
    assert workspace.get_asset(second["id"])["id"] == second["id"]
    assert workspace.count_assets(collection_id=alpha["id"]) == 1
    # 归档集合后，只属于它的资产重新成为零散资产
    workspace.archive_collection(beta["id"])
    assert [item["name"] for item in workspace.list_collections()] == ["集合甲"]
    assert {asset["id"] for asset in workspace.list_assets(limit=10, scattered=True)} == \
        {second["id"], third["id"]}
    with pytest.raises(ValueError, match="名称已存在"):
        workspace.create_collection("集合甲")


def test_asset_pagination_offset_and_keyset(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    created = [add_generated(workspace, seed=index, name=f"分页{index}")
               for index in range(7)]
    assert workspace.count_assets() == 7
    assert workspace.list_assets(limit=3, offset=7) == []
    pages = [workspace.list_assets(limit=3, offset=offset) for offset in (0, 3, 6)]
    ids = [asset["id"] for page in pages for asset in page]
    assert len(ids) == 7 and len(set(ids)) == 7
    # 键集分页：以上一页最后一行的 (created_at, id) 为起点，不重不漏
    collected, after = [], None
    while True:
        page = workspace.list_assets(limit=3, after=after)
        if not page:
            break
        collected.extend(asset["id"] for asset in page)
        last = page[-1]
        after = (last["created_at"], last["id"])
        assert len(page) <= 3
    assert collected == ids
    assert set(collected) == {asset["id"] for asset in created}
    with pytest.raises(ValueError):
        workspace.list_assets(limit=0)
    # 搜索与分页组合
    assert workspace.count_assets("分页1") == 1


def test_asset_archive_hides_from_listing(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    asset = add_generated(workspace, seed=1)
    workspace.archive_asset(asset["id"])
    assert workspace.list_assets(limit=10) == []
    assert workspace.count_assets() == 0
    assert workspace.list_assets(limit=10, include_archived=True)[0]["id"] == asset["id"]
    assert workspace.archived_assets()[0]["id"] == asset["id"]
    workspace.archive_asset(asset["id"], archived=False)
    assert workspace.count_assets() == 1


def test_shard_writer_roundtrip_seal_and_tamper(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    writer = workspace.create_shard(name="批量导入 1")
    samples_a, summary = generate_iq(
        200_000.0, 0.05,
        [{"mode": "fm", "offset": 0.0, "bandwidth": 50_000.0, "power_dbfs": -6.0}],
        {"enabled": True, "snr_db": 15.0}, 5)
    first = writer.append(samples_a, 200_000.0, "分片甲", "generated:iq_fm_v1",
                          metadata={"generation": summary})
    second = writer.append(np.ones(777, dtype=np.complex64), 48_000.0, "分片乙", "imported:x")
    assert first["storage_kind"] == "shard"
    assert first["shard_offset"] == 0 and first["shard_length"] == samples_a.size
    assert second["shard_offset"] == samples_a.size and second["shard_length"] == 777
    # 分片资产没有独立文件；读取按偏移定位并校验逐条哈希
    _, loaded = workspace.load_samples(first["id"])
    np.testing.assert_array_equal(loaded, samples_a)
    _, loaded_second = workspace.load_samples(second["id"])
    np.testing.assert_array_equal(loaded_second, np.ones(777, dtype=np.complex64))
    # 生成摘要同样登记目标
    assert workspace.list_targets(first["id"])[0]["current"]["source"] == "generator"
    with pytest.raises(ValueError):
        workspace.resolve_asset(first)
    shard = writer.seal()
    assert shard["record_count"] == 2
    assert shard["size_bytes"] == (samples_a.size + 777) * 8
    assert len(shard["sha256"]) == 64 and shard["sealed_at"]
    assert writer.seal()["id"] == shard["id"]  # 重复封存幂等
    with pytest.raises(ValueError, match="封存"):
        writer.append(np.ones(4, dtype=np.complex64), 1000.0, "封存后", "imported:x")
    # 篡改记录内容后哈希校验失败
    path = workspace.root / shard["path"]
    payload = bytearray(path.read_bytes())
    payload[0] ^= 0xFF
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="校验失败"):
        workspace.load_samples(first["id"])
    assert len(workspace.list_shards()) == 1


def test_shard_writer_abort_only_when_empty(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    writer = workspace.create_shard()
    writer.abort()
    assert workspace.list_shards() == []
    assert not writer.path.exists()
    other = workspace.create_shard()
    other.append(np.ones(8, dtype=np.complex64), 1000.0, "占用", "imported:x")
    with pytest.raises(ValueError, match="不能直接放弃"):
        other.abort()
    other.seal()
    writer_again = workspace.create_shard()
    writer_again.seal()


def test_collection_summary_axis_stats_and_targets(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    collection = workspace.create_collection("统计集", source_kind="generated")
    fm = add_generated(workspace, mode="fm", seed=1, name="FM")
    qpsk = add_generated(workspace, mode="qpsk", seed=2, name="QPSK")
    hop = add_generated(workspace, mode="fh_rc", seed=3, name="跳频")
    for asset in (fm, qpsk, hop):
        workspace.add_collection_member(collection["id"], asset["id"])
    summary = workspace.collection_summary(collection["id"])
    assert summary["asset_count"] == 3
    # 目标数含逐跳：两个单信号会话各 1 个，跳频资产为会话 + 若干跳
    hop_total = sum(1 for item in workspace.collection_targets(collection["id"])
                    if item["scope"] == "hop")
    assert summary["target_count"] == 2 + 1 + hop_total
    assert summary["hop_count"] == hop_total and hop_total > 0
    assert summary["source_kinds"] == {"generated": 3}
    modulations = {item["label"]: item["count"]
                   for item in workspace.target_axis_stats(collection["id"], "modulation",
                                                           scope="signal")}
    assert modulations == {"FM": 1, "QPSK": 1, "未知": 1}  # 跳频样式没有单一调制名
    waveform = {item["label"]: item["count"]
                for item in workspace.target_axis_stats(collection["id"], "waveform_mode",
                                                        scope="signal")}
    assert waveform == {"fm": 1, "qpsk": 1, "fh_rc": 1}
    hopping = {item["label"]: item["count"]
               for item in workspace.target_axis_stats(collection["id"], "is_hopping",
                                                       scope="signal")}
    assert hopping == {"否": 2, "是": 1}
    # 逐跳作用域的跳速统计
    hop_rate = workspace.target_axis_stats(collection["id"], "hop_rate_hz", scope="hop")
    assert sum(item["count"] for item in hop_rate) == hop_total
    snr = workspace.target_axis_stats(collection["id"], "snr_db",
                                      bins=[None, 10, 20, None], scope="signal")
    assert sum(item["count"] for item in snr) == 3
    assert {"label": "[10, 20)", "count": 3} in snr
    with pytest.raises(ValueError):
        workspace.target_axis_stats(collection["id"], "不存在")
    with pytest.raises(ValueError, match="目标粒度"):
        workspace.target_axis_stats(collection["id"], "modulation", scope="bad")
    # 集合目标列表：会话在前、逐跳按序号排列，且带当前参数版本
    targets = workspace.collection_targets(collection["id"])
    hop_targets = [item["target_key"] for item in targets if item["scope"] == "hop"]
    assert hop_targets[:5] == ["s0.h0", "s0.h1", "s0.h2", "s0.h3", "s0.h4"]
    assert targets[0]["target_key"] == "s0"
    assert targets[0]["current"]["source"] == "generator"
    assert "asset_name" in targets[0]
