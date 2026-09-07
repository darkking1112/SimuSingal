import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_static_dependency_boundaries():
    for package, forbidden in {"common": {"signal_analysis", "communication_sim"},
                               "signal_analysis": {"communication_sim", "simpy"},
                               "communication_sim": {"signal_analysis"}}.items():
        for path in (ROOT / "src" / package).glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    names = [node.module or ""]
                assert not {name.split('.')[0] for name in names} & forbidden, path


@pytest.mark.parametrize("package,forbidden", [("signal_analysis", "communication_sim"),
                                                ("communication_sim", "signal_analysis")])
def test_cli_does_not_load_other_project(package, forbidden, tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    code = (f"import sys; from {package}.cli import main; "
            f"assert main(['--workspace', {str(tmp_path)!r}, 'list']) == 0; "
            f"assert not any(n == '{forbidden}' or n.startswith('{forbidden}.') for n in sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True, env=env, capture_output=True)


def test_workspaces_and_tables_are_separate(tmp_path):
    from signal_analysis.storage import Workspace as AnalysisWorkspace
    from communication_sim.storage import Workspace as SimulationWorkspace
    analysis = AnalysisWorkspace(tmp_path / "analysis")
    simulation = SimulationWorkspace(tmp_path / "simulation")
    with simulation.connect() as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "assets" not in tables
    with pytest.raises(ValueError, match="另一项目"):
        SimulationWorkspace(analysis.root)
    with pytest.raises(ValueError, match="另一项目"):
        AnalysisWorkspace(simulation.root)


def test_legacy_database_not_overwritten(tmp_path):
    from signal_analysis.storage import Workspace
    database = tmp_path / "catalog.sqlite3"
    database.write_bytes(b"legacy test data")
    with pytest.raises(ValueError, match="旧版"):
        Workspace(tmp_path)
    assert database.read_bytes() == b"legacy test data"


def test_shared_reports_and_run_rollback(tmp_path):
    from common.storage import Workspace
    from common.reports import export_report
    store = Workspace(tmp_path)
    result = store.save_run("test", {"value": "<script>"})
    assert store.get_run(result["run_id"]) == result
    report = tmp_path / "report.html"
    export_report(result, report)
    assert "&lt;script&gt;" in report.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        store.save_run("test", {"value": float("nan")})
    assert len(store.list_runs()) == 1
    assert len(list((tmp_path / "runs").iterdir())) == 1
