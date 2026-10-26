from pathlib import Path

from analysis.targeted_extension.replay_final_analysis import build_commands


def test_final_replay_covers_sources_scopes_and_response_designs(tmp_path: Path) -> None:
    commands = build_commands(
        tmp_path / "scripts",
        tmp_path / "retained",
        tmp_path / "data",
        tmp_path / "models",
        tmp_path / "output/Reviewer_repair",
    )
    joined = [" ".join(item) for item in commands]
    assert len(commands) == 18
    assert sum("analyse_repaired_controls.py" in item for item in joined) == 4
    assert sum("analyse_repaired_spatial.py" in item for item in joined) == 4
    assert sum("analyse_seasonal_reallocation.py" in item for item in joined) == 4
    assert sum("analyse_common_baseline_dr.py" in item for item in joined) == 2
    assert sum("analyse_own_baseline_dr.py" in item for item in joined) == 4
    assert any("ASHRAE=" in item and "CTCD=" in item for item in joined)
