import pandas as pd
import yaml

from twinmem4d.cli import main
from twinmem4d.registry import read_registry


def test_cli_oracle_frontend_then_memory_eval(swap_episode, tmp_path, capsys):
    episodes = swap_episode.root.parent
    results = tmp_path / "results"
    main(["frontend", "--episodes", str(episodes), "--results", str(results), "--oracle"])
    fe_run = capsys.readouterr().out.strip().splitlines()[-1]

    cfg = tmp_path / "memory_eval.yaml"
    cfg.write_text(yaml.safe_dump({
        "space": "3d", "methods": ["raw", "reconcile_proxy", "twin_aware"],
        "queries": {"on_plate_radius_px": 8.0, "plate_merge_px": 5.0, "space": "3d"},
    }))
    main(["memory-eval", "--episodes", str(episodes), "--results", str(results), "--frontend-run", fe_run,
          "--config", str(cfg)])
    eval_run = capsys.readouterr().out.strip().splitlines()[-1]

    reg = read_registry(results)
    assert list(reg.stage) == ["frontend_oracle", "memory_eval"]
    acc = pd.read_csv(f"{eval_run}/accuracy.csv")
    overall = acc[acc.qtype == "all"].set_index("method")["acc"]
    assert set(overall.index) == {"raw", "reconcile_proxy", "twin_aware"}
    assert (overall == 1.0).all()            # oracle tracks never break, so every variant is perfect
