"""Command line for unattended molab jobs: ``python -m twinmem4d.cli <command> ...``."""

import argparse
import logging
from pathlib import Path

from .config import load_config
from .data.episode import list_episodes


def _episodes(args):
    eps = list_episodes(args.episodes)
    if args.only:
        eps = [e for e in eps if any(s in e.name for s in args.only)]
    if not eps:
        raise SystemExit(f"no converted episodes under {args.episodes}")
    return eps


def main(argv=None):
    p = argparse.ArgumentParser(prog="twinmem4d")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("frontend", "memory-eval", "vlm"):
        s = sub.add_parser(name)
        s.add_argument("--episodes", required=True, help="folder of converted episodes")
        s.add_argument("--results", required=True, help="results/ folder holding registry.csv")
        s.add_argument("--config", required=False)
        s.add_argument("--only", nargs="*", help="substrings of episode names to keep")
        s.add_argument("--notes", default="")
        if name == "frontend":
            s.add_argument("--oracle", action="store_true")
            s.add_argument("--checkpoints", help="folder holding the weights named in the config (relative paths)")
        else:
            s.add_argument("--frontend-run", required=True)
        if name == "vlm":
            s.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16])
            s.add_argument("--queries-config", required=True, help="memory_eval.yaml (its `queries` section)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from . import pipelines

    eps = _episodes(args)
    if args.cmd == "frontend":
        from .frontend.runner import FrontendConfig

        cfg = load_config(FrontendConfig, args.config) if args.config else FrontendConfig()
        if args.checkpoints:
            for key in ("fastsam_checkpoint", "reid_weights"):
                value = Path(getattr(cfg, key))
                if not value.is_absolute():
                    setattr(cfg, key, str(Path(args.checkpoints) / value.name))
        out = pipelines.run_frontend(eps, cfg, args.results, args.config or "", oracle=args.oracle, notes=args.notes)
    elif args.cmd == "memory-eval":
        from .eval.memory_eval import MemoryEvalConfig

        cfg = load_config(MemoryEvalConfig, args.config) if args.config else MemoryEvalConfig()
        out = pipelines.run_memory_eval(eps, args.frontend_run, cfg, args.results, args.config or "", notes=args.notes)
    else:
        from .baselines.vlm import VLMConfig
        from .queries.generate import QueryConfig

        vlm_cfg = load_config(VLMConfig, args.config) if args.config else VLMConfig()
        q_cfg = load_config(QueryConfig, args.queries_config, section="queries")
        out = pipelines.run_vlm_baselines(eps, args.frontend_run, vlm_cfg, q_cfg, args.ks, args.results,
                                          args.config or "", notes=args.notes)
    print(Path(out))


if __name__ == "__main__":
    main()
