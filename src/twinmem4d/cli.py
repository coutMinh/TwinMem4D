"""Command line for unattended molab jobs: ``python -m twinmem4d.cli <command> ...``.

Long steps run through here as detached processes so the notebook kernel stays free.
"""

import argparse
import logging
from pathlib import Path

from .config import load_config
from .data.episode import list_episodes

HF_MODELS = ("nvidia/DAM-3B", "sentence-transformers/sentence-t5-xl", "Qwen/Qwen2.5-VL-7B-Instruct")


def _episodes(args):
    eps = list_episodes(args.episodes)
    if args.only:
        eps = [e for e in eps if any(s in e.name for s in args.only)]
    if not eps:
        raise SystemExit(f"no converted episodes under {args.episodes}")
    return eps


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="twinmem4d")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("frontend", "memory-eval", "vlm", "dam-curve", "backend"):
        s = sub.add_parser(name)
        s.add_argument("--episodes", required=True, help="folder of converted episodes")
        s.add_argument("--results", required=True, help="results/ folder holding registry.csv")
        s.add_argument("--config", required=False)
        s.add_argument("--only", nargs="*", help="substrings of episode names to keep")
        s.add_argument("--notes", default="")
        if name == "frontend":
            s.add_argument("--oracle", action="store_true")
            s.add_argument("--checkpoints", help="folder holding the weights named in the config (relative paths)")
        if name in ("memory-eval", "vlm", "dam-curve"):
            s.add_argument("--frontend-run", required=True)
        if name == "vlm":
            s.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16])
            s.add_argument("--queries-config", required=True, help="memory_eval.yaml (its `queries` section)")
        if name == "backend":
            s.add_argument("--stage", choices=("h0", "all"), required=True, help="h0: first T7 episode; all: R5")
            s.add_argument("--ws", required=True, help="ROS workspace built by install_backend_full.sh")
            s.add_argument("--hydra-config", required=True)
            s.add_argument("--docs", required=True, help="docs/ folder for h0_gate.json")
            s.add_argument("--python", default="python3", help="interpreter DAAAM was installed into (VERIFY)")
    s = sub.add_parser("refcocog")
    s.add_argument("--ref-dir", required=True)
    s.add_argument("--results", required=True)
    s.add_argument("--notes", default="")
    s = sub.add_parser("prefetch", help="download the HF models into the HF cache")
    s.add_argument("--models", nargs="*", default=list(HF_MODELS))
    s = sub.add_parser("libero-download", help="download LIBERO-Mem files (HDF5, metainfo.json)")
    s.add_argument("--files", nargs="+", required=True)
    s.add_argument("--dest", required=True)
    s = sub.add_parser("camera", help="L2: agentview camera from the LIBERO env + table-plane gate")
    s.add_argument("--hdf5", required=True, help="task HDF5 (its data.attrs env_args build the env)")
    s.add_argument("--convert-config", required=True, help="configs/convert_libero.yaml (flip, depth mode)")
    s.add_argument("--out", required=True, help="docs/l2_camera_gate.json")
    s.add_argument("--demo-index", type=int, default=100, help="first val demo (paper's 100/20 split)")
    return p


def main(argv=None):
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from . import pipelines

    if args.cmd == "prefetch":
        from huggingface_hub import snapshot_download

        for m in args.models:
            print(snapshot_download(m))
        return
    if args.cmd == "libero-download":
        from .data.libero_mem import download

        for path in download(args.files, args.dest):
            print(path)
        return
    if args.cmd == "camera":
        import json

        from .data.convert import ConvertConfig
        from .data.evidence import camera_gate_report

        report = camera_gate_report(args.hdf5, load_config(ConvertConfig, args.convert_config), args.out, args.demo_index)
        print(json.dumps({k: v for k, v in report.items() if k != "camera"}, indent=2))
        return
    if args.cmd == "refcocog":
        print(Path(pipelines.run_refcocog(args.ref_dir, args.results, notes=args.notes)))
        return

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
    elif args.cmd == "vlm":
        from .baselines.vlm import VLMConfig
        from .queries.generate import QueryConfig

        vlm_cfg = load_config(VLMConfig, args.config) if args.config else VLMConfig()
        q_cfg = load_config(QueryConfig, args.queries_config, section="queries")
        out = pipelines.run_vlm_baselines(eps, args.frontend_run, vlm_cfg, q_cfg, args.ks, args.results,
                                          args.config or "", notes=args.notes)
    elif args.cmd == "dam-curve":
        out = pipelines.run_dam_curve(eps, args.frontend_run, args.results, notes=args.notes)
    else:
        from .backend_full import run as backend

        if args.stage == "h0":
            ep = next((e for e in eps if e.name.startswith("T7")), eps[0])
            print(backend.run_h0(args.ws, ep, args.hydra_config, args.results, args.docs, args.config, args.python))
            return
        out = backend.run_all(args.ws, eps, args.hydra_config, args.results, args.config, args.python)
    print(Path(out))


if __name__ == "__main__":
    main()
