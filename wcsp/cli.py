# python -m wcsp <command> --config configs/<name>.yaml [options]
#
#   predict    --split dev --shard 3/20     classifier -> probability cache
#   calibrate  --split dev                  fit temperature on cached dev predictions
#   solve      --split dev --shard 3/20 --run NAME [--set problem.lam=2 ...]
#   summarize  --run NAME                   merge shards -> summary.csv
#   compare    --run B --against A          ROI bootstrap of run B vs run A (same split)
#   elicit     --split fit                  ground-truth pair statistics -> cost table CSV
#   show       --run NAME --id SAMPLE       comparison figure for one sample

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import pipeline
from .config import load_config
from .data import parse_shard, select


def _args(argv):
    parser = argparse.ArgumentParser(prog="python -m wcsp")
    parser.add_argument("command", choices=["predict", "calibrate", "solve", "summarize", "compare", "elicit", "show"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--split")
    parser.add_argument("--shard", help="i/n, 0-based; defaults to $SLURM_ARRAY_TASK_ID/--shards")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--run", help="run name under paths.runs (default: config name)")
    parser.add_argument("--against", help="baseline run name for `compare`")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. problem.lam=2")
    parser.add_argument("--limit", type=int, default=0,
                        help="use N samples spread over the split (0 = all), before sharding")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--id", help="sample id for `show`")
    parser.add_argument("--out", help="output path for `elicit` / `show`")
    return parser.parse_args(argv)


def _shard(args):
    if args.shard:
        return parse_shard(args.shard)
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    return (int(task), args.shards) if task is not None else (0, args.shards)


def main(argv=None):
    args = _args(argv if argv is not None else sys.argv[1:])
    cfg = load_config(args.config, args.set)
    split = args.split or cfg.dataset["split"]
    run_dir = cfg.runs_root / (args.run or cfg.name) / split

    if args.command == "predict":
        pipeline.predict(cfg, split, _shard(args), args.batch_size, args.limit)

    elif args.command == "calibrate":
        best, curve = pipeline.calibrate(cfg, split)
        print(f"temperature = {best:.3f} (NLL {curve[best]:.4f}; at T=1: "
              f"{min(curve.items(), key=lambda kv: abs(kv[0] - 1))[1]:.4f})")

    elif args.command == "solve":
        pipeline.solve(cfg, split, _shard(args), run_dir, args.limit)

    elif args.command == "summarize":
        summary = pipeline.summarize(run_dir)
        with_cols = [c for c in summary.columns if not c.startswith("Dice_")]
        print(summary[with_cols].round(4).to_string(index=False))

    elif args.command == "compare":
        if not args.run or not args.against:
            raise SystemExit("compare needs --run and --against.")
        result = pipeline.compare(cfg.runs_root / args.against / split, run_dir)
        out = run_dir / f"compare_vs_{args.against}.json"
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))

    elif args.command == "elicit":
        from .evaluate.elicit import counts_by_type, fill_background, pmi_table, unlike_rate_by_type
        from .labels import IGNORE
        from .model.tables import save_table

        data = pipeline.dataset_for(cfg, split)
        positions = select(len(data), limit=args.limit)

        def masks():
            for i in positions:
                sample = data[i]
                target = sample.grid_target
                if cfg.dataset["name"] == "pannuke":   # nuclei -> cell territories
                    target = fill_background(target, target != IGNORE)
                yield target

        counts = counts_by_type(masks())
        table = pmi_table(sum(counts.values()))
        out = Path(args.out or f"configs/tables/{cfg.dataset['name']}_elicited.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        save_table(table, out)
        stats = {
            "unlike_rate_by_type": unlike_rate_by_type(counts),
            "counts_by_type": {t: c.tolist() for t, c in counts.items()},
        }
        out.with_suffix(".json").write_text(json.dumps(stats, indent=2))
        print(f"wrote {out}")
        print(np.round(table, 2))
        print("unlike-pair rate by neighbor type:", stats["unlike_rate_by_type"])

    elif args.command == "show":
        from .viz import comparison_figure

        data = pipeline.dataset_for(cfg, split)
        sample = data[data.index(args.id)]
        for path in sorted(run_dir.glob("shard_*_of_*.npz")):
            z = np.load(path)
            ids = z["ids"].tolist()
            if args.id in ids:
                row = ids.index(args.id)
                out = args.out or str(run_dir / f"{args.id}.png")
                comparison_figure(sample.grid_image, sample.grid_target, z["argmax"][row],
                                  z["wcsp"][row], title=args.id, path=out)
                print(f"wrote {out}")
                return
        raise SystemExit(f"{args.id} not found in {run_dir}")


if __name__ == "__main__":
    main()
