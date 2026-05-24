from __future__ import annotations

import argparse

import yaml

from gotennet_other.train import TrainerConfig, evaluate, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["train", "eval", "train_eval"], default="train")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--eval-split", default="test", choices=["train", "val", "test", "all"])
    args = parser.parse_args()

    raw = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    raw.setdefault("dataset_name", "sn2rxn")
    cfg = TrainerConfig(**raw)

    if args.mode == "train":
        print(train(cfg, resume=args.resume))
        return

    if args.mode == "eval":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for --mode eval")
        print(evaluate(cfg, args.checkpoint, args.eval_split))
        return

    train_metrics = train(cfg, resume=args.resume)
    checkpoint = args.checkpoint or f"{cfg.output_dir}/checkpoints/best.pt"
    eval_metrics = evaluate(cfg, checkpoint, args.eval_split)
    print({"train": train_metrics, "eval": eval_metrics})


if __name__ == "__main__":
    main()
