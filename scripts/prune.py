from __future__ import annotations

import argparse
import yaml

from pruning.pipeline import PruningConfig, run_pruning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SparseGPT CNN 프루닝 실행 스크립트")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="프루닝 하이퍼파라미터가 담긴 YAML 파일 경로",
    )
    return parser.parse_args()


def load_config(path: str) -> PruningConfig:
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    return PruningConfig(**data)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_pruning(config)


if __name__ == "__main__":
    main()

