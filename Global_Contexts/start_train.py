from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def str_to_bool(value: str | bool) -> bool:
    """Parse command-line boolean values such as True/False and yes/no."""
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()

    if normalized in {"true", "1", "yes", "y", "on"}:
        return True

    if normalized in {"false", "0", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Expected a boolean value, but received: {value!r}"
    )


def existing_file(value: str) -> str:
    """Validate and return an absolute file path."""
    path = Path(value).expanduser()

    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")

    return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a custom YOLOv8 fire and smoke detector."
    )

    # Required inputs
    parser.add_argument(
        "--model",
        type=existing_file,
        required=True,
        help="Path to the YOLO model YAML configuration.",
    )
    parser.add_argument(
        "--data_dir",
        "--data-dir",
        dest="data_dir",
        type=existing_file,
        required=True,
        help="Path to the dataset YAML file.",
    )

    # Training configuration
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device such as 0 or 0,1; use cpu for CPU training.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="./runs/train",
        help="Parent directory for training outputs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name. Defaults to the model YAML filename.",
    )

    # Initialization
    parser.add_argument(
        "--pretrained",
        type=str_to_bool,
        default=False,
        help="Load pretrained weights before training.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help=(
            "Weights loaded when --pretrained True. This may be a local "
            "checkpoint or an Ultralytics model such as yolov8n.pt."
        ),
    )

    # Reproducibility and optimization
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument(
        "--weight_decay",
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=0.0005,
    )
    parser.add_argument(
        "--warmup_epochs",
        "--warmup-epochs",
        dest="warmup_epochs",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--close_mosaic",
        "--close-mosaic",
        dest="close_mosaic",
        type=int,
        default=10,
    )

    # Runtime options
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--amp", type=str_to_bool, default=True)
    parser.add_argument("--deterministic", type=str_to_bool, default=True)
    parser.add_argument("--exist_ok", type=str_to_bool, default=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_name = args.name or Path(args.model).stem
    project_dir = str(Path(args.project).expanduser())

    print(f"Model configuration: {args.model}")
    print(f"Dataset configuration: {args.data_dir}")
    print(f"Output run: {Path(project_dir) / run_name}")

    model = YOLO(args.model)

    if args.pretrained:
        print(f"Loading pretrained weights: {args.weights}")
        model.load(args.weights)
    else:
        print("Training from scratch: no pretrained weights loaded.")

    model.train(
        data=args.data_dir,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=project_dir,
        name=run_name,
        pretrained=args.pretrained,
        seed=args.seed,
        deterministic=args.deterministic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        close_mosaic=args.close_mosaic,
        workers=args.workers,
        patience=args.patience,
        amp=args.amp,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()