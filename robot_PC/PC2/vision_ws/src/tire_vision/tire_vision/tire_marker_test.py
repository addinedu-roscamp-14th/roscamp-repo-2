#!/usr/bin/env python3
"""Standalone UI using the production tire measurement and decision state."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import yaml

try:
    from tire_vision.marker_detector import detect_yellow_marker
    from tire_vision.wear_measurement import (
        GeometryRecheckError,
        MarkerNotFoundError,
        MeasurementError,
        TireDecisionState,
        TireNotFoundError,
        decision_state_args_from_config,
        estimate_tire_geometry_for_side,
        extract_tire_detection,
        measure_radial_gap,
    )
except ImportError:  # Direct source-file execution.
    from marker_detector import detect_yellow_marker
    from wear_measurement import (
        GeometryRecheckError,
        MarkerNotFoundError,
        MeasurementError,
        TireDecisionState,
        TireNotFoundError,
        decision_state_args_from_config,
        estimate_tire_geometry_for_side,
        extract_tire_detection,
        measure_radial_gap,
    )

try:
    import cv2
except ImportError as exc:
    cv2, CV2_ERROR = None, exc
else:
    CV2_ERROR = None
try:
    import numpy as np
except ImportError as exc:
    np, NUMPY_ERROR = None, exc
else:
    NUMPY_ERROR = None

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "vision.yaml"
DEFAULT_CSV = "tire_marker_test.csv"
WINDOW_NAME = "Tire Marker Test"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm"}
COLORS = {
    "NO_TIRE": (120, 120, 120),
    "COLLECTING": (255, 180, 0),
    "GOOD": (0, 200, 0),
    "BAD": (0, 0, 255),
    "RECHECK": (0, 165, 255),
    "ERROR": (255, 0, 255),
}
CSV_FIELDS = [
    "timestamp", "frame_index", "source", "session_id", "confidence",
    "gap_px", "radius_px", "gap_ratio", "median_ratio", "std_ratio",
    "valid_count", "current_tire", "current_marker", "stable_result",
    "marker_miss_count", "no_tire_count", "result", "error_code", "detail",
]


def require_cv():
    if cv2 is None:
        raise RuntimeError(f"cv2 import failed: {CV2_ERROR}")
    if np is None:
        raise RuntimeError(f"numpy import failed: {NUMPY_ERROR}")


def parse_hsv(value):
    if isinstance(value, (list, tuple)):
        parts = tuple(int(item) for item in value)
    else:
        try:
            parts = tuple(int(item.strip()) for item in str(value).split(","))
        except ValueError as error:
            raise argparse.ArgumentTypeError("HSV must be H,S,V integers") from error
    if (len(parts) != 3 or not 0 <= parts[0] <= 179
            or not all(0 <= item <= 255 for item in parts[1:])):
        raise argparse.ArgumentTypeError("HSV range is H=0..179, S/V=0..255")
    return parts


def load_test_config(path):
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    model_path = Path(str(config.get("model_path", "models/tire_best.pt"))).expanduser()
    if not model_path.is_absolute():
        model_path = config_path.parent.parent / model_path
    config["model_path"] = str(model_path)
    config["config_path"] = str(config_path)
    return config

def select_camera_side(config, source=None, requested_side=None):
    """Resolve side explicitly or from a configured numeric camera source."""
    if requested_side in {"left", "right"}:
        return requested_side
    if source is not None and str(source).strip().isdigit():
        source_index = int(str(source).strip())
        for side in ("left", "right"):
            camera = config.get(f"{side}_camera", {}) or {}
            if source_index == int(camera.get("index", -1)):
                return side
    return "left"



def make_parser(config=None, default_side="left"):
    config = config or load_test_config(DEFAULT_CONFIG)
    camera = config.get(f"{default_side}_camera", {})
    decision_args = decision_state_args_from_config(config, default_side)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=config.get("config_path", str(DEFAULT_CONFIG)))
    parser.add_argument("--side", choices=("left", "right"), default=default_side)
    parser.add_argument("--model", default=config["model_path"])
    parser.add_argument("--source", default=str(camera.get("index", 2)))
    parser.add_argument("--imgsz", type=int, default=int(config.get("imgsz", 640)))
    parser.add_argument("--conf", type=float, default=float(config.get("conf_threshold", .5)))
    parser.add_argument("--device", default=config.get("device", 0))
    parser.add_argument("--camera-width", type=int, default=int(camera.get("width", 640)))
    parser.add_argument("--camera-height", type=int, default=int(camera.get("height", 480)))
    parser.add_argument("--camera-fps", type=float, default=float(camera.get("fps", 30)))
    parser.add_argument("--yellow-lower", type=parse_hsv,
                        default=parse_hsv(config.get("yellow_lower", [15, 40, 60])))
    parser.add_argument("--yellow-upper", type=parse_hsv,
                        default=parse_hsv(config.get("yellow_upper", [42, 255, 255])))
    parser.add_argument("--min-marker-area", type=float,
                        default=float(config.get("min_marker_area", 5)))
    parser.add_argument("--max-marker-area", type=float,
                        default=float(config.get("max_marker_area", 500)))
    parser.add_argument("--marker-roi-scale", type=float,
                        default=float(config.get("marker_roi_scale", 1.15)))
    parser.add_argument("--no-tire-reset-frames", type=int,
                        default=decision_args["no_tire_reset_frames"])
    parser.add_argument("--marker-hold-frames", type=int,
                        default=decision_args["marker_hold_frames"])
    parser.add_argument("--marker-recheck-frames", type=int,
                        default=decision_args["marker_recheck_frames"])
    parser.add_argument("--window-size", type=int,
                        default=decision_args["window_size"])
    parser.add_argument("--min-valid-frames", type=int,
                        default=decision_args["min_valid_frames"])
    parser.add_argument("--bad-max-ratio", type=float,
                        default=decision_args["bad_max_ratio"])
    parser.add_argument("--good-min-ratio", type=float,
                        default=decision_args["good_min_ratio"])
    parser.add_argument("--bad-max-gap-px", type=float,
                        default=decision_args["bad_max_gap_px"])
    parser.add_argument("--good-min-gap-px", type=float,
                        default=decision_args["good_min_gap_px"])
    parser.add_argument("--max-ratio-std", type=float,
                        default=decision_args["max_ratio_std"])
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--save-video")
    parser.add_argument("--no-display", action="store_true")
    return parser


def parse_args(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    pre_parser.add_argument("--side", choices=("left", "right"))
    pre_parser.add_argument("--source")
    known, _ = pre_parser.parse_known_args(argv)
    config = load_test_config(known.config)
    side = select_camera_side(config, known.source, known.side)
    return make_parser(config, side).parse_args(argv)


def validate_args(args):
    if args.window_size <= 0 or args.min_valid_frames <= 0:
        raise ValueError("window-size and min-valid-frames must be positive")
    if args.min_valid_frames > args.window_size:
        raise ValueError("min-valid-frames cannot exceed window-size")
    if args.bad_max_ratio >= args.good_min_ratio:
        raise ValueError("bad-max-ratio must be less than good-min-ratio")
    if (args.bad_max_gap_px is None) != (args.good_min_gap_px is None):
        raise ValueError("gap thresholds must be configured together")
    if (
        args.bad_max_gap_px is not None
        and args.bad_max_gap_px >= args.good_min_gap_px
    ):
        raise ValueError("bad-max-gap-px must be less than good-min-gap-px")

    if args.min_marker_area > args.max_marker_area:
        raise ValueError("min-marker-area cannot exceed max-marker-area")
    if args.marker_roi_scale < 1.0:
        raise ValueError("marker-roi-scale must be at least 1.0")
    if args.no_tire_reset_frames <= 0:
        raise ValueError("no-tire-reset-frames must be positive")
    if args.marker_hold_frames < 0 or args.marker_recheck_frames < args.marker_hold_frames:
        raise ValueError("marker miss thresholds must satisfy 0 <= hold <= recheck")


def create_decision_state(args):
    return TireDecisionState(
        window_size=args.window_size,
        min_valid_frames=args.min_valid_frames,
        bad_max_ratio=args.bad_max_ratio,
        good_min_ratio=args.good_min_ratio,
        bad_max_gap_px=args.bad_max_gap_px,
        good_min_gap_px=args.good_min_gap_px,
        max_ratio_std=args.max_ratio_std,
        no_tire_reset_frames=args.no_tire_reset_frames,
        marker_hold_frames=args.marker_hold_frames,
        marker_recheck_frames=args.marker_recheck_frames,
    )


def load_model(path):
    model_path = Path(path).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"model file does not exist: {model_path}")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(f"ultralytics import failed: {error}") from error
    return YOLO(str(model_path))


def analyze_frame(frame, model, args, state):
    output = state.snapshot()
    output.update({"confidence": None})
    try:
        try:
            results = model.predict(
                source=frame, imgsz=args.imgsz, conf=args.conf,
                device=args.device, verbose=False,
            )
        except Exception as error:
            state.update_error(f"model inference failed: {error}", "MODEL_ERROR")
            output.update(state.snapshot())
            return output
        if not results:
            raise TireNotFoundError("tire not detected")
        detection = extract_tire_detection(results[0], frame.shape, args.conf)
        state.update_tire_detection(True)
        output.update(detection)
        output["confidence"] = detection["confidence"]
        geometry = estimate_tire_geometry_for_side(detection["mask"], args.side)
        geometry_mask = geometry.get("mask", detection["mask"])
        if "mask" in geometry:
            output["mask"] = geometry_mask
        output["geometry"] = geometry
        marker = detect_yellow_marker(
            frame, geometry_mask, geometry["center"], geometry["radius"],
            args.yellow_lower, args.yellow_upper, args.min_marker_area,
            args.max_marker_area, args.marker_roi_scale,
        )
        output["marker"] = marker
        measurement = measure_radial_gap(
            geometry["center"], geometry["radius"],
            marker["contour"], marker["center"],
        )
        output.update(measurement)
        state.update_marker_detection(
            True,
            ratio=measurement["gap_ratio"],
            gap_px=measurement["gap_px"],
            radius_px=measurement["radius_px"],
        )
    except TireNotFoundError as error:
        state.update_tire_detection(False, str(error))
    except GeometryRecheckError as error:
        state.update_recheck(str(error))
    except MarkerNotFoundError as error:
        state.update_marker_detection(False, detail=str(error))
    except MeasurementError as error:
        state.update_error(str(error))
    output.update(state.snapshot())
    return output


def _point(value):
    return tuple(int(round(item)) for item in value)


def draw_debug(frame, data, args):
    debug = frame.copy()
    if "mask" in data:
        blue = np.zeros_like(debug)
        blue[:, :, 0] = data["mask"]
        debug = cv2.addWeighted(debug, 1.0, blue, .35, 0)
        box = tuple(int(round(item)) for item in data["bbox"])
        cv2.rectangle(debug, box[:2], box[2:], (255, 180, 0), 2)
    if "geometry" in data:
        center = _point(data["geometry"]["center"])
        cv2.circle(debug, center, int(round(data["geometry"]["radius"])), (255, 255, 0), 2)
        cv2.circle(debug, center, 4, (255, 255, 255), -1)
        if data["geometry"].get("method") == "bounding_rect":
            bbox = data["geometry"]["bbox"]
            cv2.rectangle(
                debug, tuple(bbox[:2]), tuple(bbox[2:]), (255, 255, 0), 2
            )
    if "marker" in data:
        cv2.drawContours(debug, [data["marker"]["contour"]], -1, (0, 255, 255), 2)
        cv2.circle(debug, _point(data["marker"]["center"]), 4, (0, 255, 255), -1)
    if data.get("marker_outer_point") is not None:
        marker_outer = _point(data["marker_outer_point"])
        tire_outer = _point(data["tire_outer_point"])
        cv2.circle(debug, marker_outer, 5, (0, 255, 255), -1)
        cv2.circle(debug, tire_outer, 5, (255, 255, 255), -1)
        cv2.line(debug, marker_outer, tire_outer, (0, 255, 0), 2)

    def fmt(value, digits):
        return "N/A" if value is None else f"{value:.{digits}f}"

    geometry = data.get("geometry", {})
    geometry_lines = []
    if geometry.get("method") == "bounding_rect":
        geometry_lines = [
            f"BBOX WIDTH: {geometry['width']} px",
            f"BBOX HEIGHT: {geometry['height']} px",
            f"BBOX RADIUS: {geometry['radius']:.1f} px",
        ]

    gap_threshold_lines = []
    if args.bad_max_gap_px is not None:
        gap_threshold_lines = [
            f"MEDIAN GAP: {fmt(data.get('median_gap_px'), 1)} px",
            f"BAD GAP <= {args.bad_max_gap_px:.1f} px",
            f"GOOD GAP >= {args.good_min_gap_px:.1f} px",
        ]

    lines = [
        f"CAMERA: {args.side.upper()}",
        f"SOURCE: {args.source}",
        *geometry_lines,
        f"CONF: {fmt(data.get('confidence'), 2)}",
        f"GAP: {fmt(data.get('gap_px'), 1)} px",
        f"RADIUS: {fmt(data.get('radius_px'), 1)} px",
        f"RATIO: {fmt(data.get('gap_ratio'), 3)}",
        f"MEDIAN: {fmt(data.get('median_ratio'), 3)}",
        f"STD: {fmt(data.get('std_ratio'), 3)}",
        f"VALID: {data.get('valid_count', 0)}/{data.get('window_size', 15)}",
        f"CURRENT TIRE: {data.get('current_tire', 'UNKNOWN')}",
        f"CURRENT MARKER: {data.get('current_marker', 'UNKNOWN')}",
        f"STABLE RESULT: {data.get('stable_result') or '-'}",
        f"MISS COUNT: {data.get('marker_miss_count', 0)}/{data.get('marker_hold_frames', 3)}",
        f"NO TIRE COUNT: {data.get('no_tire_count', 0)}/{data.get('no_tire_reset_frames', 3)}",
        f"BAD <= {args.bad_max_ratio:.3f}",
        f"GOOD >= {args.good_min_ratio:.3f}",
        *gap_threshold_lines,
        f"MAX STD: {args.max_ratio_std:.3f}",
        f"RESULT: {data.get('result', 'ERROR')}",
        f"DETAIL: {data.get('detail') or '-'}",
    ]
    result_color = COLORS.get(data.get("result"), (255, 255, 255))
    line_spacing = 18 if gap_threshold_lines else (20 if geometry_lines else 23)
    font_scale = .48 if gap_threshold_lines else (.52 if geometry_lines else .58)
    y_start = 24 if geometry_lines else 26
    for index, line in enumerate(lines):
        color = result_color if line.startswith(("RESULT", "DETAIL")) else (255, 255, 255)
        cv2.putText(
            debug, line, (12, y_start + line_spacing * index),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2,
        )
    return debug


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def classify_source(source):
    if source.isdigit():
        return "camera", int(source)
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"source file does not exist: {path}")
    if path.suffix.lower() in IMAGE_EXTS:
        return "image", path
    if path.suffix.lower() in VIDEO_EXTS:
        return "video", path
    raise ValueError(f"unsupported source file extension: {path.suffix or '(none)'}")


def open_capture(kind, value, args):
    if kind == "image":
        return None
    backend = cv2.CAP_V4L2 if kind == "camera" else cv2.CAP_ANY
    capture = cv2.VideoCapture(value, backend)
    if not capture.isOpened():
        raise RuntimeError(f"capture open failed: {value}")
    if kind == "camera":
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def save_debug_frame(frame, index):
    directory = Path.cwd() / "debug_frames"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"frame_{index:06d}_{stamp}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"debug image save failed: {path}")
    return path


def _reset_display(frame, state, args):
    reset_data = state.reset_session()
    reset_data["confidence"] = None
    debug = draw_debug(frame, reset_data, args)
    cv2.imshow(WINDOW_NAME, debug)
    cv2.waitKey(1)
    print("[RESET] detection session cleared", flush=True)


def run(args):
    require_cv()
    validate_args(args)
    kind, value = classify_source(str(args.source))
    model = load_model(args.model)
    state = create_decision_state(args)
    capture = writer = None
    try:
        capture = open_capture(kind, value, args)
        image = cv2.imread(str(value)) if kind == "image" else None
        if kind == "image" and image is None:
            raise RuntimeError(f"image load failed: {value}")
        frame_index = 0
        while True:
            if kind == "image":
                if frame_index:
                    break
                frame = image.copy()
            else:
                ok, frame = capture.read()
                if not ok:
                    if kind == "video":
                        break
                    raise RuntimeError(f"camera frame read failed: index {value}")
            data = analyze_frame(frame, model, args, state)
            debug = draw_debug(frame, data, args)
            row = {key: data.get(key) for key in CSV_FIELDS}
            row.update({
                "timestamp": datetime.now().astimezone().isoformat(),
                "frame_index": frame_index,
                "source": str(args.source),
            })
            append_csv(Path(args.csv).expanduser(), row)
            if args.save_video:
                if writer is None:
                    path = Path(args.save_video).expanduser()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fps = capture.get(cv2.CAP_PROP_FPS) if capture else args.camera_fps
                    fps = fps if fps > 0 else args.camera_fps
                    writer = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (debug.shape[1], debug.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"debug video open failed: {path}")
                writer.write(debug)
            if not args.no_display:
                cv2.imshow(WINDOW_NAME, debug)
                key = cv2.waitKey(0 if kind == "image" else 1) & 0xff
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("r"), ord("R")):
                    _reset_display(frame, state, args)
                elif key in (ord("s"), ord("S")):
                    print(f"debug frame saved: {save_debug_frame(debug, frame_index)}")
            frame_index += 1
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if cv2 is not None:
            cv2.destroyAllWindows()


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
