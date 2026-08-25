#!/usr/bin/env python3
"""Standalone YOLO tire-segmentation and yellow-marker gap test."""
from __future__ import annotations

import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path

try:
    from tire_vision.marker_detector import (
        build_marker_search_mask as _shared_build_marker_search_mask,
        detect_yellow_marker as _shared_detect_yellow_marker,
    )
    from tire_vision.wear_measurement import (
        MeasurementError as _SharedMeasurementError,
        classify_history as _shared_classify_history,
        estimate_tire_geometry as _shared_estimate_tire_geometry,
        extract_tire_detection as _shared_extract_tire_detection,
        measure_radial_gap as _shared_measure_radial_gap,
    )
except ImportError:
    from marker_detector import (
        build_marker_search_mask as _shared_build_marker_search_mask,
        detect_yellow_marker as _shared_detect_yellow_marker,
    )
    from wear_measurement import (
        MeasurementError as _SharedMeasurementError,
        classify_history as _shared_classify_history,
        estimate_tire_geometry as _shared_estimate_tire_geometry,
        extract_tire_detection as _shared_extract_tire_detection,
        measure_radial_gap as _shared_measure_radial_gap,
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

DEFAULT_MODEL = "/home/seohyun/robot_PC/vision_ws/models/tire_best.pt"
DEFAULT_CSV = "~/robot_PC/vision_ws/tire_marker_test.csv"
WINDOW_NAME = "Tire Marker Test"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm"}
COLORS = {"GOOD": (0, 200, 0), "BAD": (0, 0, 255),
          "RECHECK": (0, 165, 255), "ERROR": (255, 0, 255)}


class MeasurementError(RuntimeError):
    """An expected per-frame measurement failure."""


def require_cv():
    if cv2 is None:
        raise RuntimeError(f"cv2 import failed: {CV2_ERROR}")
    if np is None:
        raise RuntimeError(f"numpy import failed: {NUMPY_ERROR}")


def parse_hsv(value):
    try:
        parts = tuple(int(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("HSV must be H,S,V integers") from exc
    if (len(parts) != 3 or not 0 <= parts[0] <= 179
            or not all(0 <= x <= 255 for x in parts[1:])):
        raise argparse.ArgumentTypeError("HSV range is H=0..179, S/V=0..255")
    return parts


def _scalar(value):
    if hasattr(value, "item"):
        return float(value.item())
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def extract_tire_detection(result, frame_shape, conf_threshold):
    """Return the best tire box, confidence, and original-size binary mask."""
    require_cv()
    if result.boxes is None or len(result.boxes) == 0:
        raise MeasurementError("tire not detected")
    choices = []
    for index, box in enumerate(result.boxes):
        class_id = int(_scalar(box.cls))
        names = result.names
        name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        confidence = _scalar(box.conf)
        if name == "tire" and confidence >= conf_threshold:
            choices.append((confidence, index, box))
    if not choices:
        raise MeasurementError("tire not detected")
    if result.masks is None or getattr(result.masks, "data", None) is None:
        raise MeasurementError("result.masks is missing")
    confidence, index, box = max(choices, key=lambda x: x[0])
    if index >= len(result.masks.data):
        raise MeasurementError("mask index does not match detected tire")
    mask = result.masks.data[index]
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    mask = np.asarray(mask, dtype=np.float32).squeeze()
    if mask.ndim != 2:
        raise MeasurementError("invalid tire mask dimensions")
    height, width = frame_shape[:2]
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 0.5).astype(np.uint8) * 255
    xyxy = box.xyxy
    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()
    bbox = tuple(float(x) for x in np.asarray(xyxy).reshape(-1)[:4])
    return {"confidence": confidence, "bbox": bbox, "mask": binary}


def estimate_tire_geometry(tire_mask, min_contour_area=100.0, min_radius=5.0):
    """Estimate geometry from the largest external contour (replaceable later)."""
    require_cv()
    mask = np.asarray(tire_mask)
    if mask.ndim != 2 or mask.size == 0:
        raise MeasurementError("invalid tire mask")
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise MeasurementError("tire contour not found")
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_contour_area:
        raise MeasurementError(f"tire contour is too small ({area:.1f} px^2)")
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    if not np.isfinite([cx, cy, radius]).all() or radius < min_radius:
        raise MeasurementError(f"tire radius is too small ({radius:.1f} px)")
    return {"center": (float(cx), float(cy)), "radius": float(radius),
            "contour": contour, "area": area}


def build_marker_search_mask(tire_mask, center, radius):
    require_cv()
    mask = np.asarray(tire_mask)
    if mask.ndim != 2 or radius <= 0:
        raise MeasurementError("invalid tire geometry for marker search")
    height, width = mask.shape
    cx, cy = center
    x1, x2 = max(0, int(cx - .8 * radius)), min(width, int(np.ceil(cx + .8 * radius)))
    y1, y2 = max(0, int(cy - radius)), min(height, int(np.ceil(cy - .1 * radius)))
    search = np.zeros_like(mask, dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        search[y1:y2, x1:x2] = 255
    return cv2.bitwise_and(search, (mask > 0).astype(np.uint8) * 255)


def detect_yellow_marker(frame, tire_mask, center, radius,
                         yellow_lower=(15, 80, 80), yellow_upper=(40, 255, 255),
                         min_marker_area=5, max_marker_area=500):
    require_cv()
    if frame is None or np.asarray(frame).ndim != 3:
        raise MeasurementError("invalid frame for marker detection")
    search = build_marker_search_mask(tire_mask, center, radius)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array(yellow_lower, np.uint8),
                         np.array(yellow_upper, np.uint8))
    yellow = cv2.bitwise_and(yellow, search)
    kernel = np.ones((3, 3), np.uint8)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    choices = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        moments = cv2.moments(contour)
        if min_marker_area <= area <= max_marker_area and moments["m00"] > 0:
            marker_center = (moments["m10"] / moments["m00"],
                             moments["m01"] / moments["m00"])
            if marker_center[1] < center[1]:
                choices.append((area, contour, marker_center))
    if not choices:
        raise MeasurementError("yellow marker not found")
    area, contour, marker_center = max(choices, key=lambda x: x[0])
    return {"contour": contour, "center": marker_center, "area": area,
            "binary_mask": yellow, "search_mask": search}


def measure_radial_gap(center, radius, marker_contour, marker_center=None):
    require_cv()
    if radius <= 0 or not np.isfinite(radius):
        raise MeasurementError("invalid tire radius")
    points = np.asarray(marker_contour, np.float64).reshape(-1, 2)
    if not len(points):
        raise MeasurementError("marker contour is empty")
    c = np.asarray(center, np.float64)
    if marker_center is None:
        moments = cv2.moments(points.astype(np.float32).reshape(-1, 1, 2))
        if moments["m00"] <= 0:
            raise MeasurementError("marker center cannot be calculated")
        marker_center = (moments["m10"] / moments["m00"],
                         moments["m01"] / moments["m00"])
    m = np.asarray(marker_center, np.float64)
    direction = m - c
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        raise MeasurementError("marker center overlaps tire center")
    distances = np.linalg.norm(points - c, axis=1)
    marker_outer = points[int(np.argmax(distances))]
    if float(np.max(distances)) > radius + 1e-3:
        raise MeasurementError("marker outer boundary is outside tire radius")
    tire_outer = c + radius * direction / length
    gap_px = float(np.linalg.norm(tire_outer - marker_outer))
    if not np.isfinite(gap_px) or gap_px < 0 or gap_px > radius:
        raise MeasurementError("radial gap is outside the valid range")
    return {"gap_px": gap_px, "radius_px": float(radius),
            "gap_ratio": gap_px / float(radius), "marker_center": m.tolist(),
            "marker_outer_point": marker_outer.tolist(),
            "tire_outer_point": tire_outer.tolist()}


def classify_history(history, min_valid_frames, bad_max_ratio,
                     good_min_ratio, max_ratio_std):
    require_cv()
    values = np.asarray(list(history), np.float64)
    median = float(np.median(values)) if values.size else None
    std = float(np.std(values)) if values.size else None
    result = "RECHECK"
    if values.size >= min_valid_frames and std <= max_ratio_std:
        if median >= good_min_ratio:
            result = "GOOD"
        elif median <= bad_max_ratio:
            result = "BAD"
    return {"result": result, "median_ratio": median, "std_ratio": std}


# Keep the standalone script API compatible while executing the shared modules.
MeasurementError = _SharedMeasurementError
extract_tire_detection = _shared_extract_tire_detection
estimate_tire_geometry = _shared_estimate_tire_geometry
build_marker_search_mask = _shared_build_marker_search_mask
detect_yellow_marker = _shared_detect_yellow_marker
measure_radial_gap = _shared_measure_radial_gap
classify_history = _shared_classify_history


def analyze_frame(frame, model, args, history):
    output = {"confidence": None, "gap_px": None, "radius_px": None,
              "gap_ratio": None, "median_ratio": None, "std_ratio": None,
              "result": "ERROR", "error_reason": ""}
    try:
        results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf,
                                device=args.device, verbose=False)
        if not results:
            raise MeasurementError("YOLO returned no result")
        detection = extract_tire_detection(results[0], frame.shape, args.conf)
        output.update(detection)
        geometry = estimate_tire_geometry(detection["mask"])
        output["geometry"] = geometry
        marker = detect_yellow_marker(
            frame, detection["mask"], geometry["center"], geometry["radius"],
            args.yellow_lower, args.yellow_upper, args.min_marker_area,
            args.max_marker_area)
        output["marker"] = marker
        measurement = measure_radial_gap(
            geometry["center"], geometry["radius"], marker["contour"], marker["center"])
        history.append(measurement["gap_ratio"])
        summary = classify_history(
            history, args.min_valid_frames, args.bad_max_ratio,
            args.good_min_ratio, args.max_ratio_std)
        output.update(measurement)
        output.update(summary)
    except MeasurementError as exc:
        output["error_reason"] = str(exc)
        summary = classify_history(
            history, args.min_valid_frames, args.bad_max_ratio,
            args.good_min_ratio, args.max_ratio_std)
        output["median_ratio"], output["std_ratio"] = (
            summary["median_ratio"], summary["std_ratio"])
    return output


def _point(value):
    return tuple(int(round(x)) for x in value)


def draw_debug(frame, data, valid_count, window_size,
               bad_max_ratio, good_min_ratio):
    debug = frame.copy()
    if "mask" in data:
        blue = np.zeros_like(debug)
        blue[:, :, 0] = data["mask"]
        debug = cv2.addWeighted(debug, 1.0, blue, .35, 0)
        box = tuple(int(round(x)) for x in data["bbox"])
        cv2.rectangle(debug, box[:2], box[2:], (255, 180, 0), 2)
    if "geometry" in data:
        center = _point(data["geometry"]["center"])
        cv2.circle(debug, center, int(round(data["geometry"]["radius"])), (255, 255, 0), 2)
        cv2.circle(debug, center, 4, (255, 255, 255), -1)
    if "marker" in data:
        cv2.drawContours(debug, [data["marker"]["contour"]], -1, (0, 255, 255), 2)
        cv2.circle(debug, _point(data["marker"]["center"]), 4, (0, 255, 255), -1)
    if data.get("marker_outer_point") is not None:
        marker_outer, tire_outer = _point(data["marker_outer_point"]), _point(data["tire_outer_point"])
        cv2.circle(debug, marker_outer, 5, (0, 255, 255), -1)
        cv2.circle(debug, tire_outer, 5, (255, 255, 255), -1)
        cv2.line(debug, marker_outer, tire_outer, (0, 255, 0), 2)

    def fmt(value, digits):
        return "N/A" if value is None else f"{value:.{digits}f}"
    lines = [f"CONF: {fmt(data.get('confidence'), 2)}",
             f"GAP: {fmt(data.get('gap_px'), 1)} px",
             f"RADIUS: {fmt(data.get('radius_px'), 1)} px",
             f"RATIO: {fmt(data.get('gap_ratio'), 3)}",
             f"MEDIAN: {fmt(data.get('median_ratio'), 3)}",
             f"STD: {fmt(data.get('std_ratio'), 3)}",
             f"VALID: {valid_count}/{window_size}",
             f"BAD <= {bad_max_ratio:.3f}",
             f"GOOD >= {good_min_ratio:.3f}",
             f"RESULT: {data['result']}"]
    if data.get("error_reason"):
        lines.append(f"ERROR: {data['error_reason']}")
    for index, line in enumerate(lines):
        color = COLORS[data["result"]] if line.startswith(("RESULT", "ERROR")) else (255, 255, 255)
        cv2.putText(debug, line, (12, 28 + 25 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
    return debug


CSV_FIELDS = ["timestamp", "frame_index", "source", "confidence", "gap_px",
              "radius_px", "gap_ratio", "median_ratio", "std_ratio",
              "valid_count", "result", "error_reason"]


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
    if kind == "camera":
        capture = cv2.VideoCapture(value, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        message = f"camera open failed: index {value}"
    elif kind == "video":
        capture, message = cv2.VideoCapture(str(value)), f"video open failed: {value}"
    else:
        return None
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(message)
    return capture


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source", default="2")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30)
    parser.add_argument("--yellow-lower", type=parse_hsv, default=parse_hsv("15,80,80"))
    parser.add_argument("--yellow-upper", type=parse_hsv, default=parse_hsv("40,255,255"))
    parser.add_argument("--min-marker-area", type=float, default=5)
    parser.add_argument("--max-marker-area", type=float, default=500)
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--min-valid-frames", type=int, default=10)
    parser.add_argument("--bad-max-ratio", type=float, default=.23)
    parser.add_argument("--good-min-ratio", type=float, default=.30)
    parser.add_argument("--max-ratio-std", type=float, default=.01)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--save-video")
    parser.add_argument("--no-display", action="store_true")
    return parser


def validate_args(args):
    if args.window_size <= 0 or args.min_valid_frames <= 0:
        raise ValueError("window-size and min-valid-frames must be positive")
    if args.min_valid_frames > args.window_size:
        raise ValueError("min-valid-frames cannot exceed window-size")
    if args.bad_max_ratio >= args.good_min_ratio:
        raise ValueError("bad-max-ratio must be less than good-min-ratio")
    if args.min_marker_area > args.max_marker_area:
        raise ValueError("min-marker-area cannot exceed max-marker-area")


def load_model(path):
    if not path.is_file():
        raise FileNotFoundError(f"model file does not exist: {path}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(f"ultralytics import failed: {exc}") from exc
    return YOLO(str(path))


def save_debug_frame(frame, index):
    directory = Path("~/robot_PC/vision_ws/debug_frames").expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"frame_{index:06d}_{stamp}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"debug image save failed: {path}")
    return path


def run(args):
    require_cv()
    validate_args(args)
    kind, value = classify_source(str(args.source))
    model = load_model(Path(args.model).expanduser())
    capture = writer = None
    try:
        capture = open_capture(kind, value, args)
        image = cv2.imread(str(value)) if kind == "image" else None
        if kind == "image" and image is None:
            raise RuntimeError(f"image load failed: {value}")
        history = deque(maxlen=args.window_size)
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
            data = analyze_frame(frame, model, args, history)
            debug = draw_debug(
                frame, data, len(history), args.window_size,
                args.bad_max_ratio, args.good_min_ratio)
            row = {key: data.get(key) for key in CSV_FIELDS}
            row.update({"timestamp": datetime.now().astimezone().isoformat(),
                        "frame_index": frame_index, "source": str(args.source),
                        "valid_count": len(history)})
            append_csv(Path(args.csv).expanduser(), row)
            if args.save_video:
                if writer is None:
                    path = Path(args.save_video).expanduser()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fps = capture.get(cv2.CAP_PROP_FPS) if capture else args.camera_fps
                    fps = fps if fps > 0 else args.camera_fps
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (debug.shape[1], debug.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"debug video open failed: {path}")
                writer.write(debug)
            if not args.no_display:
                cv2.imshow(WINDOW_NAME, debug)
                key = cv2.waitKey(0 if kind == "image" else 1) & 0xff
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    history.clear()
                    print("history reset")
                elif key == ord("s"):
                    print(f"debug frame saved: {save_debug_frame(debug, frame_index)}")
            frame_index += 1
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if cv2 is not None:
            cv2.destroyAllWindows()


def main():
    args = make_parser().parse_args()
    try:
        run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
