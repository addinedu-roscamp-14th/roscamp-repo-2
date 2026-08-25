import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


DEFAULT_DATASET_DIR = Path.home() / "handeye_dataset"
MIN_SAMPLES = 10


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def average_rotation(rotations: List[np.ndarray]) -> np.ndarray:
    M = np.mean(np.stack(rotations, axis=0), axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_delta = R_a.T @ R_b
    cos_angle = (np.trace(R_delta) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def load_samples(dataset_dir: Path) -> List[Dict]:
    samples: List[Dict] = []

    for path in sorted(dataset_dir.glob("sample_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["_path"] = str(path)
            data["_name"] = path.name
            samples.append(data)
        except Exception as exc:
            print(f"[경고] {path.name} 읽기 실패: {exc}")

    return samples


def calibrate_park(samples: List[Dict]) -> np.ndarray:
    if len(samples) < MIN_SAMPLES:
        raise RuntimeError(
            f"샘플이 {len(samples)}개입니다. 최소 {MIN_SAMPLES}개가 필요합니다."
        )

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for sample in samples:
        T_bg = np.asarray(
            sample["robot"]["T_base_gripper"],
            dtype=np.float64,
        ).reshape(4, 4)

        R_gripper2base.append(T_bg[:3, :3])
        t_gripper2base.append(T_bg[:3, 3].reshape(3, 1))

        R_target2cam.append(
            np.asarray(
                sample["camera_target"]["R_target2cam"],
                dtype=np.float64,
            ).reshape(3, 3)
        )
        t_target2cam.append(
            np.asarray(
                sample["camera_target"]["t_target2cam_m"],
                dtype=np.float64,
            ).reshape(3, 1)
        )

    R_gc, t_gc = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_PARK,
    )

    R_gc = np.asarray(R_gc, dtype=np.float64).reshape(3, 3)
    t_gc = np.asarray(t_gc, dtype=np.float64).reshape(3)

    if not np.all(np.isfinite(R_gc)) or not np.all(np.isfinite(t_gc)):
        raise RuntimeError("PARK Hand-Eye 결과가 비정상입니다.")

    return make_transform(R_gc, t_gc)


def base_target_transforms(
    samples: List[Dict],
    T_gripper_camera: np.ndarray,
) -> List[np.ndarray]:
    transforms: List[np.ndarray] = []

    for sample in samples:
        T_bg = np.asarray(
            sample["robot"]["T_base_gripper"],
            dtype=np.float64,
        ).reshape(4, 4)

        R_ct = np.asarray(
            sample["camera_target"]["R_target2cam"],
            dtype=np.float64,
        ).reshape(3, 3)

        t_ct = np.asarray(
            sample["camera_target"]["t_target2cam_m"],
            dtype=np.float64,
        ).reshape(3)

        T_ct = make_transform(R_ct, t_ct)

        # ^base T_target
        transforms.append(T_bg @ T_gripper_camera @ T_ct)

    return transforms


def evaluate(
    samples: List[Dict],
    T_gripper_camera: np.ndarray,
) -> Tuple[Dict[str, float], List[Dict]]:
    transforms = base_target_transforms(samples, T_gripper_camera)

    translations = np.stack([T[:3, 3] for T in transforms], axis=0)
    rotations = [T[:3, :3] for T in transforms]

    t_center = np.median(translations, axis=0)
    R_center = average_rotation(rotations)

    trans_errors_mm = (
        np.linalg.norm(translations - t_center, axis=1) * 1000.0
    )

    rot_errors_deg = np.asarray(
        [rotation_angle_deg(R_center, R) for R in rotations],
        dtype=np.float64,
    )

    per_sample = []

    for sample, trans_error, rot_error, T_bt in zip(
        samples,
        trans_errors_mm,
        rot_errors_deg,
        transforms,
    ):
        quality = sample.get("quality", {})

        per_sample.append(
            {
                "sample": sample["_name"],
                "translation_error_mm": float(trans_error),
                "rotation_error_deg": float(rot_error),
                "reprojection_error_px": float(
                    quality.get("median_reprojection_error_px", float("nan"))
                ),
                "translation_spread_mm": float(
                    quality.get("translation_spread_mm", float("nan"))
                ),
                "rotation_spread_deg": float(
                    quality.get("rotation_spread_deg", float("nan"))
                ),
                "base_target_x_mm": float(T_bt[0, 3] * 1000.0),
                "base_target_y_mm": float(T_bt[1, 3] * 1000.0),
                "base_target_z_mm": float(T_bt[2, 3] * 1000.0),
            }
        )

    metrics = {
        "translation_rms_mm": float(
            np.sqrt(np.mean(trans_errors_mm ** 2))
        ),
        "translation_median_mm": float(np.median(trans_errors_mm)),
        "translation_max_mm": float(np.max(trans_errors_mm)),
        "rotation_rms_deg": float(
            np.sqrt(np.mean(rot_errors_deg ** 2))
        ),
        "rotation_median_deg": float(np.median(rot_errors_deg)),
        "rotation_max_deg": float(np.max(rot_errors_deg)),
    }

    return metrics, per_sample


def transform_difference(
    T_a: np.ndarray,
    T_b: np.ndarray,
) -> Tuple[float, float]:
    translation_mm = float(
        np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]) * 1000.0
    )
    rotation_deg = rotation_angle_deg(T_a[:3, :3], T_b[:3, :3])
    return translation_mm, rotation_deg


def robust_threshold(values: List[float], scale: float = 2.5) -> float:
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))

    # MAD가 0에 가까우면 표준편차 기반으로 보완
    if mad < 1e-9:
        return median + scale * float(np.std(arr))

    robust_sigma = 1.4826 * mad
    return median + scale * robust_sigma


def main() -> None:
    dataset_dir = (
        Path(sys.argv[1]).expanduser().resolve()
        if len(sys.argv) >= 2
        else DEFAULT_DATASET_DIR.resolve()
    )

    samples = load_samples(dataset_dir)

    print()
    print("==================================================")
    print("Eye-in-Hand 이상치 분석")
    print("데이터셋:", dataset_dir)
    print("샘플 수:", len(samples))
    print("==================================================")

    if len(samples) < MIN_SAMPLES:
        raise RuntimeError(
            f"샘플이 {len(samples)}개뿐입니다. 최소 {MIN_SAMPLES}개가 필요합니다."
        )

    T_full = calibrate_park(samples)
    full_metrics, per_sample = evaluate(samples, T_full)

    print()
    print("[현재 전체 PARK 결과]")
    print(
        f"위치 RMS: {full_metrics['translation_rms_mm']:.3f} mm"
    )
    print(
        f"회전 RMS: {full_metrics['rotation_rms_deg']:.3f} deg"
    )
    print(
        f"위치 최대오차: {full_metrics['translation_max_mm']:.3f} mm"
    )
    print(
        f"회전 최대오차: {full_metrics['rotation_max_deg']:.3f} deg"
    )

    # Leave-One-Out
    loo_rows = []

    for index, excluded_sample in enumerate(samples):
        reduced = samples[:index] + samples[index + 1 :]

        try:
            T_loo = calibrate_park(reduced)
            loo_metrics, _ = evaluate(reduced, T_loo)

            shift_mm, shift_deg = transform_difference(T_full, T_loo)

            loo_rows.append(
                {
                    "sample": excluded_sample["_name"],
                    "loo_translation_rms_mm": loo_metrics[
                        "translation_rms_mm"
                    ],
                    "loo_rotation_rms_deg": loo_metrics[
                        "rotation_rms_deg"
                    ],
                    "translation_rms_improvement_mm": (
                        full_metrics["translation_rms_mm"]
                        - loo_metrics["translation_rms_mm"]
                    ),
                    "rotation_rms_improvement_deg": (
                        full_metrics["rotation_rms_deg"]
                        - loo_metrics["rotation_rms_deg"]
                    ),
                    "solution_shift_mm": shift_mm,
                    "solution_shift_deg": shift_deg,
                }
            )

        except Exception as exc:
            loo_rows.append(
                {
                    "sample": excluded_sample["_name"],
                    "error": str(exc),
                }
            )

    per_sample_by_name = {row["sample"]: row for row in per_sample}

    combined_rows = []

    for loo in loo_rows:
        name = loo["sample"]
        row = dict(per_sample_by_name[name])
        row.update(loo)
        combined_rows.append(row)

    combined_rows.sort(
        key=lambda row: row.get(
            "translation_rms_improvement_mm",
            -999.0,
        ),
        reverse=True,
    )

    trans_threshold = robust_threshold(
        [row["translation_error_mm"] for row in per_sample]
    )

    rot_threshold = robust_threshold(
        [row["rotation_error_deg"] for row in per_sample]
    )

    print()
    print("[한 장씩 제외했을 때 개선 효과 상위 10개]")
    print(
        "샘플             자체위치오차  자체회전오차  "
        "제외후 위치RMS  개선량    결과이동"
    )

    for row in combined_rows[:10]:
        if "error" in row:
            print(f"{row['sample']:16s} 계산 실패: {row['error']}")
            continue

        print(
            f"{row['sample']:16s} "
            f"{row['translation_error_mm']:9.3f}mm  "
            f"{row['rotation_error_deg']:9.3f}deg  "
            f"{row['loo_translation_rms_mm']:9.3f}mm  "
            f"{row['translation_rms_improvement_mm']:+7.3f}mm  "
            f"{row['solution_shift_mm']:7.3f}mm"
        )

    candidates = []

    for row in combined_rows:
        if "error" in row:
            continue

        residual_bad = (
            row["translation_error_mm"] > trans_threshold
            or row["rotation_error_deg"] > rot_threshold
        )

        meaningful_improvement = (
            row["translation_rms_improvement_mm"] >= 0.20
            or row["rotation_rms_improvement_deg"] >= 0.08
        )

        # 제외 후 해가 지나치게 크게 변하는 샘플은
        # 단순 이상치가 아니라 데이터 다양성에 중요한 샘플일 수 있어 보수적으로 제외
        stable_solution = (
            row["solution_shift_mm"] <= 5.0
            and row["solution_shift_deg"] <= 1.5
        )

        if residual_bad and meaningful_improvement and stable_solution:
            candidates.append(row)

    print()
    print("[삭제가 아니라 우선 검토할 후보]")
    if not candidates:
        print("자동 기준으로 뚜렷한 이상치 후보가 없습니다.")
        print("이 경우 더 지우지 말고 좌표 프레임/자세 변환 문제를 확인하세요.")
    else:
        for row in candidates[:3]:
            print(
                f"- {row['sample']}: "
                f"자체 위치오차 {row['translation_error_mm']:.3f}mm, "
                f"제외 시 위치 RMS "
                f"{row['loo_translation_rms_mm']:.3f}mm "
                f"({row['translation_rms_improvement_mm']:+.3f}mm 개선)"
            )

    csv_path = dataset_dir / "handeye_outlier_analysis.csv"
    json_path = dataset_dir / "handeye_outlier_analysis.json"

    fieldnames = sorted(
        {key for row in combined_rows for key in row.keys()}
    )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)

    report = {
        "dataset_dir": str(dataset_dir),
        "sample_count": len(samples),
        "full_metrics": full_metrics,
        "translation_outlier_threshold_mm": trans_threshold,
        "rotation_outlier_threshold_deg": rot_threshold,
        "review_candidates": [
            {
                "sample": row["sample"],
                "translation_error_mm": row["translation_error_mm"],
                "rotation_error_deg": row["rotation_error_deg"],
                "loo_translation_rms_mm": row[
                    "loo_translation_rms_mm"
                ],
                "translation_rms_improvement_mm": row[
                    "translation_rms_improvement_mm"
                ],
                "solution_shift_mm": row["solution_shift_mm"],
                "solution_shift_deg": row["solution_shift_deg"],
            }
            for row in candidates[:3]
        ],
        "all_rows": combined_rows,
    }

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("분석 CSV:", csv_path)
    print("분석 JSON:", json_path)
    print()
    print("중요: 이 코드는 샘플을 자동 삭제하지 않습니다.")
    print("후보가 1~3개 나오면 그 결과를 먼저 확인한 뒤 제외하세요.")
    print("==================================================")


if __name__ == "__main__":
    main()