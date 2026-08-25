import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_HAND_EYE = (
    Path.home()
    / "handeye_dataset_v2"
    / "handeye_calibration.npz"
)


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
    cosine = (np.trace(R_delta) - 1.0) / 2.0
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def load_samples(dataset_dir: Path) -> List[Dict]:
    samples: List[Dict] = []

    for path in sorted(dataset_dir.glob("sample_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_name"] = path.name
        samples.append(data)

    return samples


def load_hand_eye(path: Path) -> Tuple[np.ndarray, Dict]:
    data = np.load(path, allow_pickle=True)

    if "T_gripper_camera" in data:
        T_gc = np.asarray(
            data["T_gripper_camera"],
            dtype=np.float64,
        ).reshape(4, 4)
    else:
        R_gc = np.asarray(
            data["R_cam2gripper"],
            dtype=np.float64,
        ).reshape(3, 3)

        t_gc = np.asarray(
            data["t_cam2gripper_m"],
            dtype=np.float64,
        ).reshape(3)

        T_gc = make_transform(R_gc, t_gc)

    metadata = {}

    for key in (
        "method",
        "translation_rms_mm",
        "rotation_rms_deg",
        "sample_count",
    ):
        if key in data:
            value = data[key]
            try:
                metadata[key] = value.item()
            except Exception:
                metadata[key] = np.asarray(value).tolist()

    return T_gc, metadata


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "사용법:\n"
            "python3 handeye_validate.py "
            "[handeye_calibration.npz] "
            "검증세션폴더\n\n"
            "예:\n"
            "python3 handeye_validate.py "
            "~/handeye_dataset_v2/handeye_calibration.npz "
            "~/handeye_validation_v2/session_20260720_140000"
        )
        raise SystemExit(2)

    if len(sys.argv) == 2:
        hand_eye_path = DEFAULT_HAND_EYE
        validation_dir = Path(sys.argv[1]).expanduser().resolve()
    else:
        hand_eye_path = Path(sys.argv[1]).expanduser().resolve()
        validation_dir = Path(sys.argv[2]).expanduser().resolve()

    if not hand_eye_path.exists():
        raise FileNotFoundError(
            f"Hand-Eye 파일이 없습니다: {hand_eye_path}"
        )

    if not validation_dir.exists():
        raise FileNotFoundError(
            f"검증 폴더가 없습니다: {validation_dir}"
        )

    samples = load_samples(validation_dir)

    if len(samples) < 4:
        raise RuntimeError(
            f"검증 샘플이 {len(samples)}개입니다. 최소 4개, 권장 5개입니다."
        )

    T_gripper_camera, calibration_metadata = load_hand_eye(
        hand_eye_path
    )

    base_target_transforms: List[np.ndarray] = []

    for sample in samples:
        T_base_gripper = np.asarray(
            sample["robot"]["T_base_gripper"],
            dtype=np.float64,
        ).reshape(4, 4)

        R_camera_target = np.asarray(
            sample["camera_target"]["R_target2cam"],
            dtype=np.float64,
        ).reshape(3, 3)

        t_camera_target = np.asarray(
            sample["camera_target"]["t_target2cam_m"],
            dtype=np.float64,
        ).reshape(3)

        T_camera_target = make_transform(
            R_camera_target,
            t_camera_target,
        )

        # base -> target
        T_base_target = (
            T_base_gripper
            @ T_gripper_camera
            @ T_camera_target
        )

        base_target_transforms.append(T_base_target)

    translations = np.stack(
        [T[:3, 3] for T in base_target_transforms],
        axis=0,
    )

    rotations = [
        T[:3, :3] for T in base_target_transforms
    ]

    center_translation = np.median(
        translations,
        axis=0,
    )

    center_rotation = average_rotation(rotations)

    translation_errors_mm = (
        np.linalg.norm(
            translations - center_translation,
            axis=1,
        )
        * 1000.0
    )

    rotation_errors_deg = np.asarray(
        [
            rotation_angle_deg(center_rotation, R)
            for R in rotations
        ],
        dtype=np.float64,
    )

    translation_rms_mm = float(
        np.sqrt(np.mean(translation_errors_mm ** 2))
    )

    rotation_rms_deg = float(
        np.sqrt(np.mean(rotation_errors_deg ** 2))
    )

    translation_max_mm = float(
        np.max(translation_errors_mm)
    )

    rotation_max_deg = float(
        np.max(rotation_errors_deg)
    )

    rows = []

    for sample, T_bt, t_error, r_error in zip(
        samples,
        base_target_transforms,
        translation_errors_mm,
        rotation_errors_deg,
    ):
        rows.append(
            {
                "sample": sample["_name"],
                "base_target_x_mm": float(T_bt[0, 3] * 1000.0),
                "base_target_y_mm": float(T_bt[1, 3] * 1000.0),
                "base_target_z_mm": float(T_bt[2, 3] * 1000.0),
                "translation_error_mm": float(t_error),
                "rotation_error_deg": float(r_error),
                "reprojection_error_px": float(
                    sample.get("quality", {}).get(
                        "median_reprojection_error_px",
                        float("nan"),
                    )
                ),
            }
        )

    # 프로젝트용 실용 판정 기준. 공식 보편 규격이 아니라
    # 현재 타이어 정비 시스템의 다음 단계 진행 여부를 위한 기준이다.
    pass_translation = (
        translation_rms_mm <= 3.0
        and translation_max_mm <= 5.0
    )

    pass_rotation = (
        rotation_rms_deg <= 0.5
        and rotation_max_deg <= 1.0
    )

    passed = pass_translation and pass_rotation

    print()
    print("==================================================")
    print("Eye-in-Hand 별도 검증 결과")
    print("Hand-Eye 파일:", hand_eye_path)
    print("검증 폴더:", validation_dir)
    print("검증 샘플 수:", len(samples))
    print("캘리브레이션 정보:", calibration_metadata)
    print("--------------------------------------------------")

    for row in rows:
        print(
            f"{row['sample']}: "
            f"위치오차={row['translation_error_mm']:.3f}mm, "
            f"회전오차={row['rotation_error_deg']:.3f}deg, "
            f"reproj={row['reprojection_error_px']:.3f}px"
        )

    print("--------------------------------------------------")
    print(f"검증 위치 RMS : {translation_rms_mm:.3f} mm")
    print(f"검증 위치 MAX : {translation_max_mm:.3f} mm")
    print(f"검증 회전 RMS : {rotation_rms_deg:.3f} deg")
    print(f"검증 회전 MAX : {rotation_max_deg:.3f} deg")
    print(
        "고정 보드 대표 base 위치(mm):",
        (center_translation * 1000.0),
    )
    print("--------------------------------------------------")

    if passed:
        print("판정: PASS")
        print(
            "현재 Eye-in-Hand 결과를 실제 좌표 변환 시험에 "
            "사용해도 좋습니다."
        )
    else:
        print("판정: REVIEW")
        print(
            "검증 오차가 목표보다 큽니다. "
            "실제 적용 전에 데이터와 장착 상태를 다시 확인하세요."
        )

    print("==================================================")

    report = {
        "hand_eye_file": str(hand_eye_path),
        "validation_dir": str(validation_dir),
        "validation_sample_count": len(samples),
        "calibration_metadata": calibration_metadata,
        "metrics": {
            "translation_rms_mm": translation_rms_mm,
            "translation_max_mm": translation_max_mm,
            "rotation_rms_deg": rotation_rms_deg,
            "rotation_max_deg": rotation_max_deg,
        },
        "project_acceptance_target": {
            "translation_rms_mm_max": 3.0,
            "translation_max_mm_max": 5.0,
            "rotation_rms_deg_max": 0.5,
            "rotation_max_deg_max": 1.0,
        },
        "passed": passed,
        "center_base_target_translation_mm": (
            center_translation * 1000.0
        ).tolist(),
        "rows": rows,
    }

    json_path = validation_dir / "handeye_validation_report.json"
    csv_path = validation_dir / "handeye_validation_rows.csv"

    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("보고서:", json_path)
    print("샘플별 CSV:", csv_path)


if __name__ == "__main__":
    main()