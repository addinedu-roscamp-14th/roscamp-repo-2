import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


DEFAULT_DATASET_DIR = Path(
    "/home/choiminjoon/handeye_dataset_left"
)

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


def rotation_matrix_to_zyx_deg(R: np.ndarray) -> List[float]:
    """
    R = Rz(yaw) Ry(pitch) Rx(roll)에서 [roll, pitch, yaw] 반환.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-9

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return np.rad2deg([roll, pitch, yaw]).tolist()


def load_samples(dataset_dir: Path) -> List[Dict]:
    samples = []
    for path in sorted(dataset_dir.glob("sample_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["_path"] = str(path)
            samples.append(data)
        except Exception as exc:
            print("샘플 읽기 실패:", path, exc)
    return samples


def evaluate_solution(
    samples: List[Dict],
    T_gripper_camera: np.ndarray,
) -> Dict[str, float]:
    """
    보드가 고정되어 있으므로
    ^base T_target_i =
      ^base T_gripper_i @ ^gripper T_camera @ ^camera T_target_i
    가 모든 샘플에서 같아야 한다.
    """
    base_target_transforms = []

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
        T_bt = T_bg @ T_gripper_camera @ T_ct
        base_target_transforms.append(T_bt)

    translations = np.stack(
        [T[:3, 3] for T in base_target_transforms],
        axis=0,
    )
    rotations = [T[:3, :3] for T in base_target_transforms]

    t_center = np.median(translations, axis=0)
    R_center = average_rotation(rotations)

    trans_errors_mm = (
        np.linalg.norm(translations - t_center, axis=1)
        * 1000.0
    )
    rot_errors_deg = np.asarray(
        [rotation_angle_deg(R_center, R) for R in rotations],
        dtype=np.float64,
    )

    return {
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
        "base_target_median_m": t_center.tolist(),
        "base_target_rotation": R_center.tolist(),
    }


def robot_pose_diversity(samples: List[Dict]) -> Dict[str, float]:
    transforms = [
        np.asarray(
            sample["robot"]["T_base_gripper"],
            dtype=np.float64,
        ).reshape(4, 4)
        for sample in samples
    ]

    translations = np.stack([T[:3, 3] for T in transforms], axis=0)
    t_span_mm = (
        np.max(translations, axis=0) - np.min(translations, axis=0)
    ) * 1000.0

    rotations = [T[:3, :3] for T in transforms]
    pair_angles = []
    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            pair_angles.append(
                rotation_angle_deg(rotations[i], rotations[j])
            )

    return {
        "translation_span_x_mm": float(t_span_mm[0]),
        "translation_span_y_mm": float(t_span_mm[1]),
        "translation_span_z_mm": float(t_span_mm[2]),
        "maximum_pair_rotation_deg": float(max(pair_angles, default=0.0)),
        "median_pair_rotation_deg": float(np.median(pair_angles))
        if pair_angles
        else 0.0,
    }


def main() -> None:
    dataset_dir = (
        Path(sys.argv[1]).expanduser()
        if len(sys.argv) >= 2
        else DEFAULT_DATASET_DIR
    )

    samples = load_samples(dataset_dir)

    print("데이터셋:", dataset_dir)
    print("샘플 수:", len(samples))

    if len(samples) < MIN_SAMPLES:
        raise RuntimeError(
            f"샘플이 {len(samples)}개뿐입니다. 최소 {MIN_SAMPLES}개, "
            "권장 20~30개를 수집하세요."
        )

    diversity = robot_pose_diversity(samples)
    print("로봇 자세 다양성:", diversity)

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

    methods: List[Tuple[str, int]] = [
        ("TSAI", cv2.CALIB_HAND_EYE_TSAI),
        ("PARK", cv2.CALIB_HAND_EYE_PARK),
        ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
        ("ANDREFF", cv2.CALIB_HAND_EYE_ANDREFF),
        ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS),
    ]

    results = []

    for name, method in methods:
        try:
            R_gc, t_gc = cv2.calibrateHandEye(
                R_gripper2base,
                t_gripper2base,
                R_target2cam,
                t_target2cam,
                method=method,
            )

            R_gc = np.asarray(R_gc, dtype=np.float64).reshape(3, 3)
            t_gc = np.asarray(t_gc, dtype=np.float64).reshape(3)

            if (
                not np.all(np.isfinite(R_gc))
                or not np.all(np.isfinite(t_gc))
                or abs(np.linalg.det(R_gc) - 1.0) > 0.05
            ):
                raise ValueError("비정상 행렬 결과")

            T_gc = make_transform(R_gc, t_gc)
            metrics = evaluate_solution(samples, T_gc)

            # mm와 degree를 함께 반영한 단순 비교 점수
            score = (
                metrics["translation_rms_mm"]
                + 2.0 * metrics["rotation_rms_deg"]
            )

            result = {
                "method": name,
                "score": float(score),
                "R_cam2gripper": R_gc.tolist(),
                "t_cam2gripper_m": t_gc.tolist(),
                "t_cam2gripper_mm": (t_gc * 1000.0).tolist(),
                "cam2gripper_rpy_deg_zyx": rotation_matrix_to_zyx_deg(R_gc),
                "metrics": metrics,
            }
            results.append(result)

            print()
            print(
                f"{name:10s} | "
                f"trans RMS={metrics['translation_rms_mm']:.3f}mm | "
                f"rot RMS={metrics['rotation_rms_deg']:.3f}deg | "
                f"score={score:.3f}"
            )

        except Exception as exc:
            print(f"{name} 실패:", exc)

    if not results:
        raise RuntimeError("모든 Hand-Eye 계산 방법이 실패했습니다.")

    results.sort(key=lambda item: item["score"])
    best = results[0]

    R_best = np.asarray(
        best["R_cam2gripper"],
        dtype=np.float64,
    ).reshape(3, 3)
    t_best = np.asarray(
        best["t_cam2gripper_m"],
        dtype=np.float64,
    ).reshape(3)
    T_best = make_transform(R_best, t_best)

    output_npz = dataset_dir / "handeye_calibration.npz"
    output_report = dataset_dir / "handeye_report.json"

    np.savez(
        output_npz,
        R_cam2gripper=R_best,
        t_cam2gripper_m=t_best.reshape(3, 1),
        T_gripper_camera=T_best,
        method=np.asarray(best["method"]),
        translation_rms_mm=np.asarray(
            best["metrics"]["translation_rms_mm"]
        ),
        rotation_rms_deg=np.asarray(
            best["metrics"]["rotation_rms_deg"]
        ),
        sample_count=np.asarray(len(samples)),
    )

    report = {
        "dataset_dir": str(dataset_dir),
        "sample_count": len(samples),
        "robot_pose_diversity": diversity,
        "best": best,
        "all_methods": results,
        "output_npz": str(output_npz),
        "important_frame_meaning": {
            "T_gripper_camera": (
                "카메라 좌표의 점을 로봇 gripper/flange 좌표로 변환"
            ),
            "equation": "P_gripper = T_gripper_camera @ P_camera",
        },
    }

    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("==================================================")
    print("Eye-in-Hand 캘리브레이션 완료")
    print("선택 방법:", best["method"])
    print("카메라 -> gripper 이동량(mm):")
    print(np.asarray(best["t_cam2gripper_mm"]))
    print("카메라 -> gripper RPY(deg, ZYX):")
    print(np.asarray(best["cam2gripper_rpy_deg_zyx"]))
    print(
        "고정 보드 위치 일관성: "
        f"{best['metrics']['translation_rms_mm']:.3f}mm, "
        f"{best['metrics']['rotation_rms_deg']:.3f}deg"
    )
    print("저장:", output_npz)
    print("보고서:", output_report)
    print("==================================================")


if __name__ == "__main__":
    main()