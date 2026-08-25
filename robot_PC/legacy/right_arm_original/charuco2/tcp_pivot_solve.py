import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_SAMPLES_FILE = (
    Path.home()
    / "tcp_pivot_dataset"
    / "tcp_pivot_samples.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / "tcp_pivot_dataset"
)


def rot_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    return np.array(
        [[1, 0, 0], [0, c, -s], [0, s, c]],
        dtype=np.float64,
    )


def rot_y(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    return np.array(
        [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        dtype=np.float64,
    )


def rot_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    return np.array(
        [[c, -s, 0], [s, c, 0], [0, 0, 1]],
        dtype=np.float64,
    )


def coords_to_R_t(
    coords_mm_deg: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    if len(coords_mm_deg) != 6:
        raise ValueError(
            "coords는 6개 값이어야 합니다."
        )

    x, y, z, rx_deg, ry_deg, rz_deg = [
        float(value)
        for value in coords_mm_deg
    ]

    rx, ry, rz = np.deg2rad(
        [rx_deg, ry_deg, rz_deg]
    )

    # Hand-Eye 계산과 같은 MyCobot 회전 변환
    R = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    t_mm = np.array(
        [x, y, z],
        dtype=np.float64,
    )

    return R, t_mm


def rotation_difference_deg(
    R_a: np.ndarray,
    R_b: np.ndarray,
) -> float:
    R_delta = R_a.T @ R_b
    cosine = (np.trace(R_delta) - 1.0) / 2.0
    cosine = float(
        np.clip(cosine, -1.0, 1.0)
    )

    return math.degrees(
        math.acos(cosine)
    )


def load_samples(
    path: Path,
) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"샘플 파일이 없습니다: {path}"
        )

    samples = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(samples, list):
        raise RuntimeError(
            "샘플 파일은 리스트 형식이어야 합니다."
        )

    return samples


def solve_pivot(
    samples: List[Dict],
) -> Dict:
    if len(samples) < 4:
        raise RuntimeError(
            f"샘플이 {len(samples)}개입니다. "
            "최소 4개, 권장 8~12개입니다."
        )

    rotations: List[np.ndarray] = []
    translations: List[np.ndarray] = []

    A_rows = []
    b_rows = []

    identity = np.eye(
        3,
        dtype=np.float64,
    )

    for sample in samples:
        R, t_mm = coords_to_R_t(
            sample["coords_mm_deg"]
        )

        rotations.append(R)
        translations.append(t_mm)

        # R_i p_tcp - p_pivot = -t_i
        A_rows.append(
            np.hstack(
                [R, -identity]
            )
        )
        b_rows.append(-t_mm)

    A = np.vstack(A_rows)
    b = np.concatenate(b_rows)

    solution, residuals, rank, singular_values = (
        np.linalg.lstsq(
            A,
            b,
            rcond=None,
        )
    )

    p_flange_tcp_mm = solution[:3]
    p_base_pivot_mm = solution[3:]

    predicted_pivots = np.stack(
        [
            R @ p_flange_tcp_mm + t_mm
            for R, t_mm in zip(
                rotations,
                translations,
            )
        ],
        axis=0,
    )

    errors_mm = np.linalg.norm(
        predicted_pivots - p_base_pivot_mm,
        axis=1,
    )

    rms_mm = float(
        np.sqrt(
            np.mean(errors_mm ** 2)
        )
    )

    max_mm = float(
        np.max(errors_mm)
    )

    pair_rotations = []

    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            pair_rotations.append(
                rotation_difference_deg(
                    rotations[i],
                    rotations[j],
                )
            )

    max_pair_rotation_deg = float(
        max(pair_rotations)
        if pair_rotations
        else 0.0
    )

    median_pair_rotation_deg = float(
        np.median(pair_rotations)
        if pair_rotations
        else 0.0
    )

    condition_number = float(
        np.linalg.cond(A)
    )

    return {
        "sample_count": len(samples),
        "flange_to_tcp_mm": (
            p_flange_tcp_mm.tolist()
        ),
        "flange_to_tcp_length_mm": float(
            np.linalg.norm(
                p_flange_tcp_mm
            )
        ),
        "base_pivot_mm": (
            p_base_pivot_mm.tolist()
        ),
        "rms_error_mm": rms_mm,
        "max_error_mm": max_mm,
        "rank": int(rank),
        "condition_number": condition_number,
        "maximum_pair_rotation_deg": (
            max_pair_rotation_deg
        ),
        "median_pair_rotation_deg": (
            median_pair_rotation_deg
        ),
        "sample_errors_mm": (
            errors_mm.tolist()
        ),
        "predicted_base_pivots_mm": (
            predicted_pivots.tolist()
        ),
    }


def main() -> None:
    if len(sys.argv) >= 2:
        samples_file = Path(
            sys.argv[1]
        ).expanduser().resolve()
    else:
        samples_file = DEFAULT_SAMPLES_FILE

    result = solve_pivot(
        load_samples(samples_file)
    )

    print()
    print("=" * 64)
    print("TCP 피벗 보정 결과")
    print("-" * 64)
    print("샘플 수:", result["sample_count"])
    print(
        "플랜지 → TCP XYZ(mm):",
        np.round(
            result["flange_to_tcp_mm"],
            3,
        ),
    )
    print(
        "플랜지 → TCP 길이(mm):",
        f"{result['flange_to_tcp_length_mm']:.3f}",
    )
    print(
        "고정 피벗 base XYZ(mm):",
        np.round(
            result["base_pivot_mm"],
            3,
        ),
    )
    print(
        "오차 RMS(mm):",
        f"{result['rms_error_mm']:.3f}",
    )
    print(
        "오차 MAX(mm):",
        f"{result['max_error_mm']:.3f}",
    )
    print(
        "최대 자세 회전 차이(deg):",
        f"{result['maximum_pair_rotation_deg']:.3f}",
    )
    print(
        "중앙 자세 회전 차이(deg):",
        f"{result['median_pair_rotation_deg']:.3f}",
    )
    print(
        "행렬 rank:",
        result["rank"],
    )
    print(
        "condition number:",
        f"{result['condition_number']:.3f}",
    )
    print("-" * 64)

    if (
        result["rms_error_mm"] <= 2.0
        and result["max_error_mm"] <= 4.0
        and result["maximum_pair_rotation_deg"] >= 20.0
    ):
        print("판정: GOOD")
        print(
            "이 TCP 값을 안전 접근점 계산에 사용할 수 있습니다."
        )
    else:
        print("판정: REVIEW")
        print(
            "포인터 고정 또는 자세 다양성을 더 확보한 뒤 "
            "다시 측정하는 것이 좋습니다."
        )

    print("=" * 64)

    DEFAULT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        DEFAULT_OUTPUT_DIR
        / "tcp_pivot_calibration.json"
    )
    npz_path = (
        DEFAULT_OUTPUT_DIR
        / "tcp_pivot_calibration.npz"
    )

    json_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    np.savez(
        npz_path,
        flange_to_tcp_mm=np.asarray(
            result["flange_to_tcp_mm"],
            dtype=np.float64,
        ),
        base_pivot_mm=np.asarray(
            result["base_pivot_mm"],
            dtype=np.float64,
        ),
        rms_error_mm=np.array(
            result["rms_error_mm"],
            dtype=np.float64,
        ),
        max_error_mm=np.array(
            result["max_error_mm"],
            dtype=np.float64,
        ),
        sample_count=np.array(
            result["sample_count"],
            dtype=np.int32,
        ),
    )

    print("JSON 저장:", json_path)
    print("NPZ 저장:", npz_path)


if __name__ == "__main__":
    main()