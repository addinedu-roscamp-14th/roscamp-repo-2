from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[4]


def test_required_top_level_directories_exist():
    required = {
        "central_ws", "vision_ws", "arm_vision_ws", "raspberry_left_ws",
        "raspberry_right_ws", "calibration", "legacy", "scripts", "docs",
    }
    assert required.issubset({path.name for path in ROOT.iterdir() if path.is_dir()})


def test_deployment_excludes_generated_directories():
    common = (ROOT / "scripts" / "_deploy.sh").read_text()
    for name in ("build/", "install/", "log/"):
        assert f"--exclude={name}" in common
    for script in ROOT.glob("scripts/deploy_*.sh"):
        assert "--dry-run" in script.read_text()


def test_raspberry_workspaces_have_no_generated_directories():
    for workspace in ("raspberry_left_ws", "raspberry_right_ws"):
        names = {
            path.name for path in (ROOT / workspace).rglob("*") if path.is_dir()
        }
        assert not {"build", "install", "log"} & names


def test_no_user_absolute_paths_in_deployable_files():
    roots = [
        ROOT / "central_ws" / "src", ROOT / "vision_ws" / "src",
        ROOT / "arm_vision_ws" / "src", ROOT / "raspberry_left_ws" / "src",
        ROOT / "raspberry_right_ws" / "src", ROOT / "calibration",
    ]
    pattern = re.compile(r"/home/(?:choiminjoon|soo|seohyun)/")
    offenders = []
    for base in roots:
        for path in base.rglob("*"):
            if path.is_file() and path.suffix not in {".pyc", ".npz", ".pt"}:
                try:
                    if pattern.search(path.read_text(encoding="utf-8")):
                        offenders.append(str(path.relative_to(ROOT)))
                except UnicodeDecodeError:
                    pass
    assert offenders == []


def test_left_and_right_arm_control_code_match_except_configuration():
    left = ROOT / "raspberry_left_ws/src/tire_arm_control/tire_arm_control"
    right = ROOT / "raspberry_right_ws/src/tire_arm_control/tire_arm_control"
    for name in ("arm_driver.py", "motion_player.py", "command_processor.py", "tcp_server_node.py"):
        assert (left / name).read_bytes() == (right / name).read_bytes()
