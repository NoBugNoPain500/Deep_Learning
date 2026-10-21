import subprocess
import sys


def test_train_and_evaluate_smoke(fake_data, tmp_path):
    root, split_dir = fake_data
    output = tmp_path / "artifacts"
    train = [
        sys.executable,
        "-m",
        "src.train",
        "--exp",
        "e8_resnet18",
        "--smoke",
        "--data-dir",
        str(root),
        "--split-dir",
        str(split_dir),
        "--out-dir",
        str(output),
    ]
    subprocess.run(train, check=True, timeout=90)
    experiment = output / "e8_resnet18"
    assert (experiment / "best.pth").exists()
    evaluate = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--exp",
        "e8_resnet18",
        "--smoke",
        "--data-dir",
        str(root),
        "--split-dir",
        str(split_dir),
        "--out-dir",
        str(output),
    ]
    subprocess.run(evaluate + ["--split", "val"], check=True, timeout=90)
    subprocess.run(evaluate + ["--split", "test"], check=True, timeout=90)
    blocked = subprocess.run(
        evaluate + ["--split", "test"], capture_output=True, text=True, timeout=90
    )
    assert blocked.returncode != 0
    assert "--force" in blocked.stderr + blocked.stdout
