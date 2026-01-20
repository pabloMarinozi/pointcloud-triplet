import subprocess
import itertools

DATA_DIR = r"D:\TU_DATASET_PLY"  # CAMBIAR
WIDTHS_TO_TEST = [2, 3, 4, 8, 16, 32, 64]
N_POINTS_TO_TEST = [512, 1024, 2048, 4096]

for width, n_points in itertools.product(WIDTHS_TO_TEST, N_POINTS_TO_TEST):
    cmd = [
        "python", "-m", "src.train",
        "--data_dir", DATA_DIR,
        "--width", str(width),
        "--n_points", str(n_points),
        "--epochs", "30",
        "--batch_size", "16",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
