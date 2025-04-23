import argparse
import os
import re
import shutil
import sys
from pathlib import Path

SOURCE = Path("free-spoken-digit-dataset/recordings")


def parse():
    p = argparse.ArgumentParser("Data Partitioning")
    p.add_argument("folder", type=str)
    p.add_argument(
        "-r",
        "--regex",
        type=str,
        help="Regex to select testing data files.",
        default=r"4\d\.wav",
    )

    args = p.parse_args(None if sys.argv[1:] else ["-h"])
    args.folder = Path(args.folder)
    return args


if __name__ == "__main__":
    args = parse()

    TRAIN_DIR = args.folder / "train_data"
    TEST_DIR = args.folder / "test_data"

    try:
        os.makedirs(TRAIN_DIR)
        os.makedirs(TEST_DIR)
    except OSError as e:
        print(e)
        print("Must delete the existing folder first.")
        exit(1)

    testing_pat = re.compile(args.regex)
    num_test = 0
    num_train = 0
    wav_files = os.listdir(SOURCE)

    print(f"Found {len(wav_files)} audio files in {SOURCE}.")

    for file in wav_files:
        if testing_pat.search(file):
            shutil.copy(SOURCE / file, TEST_DIR / file)
            num_test += 1
        else:
            shutil.copy(SOURCE / file, TRAIN_DIR / file)
            num_train += 1

    print(f"Copied {num_train} training files to {TRAIN_DIR}.")
    print(f"Copied {num_test} testing files to {TEST_DIR}.")

    with open(args.folder / ".gitignore", "w") as f:
        f.write("*")
