import argparse
import pathlib
import random
import os
import shutil
import numpy as np

from typing import List


def build_repo_name(repo: str):
    repo = repo.split("/")[-1].split("-")
    user = repo[0]
    repo = "-".join(repo[1:])
    return f"{user}/{repo}"


def write_files(split_repos_array: List[np.ndarray]):
    for i, split_repos in enumerate(split_repos_array):
        with open(f"./split_repos/split_repo_{i + 1}.txt", "w",) as f:
            f.writelines(split_repos)


def main(args):
    with open(args.input_file, "r") as f:
        repos = f.readlines()

    repos = list(map(build_repo_name, repos))

    if args.randomize:
        random.shuffle(repos)

    split_repos_array = np.array_split(np.array(repos), len(repos) // args.size)

    if os.path.exists("./split_repos") and os.path.isdir("./split_repos"):
        shutil.rmtree("./split_repos")
    
    os.makedirs("./split_repos")
    write_files(split_repos_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=pathlib.Path)
    parser.add_argument("size", type=int, help="the size of the split files")
    parser.add_argument(
        "--randomize",
        action="store_true",
        default=False,
        help="enable to randomize the files first and then split",
    )

    args = parser.parse_args()
    main(args)
