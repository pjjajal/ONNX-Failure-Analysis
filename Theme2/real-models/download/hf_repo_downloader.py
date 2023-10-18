import argparse
import os
import pathlib
import logging
from huggingface_hub import snapshot_download




def main(args):
    with open(args.repo_list, "r") as f:
        repos = f.readlines()

    repos = list(map(lambda x: x.rstrip("\n"), repos))

    for repo in repos:
        repo_download_folder = "-".join(repo.split("/"))
        try:
            snapshot_download(
                repo, cache_dir=f"{args.download_location}/{repo_download_folder}"
            )
        except Exception as err:
            logging.error(f"Repo: {repo} has error: {err}")
            print(err)


if __name__ == "__main__":
    from huggingface_hub import snapshot_download
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_list", type=pathlib.Path)
    parser.add_argument("download_location", type=pathlib.Path)
    parser.add_argument("log_location", type=pathlib.Path)
    args = parser.parse_args()

    logging.basicConfig(
        filename=f"{args.log_location}/download.log",
        level=logging.ERROR,
        filemode="w",
        format="%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s",
        datefmt="%I:%M:%S %p",
    )

    main(args)