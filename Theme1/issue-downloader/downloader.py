import random
import csv
import json
import glob
import requests
import re
from pprint import pprint
from tqdm import tqdm
from pathlib import Path


class GitHubIssueDownloader:
    def __init__(
        self,
        download_path: str = ".",
        auth: tuple = None,
    ) -> None:
        self.auth: tuple = auth
        self.download_path: Path = Path(download_path)

    def download_issues(self):
        print("Downloading PyTorch issues")
        pytorch_issues = self._get_pytorch_issues()
        for issue in tqdm(pytorch_issues):
            timeline = self._get_timeline(issue["timeline_url"])
            issue["timeline"] = timeline

        self._write_to_json("all_pytorch_issues.json", pytorch_issues)

        print("Downloading tf2onnx issues")
        tf2onnx_issues = self._get_tf2onnx_issues()
        for issue in tqdm(tf2onnx_issues):
            timeline = self._get_timeline(issue["timeline_url"])
            issue["timeline"] = timeline

        self._write_to_json("all_tf2onnx_issues.json", tf2onnx_issues)

    def _get_pytorch_issues(self):
        labels = ["module: onnx"]
        labels_as_str = ",".join(labels)
        page = 1
        url = f"https://api.github.com/repos/pytorch/pytorch/issues?state=closed&per_page=100&labels={labels_as_str}&page={page}"

        response = requests.get(url, auth=self.auth)
        last_page = self._get_last_page(
            response,
        )
        issues = response.json()

        for i in range(2, last_page + 1):
            page = i
            url = f"https://api.github.com/repos/pytorch/pytorch/issues?state=closed&per_page=100&labels={labels_as_str}&page={page}"
            response = requests.get(url, auth=self.auth)
            issues = issues + response.json()

        filtered_issues = list(filter(lambda x: "pull_request" not in x, issues))
        print(f"Downloaded {len(filtered_issues)} pt issues.")
        return filtered_issues

    def _get_tf2onnx_issues(self):
        page = 1
        url = f"https://api.github.com/repos/onnx/tensorflow-onnx/issues?state=closed&per_page=100&page={page}"

        response = requests.get(url, auth=self.auth)
        last_page = self._get_last_page(
            response,
        )
        issues = response.json()

        for i in range(2, last_page + 1):
            page = i
            url = f"https://api.github.com/repos/onnx/tensorflow-onnx/issues?state=closed&per_page=100&page={page}"
            response = requests.get(url, auth=self.auth)
            issues = issues + response.json()

        filtered_issues = list(filter(lambda x: "pull_request" not in x, issues))
        print(f"Downloaded {len(filtered_issues)} tf issues.")
        filtered_issues = list(
            filter(
                lambda x: self._filter_labels(
                    x["labels"], ["pending on user response", "question"]
                ),
                filtered_issues,
            )
        )
        return filtered_issues

    def _get_timeline(self, timeline_url):
        response = requests.get(timeline_url + "?per_page=100", auth=self.auth)
        timeline = response.json()
        return timeline

    def _get_last_page(self, response):
        last_link = list(
            filter(lambda x: 'rel="last"' in x, response.headers["Link"].split(","))
        )[0]
        last_number = re.search("&page=(\d+)>", last_link).group(1)
        return int(last_number)

    def _filter_labels(self, labels, labels_to_filter):
        mapped_labels = list(map(lambda x: x["name"], labels))
        filter_results = [
            label_name in mapped_labels for label_name in labels_to_filter
        ]
        return not any(filter_results)

    def _write_to_json(self, path, issues):
        path = Path(self.download_path, path)
        with open(path, "w") as f:
            f.write(json.dumps(issues))


if __name__ == "__main__":
    GitHubIssueDownloader(
        download_path="./",
    ).download_issues()
