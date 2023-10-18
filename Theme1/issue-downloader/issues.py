import glob
import json
import csv
import random
from pathlib import Path

from downloader import GitHubIssueDownloader


class Issues:
    def __init__(
        self,
        download_path: str = ".",
        filter_path: str = ".",
        sample_path: str = ".",
        auth: tuple = None,
    ) -> None:
        self.filter_path = Path(filter_path)
        self.download_path = Path(download_path)
        self.sample_path = Path(sample_path)
        self.downloader = GitHubIssueDownloader(download_path=download_path, auth=auth)

    def download_issues(self):
        self.downloader.download_issues()

    def process_issues(self):
        pytorch_issues_path = glob.glob(
            f"{str(self.download_path)}/**/all_pytorch_issues.json", recursive=True
        )
        if pytorch_issues_path:
            print("PyTorch issues found")
            pytorch_issues = self._get_cached_issues(pytorch_issues_path[0])
            print(f"Total PyTorch Issues: {len(pytorch_issues)}")
            ptlf_pytorch_issues = list(
                filter(lambda x: self._timeline_filter(x["timeline"]), pytorch_issues)
            )

            print(f"Filtered PT Issues: {len(ptlf_pytorch_issues)}")
            self.pytorch_issues = self._issue_map(ptlf_pytorch_issues)
            self._write_to_json(
                self.filter_path, "filt_pytorch_issues.json", ptlf_pytorch_issues
            )
            self._write_issues_csv(
                self.filter_path, "filt_pytorch_issues.csv", self.pytorch_issues
            )

        tf2onnx_issues_path = glob.glob(
            f"{str(self.download_path)}/**/all_tf2onnx_issues.json", recursive=True
        )
        if tf2onnx_issues_path:
            print("tf2onnx issues found")
            tf2onnx_issues = self._get_cached_issues(tf2onnx_issues_path[0])
            print(f"Total TF Issues: {len(tf2onnx_issues)}")
            ptlf_tf2onnx_issues = list(
                filter(lambda x: self._timeline_filter(x["timeline"]), tf2onnx_issues)
            )

            print(f"Filtered TF Issues: {len(ptlf_tf2onnx_issues)}")
            self.tf2onnx_issues = self._issue_map(ptlf_tf2onnx_issues)
            self._write_to_json(
                self.filter_path, "filt_tf2onnx_issues.json", ptlf_tf2onnx_issues
            )
            self._write_issues_csv(
                self.filter_path, "filt_tf2onnx_issues.csv", self.tf2onnx_issues
            )

    def evaluate_filter(self):
        pytorch_issues_path = glob.glob(
            f"{str(self.download_path)}/**/all_pytorch_issues.json", recursive=True
        )
        if pytorch_issues_path:
            pytorch_issues = self._get_cached_issues(pytorch_issues_path[0])
            sampled_issue = random.sample(pytorch_issues, 50)
            sampledf_pytorch_issues = list(
                filter(lambda x: self._timeline_filter(x["timeline"]), sampled_issue)
            )
            sampled_mapped_issue = self._issue_map(sampled_issue)
            sampledf_mapped_issues = self._issue_map(sampledf_pytorch_issues)
            self._write_issues_csv("sampled_pytorch_issues.csv", sampled_mapped_issue)
            self._write_issues_csv(
                "sampledf_pytorch_issues.csv", sampledf_mapped_issues
            )

        tf2onnx_issues_path = glob.glob(
            f"{str(self.download_path)}/**/all_tf2onnx_issues.json", recursive=True
        )
        if tf2onnx_issues_path:
            tf2onnx_issues = self._get_cached_issues(tf2onnx_issues_path[0])
            sampled_issue = random.sample(tf2onnx_issues, 50)
            sampledf_tf2onnx_issues = list(
                filter(lambda x: self._timeline_filter(x["timeline"]), sampled_issue)
            )
            sampled_mapped_issue = self._issue_map(sampled_issue)
            sampledf_mapped_issues = self._issue_map(sampledf_tf2onnx_issues)
            self._write_issues_csv("sampled_tf2onnx_issues.csv", sampled_mapped_issue)
            self._write_issues_csv(
                "sampledf_tf2onnx_issues.csv", sampledf_mapped_issues
            )

    def sample_issues(self, n=100):
        sampled_pytorch = self.sample(self.pytorch_issues, n)
        self._write_issues_csv(self.sample_path, "pytorch_sampled.csv", sampled_pytorch)
        sampled_tf2onnx = self.sample(self.tf2onnx_issues, n)
        self._write_issues_csv(self.sample_path, "tf2onnx_sampled.csv", sampled_tf2onnx)

    def sample(self, issues, n):
        sampled_rows = random.sample(issues, n)
        return sampled_rows

    def _write_to_json(self, folder_path, path, issues):
        path = Path(folder_path, path)
        with open(path, "w") as f:
            f.write(json.dumps(issues))

    def _write_issues_csv(self, folder_path, path, issues):
        path = Path(folder_path, path)
        with open(path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=issues[0].keys())
            writer.writeheader()
            writer.writerows(issues)

    def _get_cached_issues(self, path):
        with open(path, "r") as f:
            issues = json.load(f)
        return issues

    def _timeline_filter(self, timeline):
        for timeline_event in timeline:
            event = timeline_event["event"]
            if event == "closed" and timeline_event["commit_id"]:
                return True
            if event == "referenced" and timeline_event["commit_id"]:
                return True
            if (
                event == "cross-referenced"
                and timeline_event["source"]["issue"]["repository"]["name"]
                in ["pytorch", "tensorflow-onnx"]
                and timeline_event["source"]["issue"]["state"] == "closed"
            ):
                return True
            if event == "connected":
                return True

    def _issue_map(self, issues):
        mapped_issues = list(
            map(
                lambda x: {
                    "id": x[0],
                    "title": x[1]["title"],
                    "url": x[1]["html_url"],
                },
                enumerate(issues),
            )
        )
        return mapped_issues


if __name__ == "__main__":
    issues = Issues(download_path="./", filter_path="./")
    issues.download_issues()
    issues.process_issues()
