"""Minimal GitHub REST client (stdlib only) and the interface the bot depends on."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol

API = "https://api.github.com"


class GitHubError(RuntimeError):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(f"GitHub API {status}: {message}")


@dataclass(frozen=True)
class PullRequest:
    number: int
    head_sha: str
    base_ref: str
    author: str
    draft: bool
    state: str
    merged: bool
    labels: tuple[str, ...] = ()
    head_repo: str | None = None
    merge_commit_sha: str | None = None
    merged_by: str | None = None


@dataclass(frozen=True)
class ChangedFile:
    filename: str
    status: str            # added | modified | removed | renamed | ...


class GitHubClient(Protocol):
    def get_pr(self, number: int) -> PullRequest: ...

    def list_files(self, number: int) -> list[ChangedFile]: ...

    def get_file(self, path: str, ref: str) -> bytes | None: ...

    def comment(self, number: int, body: str) -> None: ...

    def merge(self, number: int, sha: str, *, title: str | None = None) -> None: ...


class GitHubHTTP:
    """Talks to the REST API. Reads PR *data* only: the bot never checks out or executes PR code."""

    def __init__(self, repo: str, token: str, *, api: str = API, opener: Callable | None = None, timeout: float = 30.0):
        self.repo, self.token, self.api = repo, token, api.rstrip("/")
        self.opener = opener or urllib.request.urlopen
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None) -> tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.api + path, data=data, method=method,
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "excore-bot",
                     **({"Content-Type": "application/json"} if data else {})},
        )
        try:
            with self.opener(req, timeout=self.timeout) as resp:
                status, raw = resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            payload = raw.decode("utf-8", errors="replace")
        return status, payload

    def _ok(self, method: str, path: str, body: dict | None = None) -> Any:
        status, payload = self._request(method, path, body)
        if status >= 400:
            msg = payload.get("message") if isinstance(payload, dict) else str(payload)
            raise GitHubError(status, msg or "request failed")
        return payload

    def get_pr(self, number: int) -> PullRequest:
        d = self._ok("GET", f"/repos/{self.repo}/pulls/{number}")
        return PullRequest(
            number=number, head_sha=d["head"]["sha"], base_ref=d["base"]["ref"], author=d["user"]["login"],
            draft=bool(d.get("draft")), state=d["state"], merged=bool(d.get("merged")),
            labels=tuple(label["name"] for label in d.get("labels", [])),
            head_repo=(d["head"].get("repo") or {}).get("full_name"),
            merge_commit_sha=d.get("merge_commit_sha"),
            merged_by=(d.get("merged_by") or {}).get("login"),
        )

    def list_files(self, number: int) -> list[ChangedFile]:
        out: list[ChangedFile] = []
        for page in range(1, 4):                                   # at most 300 files
            batch = self._ok("GET", f"/repos/{self.repo}/pulls/{number}/files?per_page=100&page={page}")
            out += [ChangedFile(f["filename"], f["status"]) for f in batch]
            if len(batch) < 100:
                break
        return out

    def get_file(self, path: str, ref: str) -> bytes | None:
        status, d = self._request(
            "GET", f"/repos/{self.repo}/contents/{urllib.parse.quote(path)}?ref={urllib.parse.quote(ref)}")
        if status == 404:
            return None
        if status >= 400:
            raise GitHubError(status, d.get("message", "request failed") if isinstance(d, dict) else str(d))
        if not isinstance(d, dict) or d.get("encoding") != "base64":
            raise GitHubError(422, f"{path} is not a regular file or is too large to fetch")
        return base64.b64decode(d["content"])

    def comment(self, number: int, body: str) -> None:
        self._ok("POST", f"/repos/{self.repo}/issues/{number}/comments", {"body": body[:60000]})

    def merge(self, number: int, sha: str, *, title: str | None = None) -> None:
        """Squash-merge, refusing if the head moved since evaluation (``sha`` must still match)."""
        body = {"sha": sha, "merge_method": "squash"}
        if title:
            body["commit_title"] = title
        self._ok("PUT", f"/repos/{self.repo}/pulls/{number}/merge", body)
