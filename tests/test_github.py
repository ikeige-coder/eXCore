import base64
import io
import json
import urllib.error

import pytest

from evaluator.github import GitHubError, GitHubHTTP


class Resp:
    def __init__(self, status, payload):
        self.status, self._raw = status, json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class Opener:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, req, timeout=None):
        body = json.loads(req.data) if req.data else None
        self.calls.append((req.get_method(), req.full_url, body, dict(req.header_items())))
        key = (req.get_method(), req.full_url.split("?")[0].replace("https://api.github.com", ""))
        status, payload = self.routes[key] if not callable(self.routes[key]) else self.routes[key](req)
        if status >= 400:
            raise urllib.error.HTTPError(req.full_url, status, "err", {}, io.BytesIO(json.dumps(payload).encode()))
        return Resp(status, payload)


PR = {"head": {"sha": "abc", "repo": {"full_name": "fork/repo"}}, "base": {"ref": "main"}, "user": {"login": "miner"},
      "draft": False, "state": "open", "merged": False, "labels": [{"name": "x"}]}


def client(routes):
    op = Opener(routes)
    return GitHubHTTP("org/excore", "tok", opener=op), op


def test_get_pr_and_headers():
    gh, op = client({("GET", "/repos/org/excore/pulls/7"): (200, PR)})
    pr = gh.get_pr(7)
    assert (pr.head_sha, pr.author, pr.labels, pr.head_repo, pr.draft) == ("abc", "miner", ("x",), "fork/repo", False)
    h = {k.lower(): v for k, v in op.calls[0][3].items()}
    assert h["authorization"] == "Bearer tok" and "json" in h["accept"]


def test_get_pr_reads_merge_details():
    merged = {**PR, "merged": True, "state": "closed", "merge_commit_sha": "msha", "merged_by": {"login": "owner"}}
    gh, _ = client({("GET", "/repos/org/excore/pulls/7"): (200, merged)})
    pr = gh.get_pr(7)
    assert pr.merged and pr.merge_commit_sha == "msha" and pr.merged_by == "owner"
    open_pr = client({("GET", "/repos/org/excore/pulls/7"): (200, PR)})[0].get_pr(7)
    assert open_pr.merge_commit_sha is None and open_pr.merged_by is None


def test_list_files_paginates():
    page = lambda n, k: [{"filename": f"manifests/f{n}-{i}.yaml", "status": "added"} for i in range(k)]

    def files(req):
        return 200, page(1, 100) if "&page=1" in req.full_url else page(2, 5)

    gh, _ = client({("GET", "/repos/org/excore/pulls/7/files"): files})
    assert len(gh.list_files(7)) == 105


def test_get_file_decodes_and_handles_404_and_dirs():
    enc = base64.b64encode(b"hello: world\n").decode()
    gh, op = client({("GET", "/repos/org/excore/contents/manifests/a.yaml"): (200, {"encoding": "base64", "content": enc}),
                     ("GET", "/repos/org/excore/contents/manifests/none.yaml"): (404, {"message": "Not Found"}),
                     ("GET", "/repos/org/excore/contents/manifests"): (200, [{"name": "a.yaml"}])})
    assert gh.get_file("manifests/a.yaml", "abc") == b"hello: world\n"
    assert "ref=abc" in op.calls[0][1]
    assert gh.get_file("manifests/none.yaml", "abc") is None
    with pytest.raises(GitHubError, match="not a regular file"):
        gh.get_file("manifests", "abc")


def test_comment_and_merge_pin_the_sha():
    gh, op = client({("POST", "/repos/org/excore/issues/7/comments"): (201, {}),
                     ("PUT", "/repos/org/excore/pulls/7/merge"): (200, {"merged": True})})
    gh.comment(7, "hi")
    gh.merge(7, "abc", title="t")
    assert op.calls[0][2] == {"body": "hi"}
    assert op.calls[1][2] == {"sha": "abc", "merge_method": "squash", "commit_title": "t"}


def test_errors_surface():
    gh, _ = client({("PUT", "/repos/org/excore/pulls/7/merge"): (409, {"message": "Head branch was modified"}),
                    ("GET", "/repos/org/excore/pulls/8"): (500, "boom")})
    with pytest.raises(GitHubError, match="409.*Head branch was modified"):
        gh.merge(7, "old")
    with pytest.raises(GitHubError) as e:
        gh.get_pr(8)
    assert e.value.status == 500
