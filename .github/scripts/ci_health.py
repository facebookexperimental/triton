#!/usr/bin/env python3
"""Collect GPU workflow runs and render the /ci/ dashboard.

  collect DATA_DIR [--since ISO]
      Fetch runs of gpu-tests.yml, gpu-tests-nightly.yml,
      nightly-tlx-ops-bench.yml and nightly-fbtriton.yml, and push runs of
      compiler.yml (LIT), created since the saved cursor (or --since), plus
      the jobs and steps of every completed run, the failing tests in the
      log of every failed GPU or LIT job and the slowest tests of each GPU
      job of a main push run, into
      DATA_DIR/runs-YYYY-MM.jsonl.gz and DATA_DIR/state.json, and each
      tracked branch's Meta LLVM pin into DATA_DIR/bump.json. ci-health.yml
      keeps DATA_DIR on the 'ci-health' release; DATA_DIR/changed.txt lists
      the files this run rewrote, for upload.
  render OUTDIR --from DATA_DIR [--bench-from DIR]
      Write OUTDIR/index.html (the page, no data) and OUTDIR/data.json (what
      the page shows, computed by ci_health_rules.page_data), for pages.yml
      to mount at /ci/. The page loads data.json asynchronously. Perf events
      come from the 'tlx-ops-bench' release (or the summaries in DIR).

Stored run record, one JSON object per line in the month of created_at; a later
line for the same id replaces an earlier one:
  {id, wf, event, status, conclusion, created_at, head_sha, title, url, branch, pr, author,
   followup_status?, followup_producer_run_id?, followup_producer_attempt?, pr_head_sha?,
   select: {h100, b200, mi350, build, cpu?, fallback} | null (gpu-tests.yml only),
   jobs: [{name, conclusion, created_at, started_at, completed_at, labels,
           steps: [{name, conclusion, started_at, completed_at,
                    failures?: [{id, msg}], slowest?: [{id, s}]}]}] | null until the run completes}
slowest is pytest's --durations list (seconds per test, slowest first), stored on
main push runs of gpu-tests.yml only.
Records of mi350/b200/h100.yml from before #3875 may remain in the store; the
rules ignore them.

bump.json, rewritten whole by each collection:
  {lines: {<tracked branch>: {fork_branch, pinned: {tag, llvm_hash, base} | null,
   newest: {tag, llvm_hash, published_at} | null, pending: [fork change],
   pr: {number, title, url, author, created_at} | null, issues: [{number, title, url, created_at}],
   history: [{sha, t, title, pr, tag, llvm_hash, prev_hash | null}]}}}
  fork change: {sha, title, pr}, a fork commit with its fork PR number. pending is
  what newest adds over pinned; base is the upstream (OAI) LLVM commit the pin
  is built on; history is the pin's last changes on the branch,
  newest first, each with the llvm_hash it replaced (null for the first pin).
"""
import argparse
import base64
import copy
import datetime as dt
import functools
import glob
import gzip
import json
import os
import re
import shutil
import subprocess
import urllib.parse

import ci_health_logs as logs
import ci_health_rules as rules
import ci_health_sanitizer as sanitizer
import meta_llvm_bump as mlb
import tlx_ops_bench_dashboard as bench

REPO = os.environ.get("GITHUB_REPOSITORY", "facebookexperimental/triton")
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "ci_health_dashboard.html")
#: Stop fetching jobs below this many remaining API calls; the next scheduled run continues.
RATE_FLOOR = 200
#: First collection with no saved state goes back this far.
BACKFILL_DAYS = 30
#: A run still pending after this many days is treated as stuck.
STUCK_DAYS = 7
#: Workflows collected. Since #3875, gpu-tests.yml calls mi350/b200/h100.yml as reusable workflows;
#: gpu-tests-nightly.yml runs the slow suites (TritonBench) once a day.
#: compiler.yml runs LIT on every PR too; only its push runs are collected.
COLLECT = ("gpu-tests.yml", "gpu-tests-nightly.yml", rules.BENCH_WF, "nightly-fbtriton.yml", rules.LIT_WF,
           rules.FAST_WF, sanitizer.WORKFLOW)
FORK = "facebookexperimental/llvm-project"
#: Fork branch each tracked branch's cmake/meta-llvm-info.json follows, as in meta-llvm-update.yml.
FORK_BRANCHES = {"main": "triton_beta", "release/3.8.x": "triton_3.8.x"}
META_PATH = "cmake/meta-llvm-info.json"
#: Pin changes listed per branch.
PIN_HISTORY = 6


def gh_api(path):
    out = subprocess.run(["gh", "api", "--paginate", "--slurp", path], capture_output=True, text=True,
                         check=True).stdout
    return json.loads(out)


def gh_get(path):
    """One page of an endpoint. gh_api's --paginate would walk all of a fork branch's commits."""
    return json.loads(subprocess.run(["gh", "api", path], capture_output=True, text=True, check=True).stdout)


def gh_text(path):
    """A plain-text endpoint, such as a job log (gh follows the redirect to the log blob). gh refuses to print a
    response with terminal escape sequences unless allowed, and job logs can hold them."""
    return subprocess.run(["gh", "api", "--allow-escape-sequences", path], capture_output=True, text=True,
                          check=True).stdout


def gh_bytes(path):
    return subprocess.run(["gh", "api", "--allow-escape-sequences", path], capture_output=True, check=True).stdout


def month_files(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, "runs-*.jsonl.gz")))


def load_runs(data_dir):
    runs = {}
    for path in month_files(data_dir):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    runs[row["id"]] = row
    return runs


def _encode(rows):
    return "".join(
        json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n" for r in sorted(rows, key=lambda r: r["id"]))


def save_runs(data_dir, runs):
    """Write one file per month of created_at; rewrite only the files whose content changed."""
    os.makedirs(data_dir, exist_ok=True)
    by_month = {}
    for r in runs.values():
        by_month.setdefault(r["created_at"][:7], []).append(r)
    changed = []
    for month, rows in sorted(by_month.items()):
        path = os.path.join(data_dir, f"runs-{month}.jsonl.gz")
        body = _encode(rows)
        if os.path.exists(path):
            with gzip.open(path, "rt") as fh:
                if fh.read() == body:
                    continue
        with gzip.open(path, "wt") as fh:
            fh.write(body)
        changed.append(path)
    return changed


def _record(run, wf):
    return {
        "id": run["id"], "wf": wf, "event": run["event"], "status": run["status"], "conclusion": run["conclusion"],
        "created_at": run["created_at"], "head_sha": run.get("head_sha"), "title": run.get("display_title", ""), "url":
        run.get("html_url", ""), "branch": run.get("head_branch"), "pr": None, "author": None, "jobs": None, "base":
        ((run.get("pull_requests") or [{}])[0].get("base") or {}).get("ref"), "run_attempt": run.get("run_attempt", 1),
        "updated_at": run.get("updated_at") or run["created_at"], "tested_sha": None, "tested_sha_provenance": []
    }


def _jobs(pages):
    return [{
        "id":
        j.get("id"), "run_attempt":
        j.get("run_attempt", 1), "name":
        j["name"], "conclusion":
        j["conclusion"], "created_at":
        j.get("created_at"), "started_at":
        j["started_at"], "completed_at":
        j["completed_at"], "labels":
        j.get("labels") or [], "steps": [{
            "name": s["name"], "conclusion": s["conclusion"], "started_at": s.get("started_at"), "completed_at":
            s.get("completed_at")
        } for s in j.get("steps") or []]
    } for page in pages for j in page["jobs"]]


def _pr_and_author(run, api):
    """Push runs are triggered by the ShipIt bot, so the PR comes from the title's "(#N)". PR runs from forks
    have no pull_requests entry, so the PR is looked up by head branch."""
    try:
        if run["event"] == "push":
            m = re.search(r"\(#(\d+)\)\s*$", run.get("display_title") or "")
            if m:
                pr = api(f"repos/{REPO}/pulls/{m.group(1)}")[0]
                return pr["number"], pr["user"]["login"]
        elif run["event"] == "pull_request":
            author = (run.get("actor") or {}).get("login")
            if run.get("pull_requests"):
                return run["pull_requests"][0]["number"], author
            owner = ((run.get("head_repository") or {}).get("owner") or {}).get("login")
            found = api(f"repos/{REPO}/pulls?head={owner}:{run.get('head_branch')}&state=all")[0]
            return (found[0]["number"] if len(found) == 1 else None), author
    except (subprocess.CalledProcessError, KeyError, IndexError, TypeError):
        pass
    return None, None


def _followup_log_identity(jobs, text):
    """The verified producer object printed by the trusted follow-up job, or None.

    GitHub prefixes each log line with a timestamp. Decode complete JSON objects
    after removing those prefixes, then require exactly one producer result.
    """
    verify = next((job for job in jobs if job["name"] == "followup / verify"), None)
    if verify is None:
        return None
    try:
        body = text(f"repos/{REPO}/actions/jobs/{verify['id']}/logs")
    except subprocess.CalledProcessError:
        return None
    messages = []
    for line in body.splitlines():
        stamp, sep, message = line.partition(" ")
        messages.append(message if sep and stamp.endswith("Z") else line)
    body = "\n".join(messages)
    decoder, found = json.JSONDecoder(), []
    for start, char in enumerate(body):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(body[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and {"status", "pr", "pr_head", "producer_run_id", "producer_attempt"
                                        } <= value.keys():
            found.append(value)
    return (found[0], verify) if len(found) == 1 else None


def _followup_identity(jobs, text, api):
    """Return a workflow_run follow-up's API-checked Fast producer identity.

    The display title is only presentation. Correlation uses the trusted verify
    job's result and the exact producer attempt from the Actions API.
    """
    parsed = _followup_log_identity(jobs, text)
    if parsed is None:
        return None
    source, verify_job = parsed
    run_id, attempt = source.get("producer_run_id"), source.get("producer_attempt")
    if not isinstance(run_id, int) or not isinstance(attempt, int):
        return None
    try:
        response = api(f"repos/{REPO}/actions/runs/{run_id}/attempts/{attempt}")
        producer = response[0] if len(response) == 1 else None
    except (subprocess.CalledProcessError, KeyError, IndexError, TypeError):
        return None
    if producer is None:
        return None
    path = str(producer.get("path", "")).split("@", 1)[0]
    repository = (producer.get("repository") or {}).get("full_name", REPO)
    try:
        prs = api(f"repos/{REPO}/pulls/{source.get('pr')}")
        pr = prs[0] if len(prs) == 1 else None
    except (subprocess.CalledProcessError, KeyError, IndexError, TypeError):
        return None
    if pr is None:
        return None
    head, base = pr.get("head") or {}, pr.get("base") or {}
    head_repo, base_repo = head.get("repo") or {}, base.get("repo") or {}
    producer_head_repo = producer.get("head_repository") or {}
    author = (pr.get("user") or {}).get("login")
    if (producer.get("id") != run_id or producer.get("run_attempt", 1) != attempt
            or producer.get("event") != "pull_request" or producer.get("conclusion") != "success"
            or path != f".github/workflows/{rules.FAST_WF}" or repository != REPO
            or producer.get("head_sha") != source.get("pr_head")
            or producer.get("head_branch") != source.get("head_ref")
            or producer_head_repo.get("full_name") != source.get("head_repo") or pr.get("number") != source.get("pr")
            or head_repo.get("full_name") != source.get("head_repo") or head.get("ref") != source.get("head_ref")
            or base_repo.get("full_name") != REPO or base.get("ref") != source.get("base_ref") or not author):
        return None
    status = source.get("status")
    if status not in ("ok", "stale"):
        return None
    out = {
        "followup_status": status, "followup_producer_run_id": run_id, "followup_producer_attempt": attempt, "pr":
        pr["number"], "author": author, "pr_head_sha": source["pr_head"], "pr_branch": source.get("head_ref"),
        "head_repo": source.get("head_repo"), "base": source.get("base_ref"), "followup_verify_job": verify_job["id"]
    }
    if status == "ok":
        merge, base = source.get("merge"), source.get("base")
        if not (isinstance(merge, str) and re.fullmatch(r"[0-9a-f]{40}", merge) and isinstance(base, str)
                and re.fullmatch(r"[0-9a-f]{40}", base)):
            return None
        out.update(followup_merge=merge, followup_base=base)
    return out


def _attach_failures(jobs, text, wf="gpu-tests.yml"):
    for j in jobs:
        if j["conclusion"] != "failure" or rules.test_platform_of({"wf": wf}, j) is None:
            continue
        try:
            logs.attach_failures(j, text(f"repos/{REPO}/actions/jobs/{j['id']}/logs"), wf != rules.BENCH_WF)
        except subprocess.CalledProcessError as exc:  # logs expired or not readable: no test result
            print(f"no log for job {j['id']} ({j['name']}): {exc.stderr.strip() if exc.stderr else exc}")


def _attach_durations(jobs, text):
    """The slowest tests of each GPU job that ran its tests, from pytest's --durations output."""
    for j in jobs:
        if j["conclusion"] not in ("success", "failure") or rules.gpu_of({"wf": "gpu-tests.yml"}, j) is None:
            continue
        try:
            logs.attach_durations(j, text(f"repos/{REPO}/actions/jobs/{j['id']}/logs"))
        except subprocess.CalledProcessError as exc:  # logs expired or not readable: no durations
            print(f"no log for job {j['id']} ({j['name']}): {exc.stderr.strip() if exc.stderr else exc}")


def _attach_lit_failures(jobs, text):
    for j in jobs:
        if j["name"] != rules.LIT_JOB or j["conclusion"] != "failure":
            continue
        try:
            logs.attach_lit_failures(j, text(f"repos/{REPO}/actions/jobs/{j['id']}/logs"))
        except subprocess.CalledProcessError as exc:  # logs expired or not readable: no test result
            j["log_missing"] = True  # so the failure is not mistaken for a build failure
            print(f"no log for job {j['id']} ({j['name']}): {exc.stderr.strip() if exc.stderr else exc}")


def _selection(jobs, text):
    for j in jobs:
        if rules.job_name(j["name"]) == "select" and j["conclusion"] in ("success", "failure"):
            try:
                return logs.parse_selection(text(f"repos/{REPO}/actions/jobs/{j['id']}/logs"))
            except subprocess.CalledProcessError:
                return None
    return None


def _checkout_from_step(job, log_text, step_name=None):
    """Read one checkout's logged HEAD, optionally restricting the parser to one named step."""
    if step_name is None:
        return logs.checkout_sha(job, log_text)
    steps = [{**step, "name": "Checkout"} for step in job["steps"] if step["name"] == step_name]
    return logs.checkout_sha({**job, "steps": steps}, log_text) if len(steps) == 1 else None


def _tested_revision(run, jobs, text):
    """Verify each stage against its selector and the one shared wheel producer."""
    if run["wf"] == "gpu-tests.yml" and run["event"] == "workflow_run":
        run["provenance_version"] = 2
        run["run_attempt"] = run.get("run_attempt", 1)
        evidence = []
        merge = run.get("followup_merge") if run.get("followup_status") == "ok" else None
        if merge:
            evidence.append({
                "sha": merge, "job": "followup / verify", "job_id": run["followup_verify_job"], "run_attempt":
                run["run_attempt"], "job_attempt": 1, "source": "API-verified Fast producer and test merge", "url":
                f"{run['url']}/job/{run['followup_verify_job']}"
            })
        select = next((job for job in jobs if job["name"] == "select"), None)
        selected_sha = None
        if select is not None:
            try:
                selected_sha = _checkout_from_step(select, text(f"repos/{REPO}/actions/jobs/{select['id']}/logs"),
                                                   "Checkout the PR's test merge commit")
            except subprocess.CalledProcessError:
                pass
        if selected_sha:
            evidence.append({
                "sha": selected_sha, "job": "select", "job_id": select["id"], "run_attempt": run["run_attempt"],
                "job_attempt": select.get("run_attempt", 1), "source": "selected test merge checkout log", "url":
                f"{run['url']}/job/{select['id']}"
            })
        run["tested_sha"] = merge if merge and selected_sha == merge else None
        run["tested_sha_provenance"] = evidence
        run["stage_provenance"] = {"gpu-tests.yml": {"sha": run["tested_sha"], "evidence": evidence}}
        return

    run["provenance_version"] = 1
    run["run_attempt"] = run.get("run_attempt", 1)
    stages = {}
    for view in rules.stage_runs(run | {"jobs": jobs}):
        evidence = []
        restarted = [
            job for _, job in rules._test_jobs(view)
            if run["run_attempt"] > 1 and job.get("run_attempt") == run["run_attempt"] and job["conclusion"] in (
                "success", "failure")
        ]
        required = [
            job for job in view["jobs"]
            if job["name"] in ("select", "build", "build / build") and job["conclusion"] in ("success", "failure")
        ]
        for job in [*required, *restarted]:
            try:
                sha = logs.checkout_sha(job, text(f"repos/{REPO}/actions/jobs/{job['id']}/logs"))
            except subprocess.CalledProcessError:
                sha = None
            if sha:
                evidence.append({
                    "sha": sha, "job": job["name"], "job_id": job["id"], "run_attempt": run["run_attempt"],
                    "job_attempt": job.get("run_attempt",
                                           1), "source": "source checkout log", "url": f"{run['url']}/job/{job['id']}"
                })
        shas = {item["sha"] for item in evidence}
        complete = (any(item["job"] == "select" for item in evidence)
                    and any(item["job"].startswith("build") for item in evidence)
                    and len(evidence) == len(required) + len(restarted) and len(shas) == 1)
        stages[view["wf"]] = {"sha": next(iter(shas)) if complete else None, "evidence": evidence}
    run["stage_provenance"] = stages
    identity = stages[run["wf"]]
    run["tested_sha"], run["tested_sha_provenance"] = identity["sha"], identity["evidence"]


def _q(ref):
    return urllib.parse.quote(ref, safe="")


def _meta(get, ref):
    try:
        data = get(f"repos/{REPO}/contents/{META_PATH}?ref={_q(ref)}")
    except subprocess.CalledProcessError:  # no cmake/meta-llvm-info.json at ref
        return None
    meta = json.loads(base64.b64decode(data["content"]))
    return {
        "tag": meta["release_base_url"].rsplit("/", 1)[-1], "llvm_hash": meta["llvm_hash"], "base":
        meta.get("based_on_oai_llvm_hash")
    }


def _pr_of(title):
    m = re.search(r"\s*\(#(\d+)\)\s*$", title)
    return (title[:m.start()], int(m.group(1))) if m else (title, None)


def fork_changes(get, old, new):
    """The fork commits new adds over old, oldest first, without ShipIt's empty sync commits; None if unknown."""
    try:
        commits = get(f"repos/{FORK}/compare/{old}...{new}")["commits"]
    except subprocess.CalledProcessError:
        return None
    out = []
    for c in commits:
        title, pr = _pr_of(c["commit"]["message"].splitlines()[0])
        if not title.startswith("fbshipit-source-id:"):
            out.append({"sha": c["sha"], "title": title, "pr": pr})
    return out


def snapshot_line(get, target, fork_branch, releases, issues):
    """One tracked branch's Meta LLVM pin: what it pins, the newest fork release it could take and what that adds,
    the open bump PR and issues, and the pin's recent changes."""
    pinned = _meta(get, target)
    commits = get(f"repos/{FORK}/commits?sha={_q(fork_branch)}&per_page=30")
    release = mlb.release_for(mlb.same_tree_chain([(c["sha"], c["commit"]["tree"]["sha"]) for c in commits]), releases)
    newest = {
        "tag": release["tag_name"], "llvm_hash": release["target_commitish"], "published_at":
        release.get("published_at")
    } if release else None
    pending = []
    if pinned and newest and pinned["tag"] != newest["tag"]:
        pending = fork_changes(get, pinned["llvm_hash"], newest["llvm_hash"]) or []
    branch = mlb.bot_branch_for(target)
    prs = get(f"repos/{REPO}/pulls?head={REPO.split('/')[0]}:{_q(branch)}&state=open")
    pr = {
        "number": prs[0]["number"], "title": prs[0]["title"], "url": prs[0]["html_url"], "author":
        prs[0]["user"]["login"], "created_at": prs[0].get("created_at")
    } if prs else None
    marker = mlb.issue_marker(target)
    open_issues = [{"number": i["number"], "title": i["title"], "url": i["html_url"], "created_at": i["created_at"]}
                   for i in issues
                   if i["state"] == "open" and marker in (i.get("body") or "")]
    # The pin's changes: commits that touched the file and moved llvm_hash, each with the hash it replaced.
    touched = get(f"repos/{REPO}/commits?sha={_q(target)}&path={META_PATH}&per_page={PIN_HISTORY + 1}")
    metas = [_meta(get, c["sha"]) for c in touched]
    history = []
    for i, c in enumerate(touched):
        before = metas[i + 1] if i + 1 < len(touched) else None
        if metas[i] is None or before and before["llvm_hash"] == metas[i]["llvm_hash"]:
            continue
        title, pr_number = _pr_of(c["commit"]["message"].splitlines()[0])
        history.append({
            "sha": c["sha"], "t": c["commit"]["committer"]["date"], "title": title, "pr": pr_number, "tag":
            metas[i]["tag"], "llvm_hash": metas[i]["llvm_hash"], "prev_hash": before["llvm_hash"] if before else None
        })
    return {
        "fork_branch": fork_branch, "pinned": pinned, "newest": newest, "pending": pending, "pr": pr, "issues":
        open_issues, "history": history[:PIN_HISTORY]
    }


def snapshot_bumps(get):
    releases = get(f"repos/{FORK}/releases?per_page=100")
    q = urllib.parse.quote(f'repo:{REPO} is:issue is:open in:title "[Meta LLVM]"')
    issues = get(f"search/issues?q={q}&sort=created&order=desc&per_page=50")["items"]
    return {
        "lines": {
            t: snapshot_line(get, t, fb, releases, issues)
            for t, fb in FORK_BRANCHES.items()
            if t in rules.TRACKED_BRANCHES
        }
    }


def _write_if_changed(path, data):
    body = json.dumps(data, indent=1, sort_keys=True)
    if os.path.exists(path) and open(path).read() == body:
        return False
    with open(path, "w") as fh:
        fh.write(body)
    return True


def _remaining(api):
    return api("rate_limit")[0]["resources"]["core"]["remaining"]


def collect(data_dir, since, now, api=gh_api, text=gh_text, get=gh_get, download=gh_bytes):
    state_path = os.path.join(data_dir, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    if since is None:
        since = (rules.iso(rules.t(state["cursor"]) - dt.timedelta(hours=1))
                 if state.get("cursor") else rules.iso(rules.t(now) - dt.timedelta(days=BACKFILL_DAYS)))
    runs = load_runs(data_dir)
    enrich = set()
    sanitizer_retry = {
        r["id"]
        for r in runs.values()
        if r.get("wf") == sanitizer.WORKFLOW and (r.get("sanitizer") or {}).get("retry")
        and r["created_at"] >= rules.iso(rules.t(now) - dt.timedelta(days=BACKFILL_DAYS))
    }
    recent = rules.iso(rules.t(now) - dt.timedelta(days=7))
    for wf in COLLECT:
        event = "&event=push" if wf == rules.LIT_WF else ""
        # Reruns keep their original created_at. Re-list only the comparison window;
        # unchanged completed attempts still need no jobs or logs.
        listed_since = min(since, rules.iso(rules.t(now) - dt.timedelta(days=7))) if wf in ("gpu-tests.yml",
                                                                                            rules.FAST_WF) else since
        if wf == sanitizer.WORKFLOW:
            listed_since = min(since, recent) if state.get("sanitizer_collected") else rules.iso(
                rules.t(now) - dt.timedelta(days=BACKFILL_DAYS))
        for page in api(f"repos/{REPO}/actions/workflows/{wf}/runs?created=>={listed_since}{event}&per_page=100"):
            for run in page["workflow_runs"]:
                old = runs.get(run["id"])
                same_attempt = old and old.get("run_attempt", 1) == run.get("run_attempt", 1)
                comparison = (wf == rules.FAST_WF and run["event"] == "pull_request") or (
                    wf == "gpu-tests.yml" and run["event"] in ("pull_request", "workflow_run"))
                if old and old["status"] == "completed" and old["jobs"] is not None:
                    if not comparison and (wf != sanitizer.WORKFLOW or same_attempt):
                        continue
                    if same_attempt:
                        if old.get("provenance_version") or run["created_at"] < recent:
                            continue
                        # Enrich recent legacy PR records once, using their cached jobs.
                        old["run_attempt"] = run.get("run_attempt", 1)
                        old["pr"], old["author"] = _pr_and_author(run, api)
                        enrich.add(run["id"])
                        continue
                record = _record(run, wf)
                if comparison:
                    history = list((old or {}).get("attempts", []))
                    if old and not same_attempt and old.get("jobs") is not None:
                        history.append({k: v for k, v in old.items() if k != "attempts"})
                    history = [
                        item for item in history
                        if item["created_at"] >= rules.iso(rules.t(now) - dt.timedelta(days=30))
                    ]
                    if history:
                        record["attempts"] = history
                if wf == "gpu-tests.yml" or comparison:
                    record["pr"], record["author"] = _pr_and_author(run, api)
                runs[run["id"]] = record
    todo = sorted(
        (r for r in runs.values()
         if (r["status"] == "completed" and (r["jobs"] is None or r["id"] in enrich or r["id"] in sanitizer_retry)) or (
             r["wf"] == rules.FAST_WF and r["event"] == "pull_request" and r["status"] != "completed") or (
                 r["wf"] == "gpu-tests.yml" and r["event"] == "workflow_run" and r["status"] != "completed")),
        key=lambda r: r["created_at"])
    unfetched = []
    for i, r in enumerate(todo):
        if i % 50 == 0 and _remaining(api) < RATE_FLOOR:
            unfetched = todo[i:]
            break
        legacy = r["id"] in enrich
        if legacy or (r["id"] in sanitizer_retry and r.get("jobs") is not None):
            jobs = r["jobs"]
        else:
            rerun = ((r["wf"] == rules.FAST_WF and r["event"] == "pull_request") or
                     (r["wf"] == "gpu-tests.yml" and r["event"] in ("pull_request", "workflow_run"))) and r.get(
                         "run_attempt", 1) > 1
            jobs = _jobs(
                api(f"repos/{REPO}/actions/runs/{r['id']}/jobs?per_page=100&filter={'all' if rerun else 'latest'}"))
            if rerun:
                # A failed-jobs-only rerun retains successful jobs from older attempts.
                latest = {}
                for job in jobs:
                    previous = latest.get(job["name"])
                    if previous is None or (job["run_attempt"], job["id"]) > (previous["run_attempt"], previous["id"]):
                        latest[job["name"]] = job
                jobs = list(latest.values())
        # Failures, selection and revision evidence can need the same log: fetch it once.
        log = functools.lru_cache(maxsize=None)(text)
        if r["wf"] == "gpu-tests.yml" and r["event"] == "workflow_run":
            identity = _followup_identity(jobs, log, api)
            if identity:
                r.update(identity)
        if legacy:
            # Keep the full-only PR view's existing normalized failures unchanged.
            evidence_jobs = copy.deepcopy(jobs)
            _attach_failures(evidence_jobs, log, r["wf"])
            for job, evidence_job in zip(jobs, evidence_jobs):
                for step, evidence_step in zip(job["steps"], evidence_job["steps"]):
                    if "test_failures" in evidence_step:
                        step["test_failures"] = evidence_step["test_failures"]
        else:
            if r["wf"] in rules.GPU_RUN_WORKFLOWS:
                _attach_failures(jobs, log, r["wf"])
            if rules.slow_test_run(r):
                _attach_durations(jobs, log)
            if r["wf"] == rules.LIT_WF:
                _attach_lit_failures(jobs, log)
            if r["wf"] == "gpu-tests.yml":
                r["select"] = _selection(jobs, log)
        if r["wf"] == rules.FAST_WF and r["event"] == "pull_request":
            r["full_select"] = _selection([job for job in jobs if job["name"].startswith("full / ")], log)
        if ((r["wf"] == rules.FAST_WF and r["event"] == "pull_request")
                or (r["wf"] == "gpu-tests.yml" and r["event"] in ("pull_request", "workflow_run"))):
            _tested_revision(r, jobs, log)
        if r["wf"] == sanitizer.WORKFLOW:
            try:
                r["sanitizer"] = sanitizer.collect(r, api, download, REPO)
            except (subprocess.CalledProcessError, OSError, ValueError) as exc:
                r["sanitizer"] = {"tools": [], "missing": [{"gpu": "all", "reason": str(exc)}], "retry": True}
        r["jobs"] = jobs
    # A run still not completed after STUCK_DAYS (a nightly stuck in "pending") no longer holds the cursor back.
    stuck = rules.iso(rules.t(now) - dt.timedelta(days=STUCK_DAYS))
    pending = [r["created_at"] for r in runs.values() if r["status"] != "completed" and r["created_at"] >= stuck]
    pending += [r["created_at"] for r in unfetched]
    state = {"cursor": min(pending) if pending else now, "collected_at": now, "sanitizer_collected": True}
    changed = save_runs(data_dir, runs)
    # A failed snapshot keeps the previous bump.json, so the rest of the page still updates.
    try:
        bump_path = os.path.join(data_dir, "bump.json")
        if _write_if_changed(bump_path, snapshot_bumps(get)):
            changed.append(bump_path)
    except (subprocess.CalledProcessError, KeyError, ValueError) as exc:
        print(f"warning: no Meta LLVM bump snapshot ({exc})")
    with open(state_path, "w") as fh:
        json.dump(state, fh)
    with open(os.path.join(data_dir, "changed.txt"), "w") as fh:
        fh.write(" ".join(changed))
    print(f"{len(runs)} runs stored, {len(todo) - len(unfetched)} fetched, {len(unfetched)} left for the next run; "
          f"cursor {state['cursor']}")
    return state


def bench_events(bench_dir=None):
    """Per platform, the bench dashboard's per-night events. A failure here leaves the Nightly tab without perf."""
    try:
        summaries = bench.read_summaries(bench_dir) if bench_dir else bench.load_summaries()
        return {platform: bench.events(nights) for platform, nights in summaries.items()}
    except Exception as exc:  # noqa: BLE001 - the CI page must render even if the bench data cannot be read
        print(f"warning: no bench perf events ({exc})")
        return {}


def render(data_dir, outdir, bench_dir=None):
    state_path = os.path.join(data_dir, "state.json")
    asof = json.load(open(state_path))["collected_at"] if os.path.exists(state_path) else None
    if asof:
        bump_path = os.path.join(data_dir, "bump.json")
        bump = json.load(open(bump_path)) if os.path.exists(bump_path) else None
        data = rules.page_data(list(load_runs(data_dir).values()), asof, bench_events(bench_dir), bump)
    else:
        data = {"asof": None}
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "data.json"), "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    shutil.copyfile(TEMPLATE, os.path.join(outdir, "index.html"))
    print(f"rendered /ci/ as of {asof}: {os.path.getsize(os.path.join(outdir, 'data.json')) // 1024} KB of data")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("data_dir")
    c.add_argument("--since", help="re-collect runs created at or after this ISO time")
    r = sub.add_parser("render")
    r.add_argument("outdir")
    r.add_argument("--from", dest="data_dir", required=True)
    r.add_argument("--bench-from", dest="bench_dir", help="read tlx-ops-bench summaries from this directory")
    args = parser.parse_args()
    if args.cmd == "collect":
        collect(args.data_dir, args.since, rules.iso(dt.datetime.now(dt.timezone.utc)))
    else:
        render(args.data_dir, args.outdir, args.bench_dir)


if __name__ == "__main__":
    main()
