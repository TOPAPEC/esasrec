from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], api_key: str, insecure_ssl: bool, no_proxy: bool) -> str:
    env = os.environ.copy()
    env["VAST_API_KEY"] = api_key
    wrapper = [sys.executable, "scripts/vast_cli.py"]
    if no_proxy:
        wrapper.append("--no-proxy")
    if insecure_ssl:
        wrapper.append("--insecure")
    p = subprocess.run([*wrapper, *cmd], text=True, capture_output=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}")
    return p.stdout.strip()


def search_offer(api_key: str, insecure_ssl: bool, no_proxy: bool, gpu_name: str) -> dict:
    q = f"gpu_name={gpu_name} num_gpus=1 inet_down>300 inet_down_cost<0.0005 rentable=True"
    out = run(["search", "offers", "--raw", "--no-default", q, "-o", "dph"], api_key, insecure_ssl, no_proxy)
    data = json.loads(out)
    if not data:
        raise RuntimeError("no offers found for requested constraints")

    def price(x: dict) -> float:
        return float(x.get("dph_total", x.get("dph", 1e9)))

    return min(data, key=price)


def create_instance(api_key: str, ask_id: int, image: str, disk: int, insecure_ssl: bool, no_proxy: bool, onstart: str | None = None) -> int:
    cmd = ["create", "instance", str(ask_id), "--raw", "--image", image, "--disk", str(disk), "--ssh", "--direct", "--cancel-unavail"]
    if onstart:
        cmd += ["--onstart", onstart]
    out = run(cmd, api_key, insecure_ssl, no_proxy)
    data = json.loads(out)
    if not data.get("success"):
        raise RuntimeError(f"instance creation failed: {data}")
    return int(data["new_contract"])


def get_instance(api_key: str, instance_id: int, insecure_ssl: bool, no_proxy: bool) -> dict:
    out = run(["show", "instances", "--raw"], api_key, insecure_ssl, no_proxy)
    items = json.loads(out)
    for it in items:
        if int(it.get("id", -1)) == instance_id:
            return it
    raise RuntimeError(f"instance {instance_id} not found")


def wait_running(api_key: str, instance_id: int, timeout_s: int, insecure_ssl: bool, no_proxy: bool) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        it = get_instance(api_key, instance_id, insecure_ssl, no_proxy)
        if str(it.get("actual_status", "")).lower() == "running":
            return it
        time.sleep(10)
    raise TimeoutError("instance did not become running in time")


def copy_repo(api_key: str, instance_id: int, repo: Path, insecure_ssl: bool, no_proxy: bool) -> None:
    run(["copy", f"local:{repo}", f"C.{instance_id}:/workspace/esasrec"], api_key, insecure_ssl, no_proxy)


def execute(api_key: str, instance_id: int, command: str, insecure_ssl: bool, no_proxy: bool) -> str:
    return run(["execute", str(instance_id), command], api_key, insecure_ssl, no_proxy)


def get_logs(api_key: str, instance_id: int, insecure_ssl: bool, no_proxy: bool, tail: int = 300) -> str:
    return run(["logs", str(instance_id), "--tail", str(tail)], api_key, insecure_ssl, no_proxy)


def destroy(api_key: str, instance_id: int, insecure_ssl: bool, no_proxy: bool) -> None:
    run(["destroy", "instance", str(instance_id), "--raw"], api_key, insecure_ssl, no_proxy)


def write_onstart(args: argparse.Namespace) -> Path:
    max_eval_flag = f"--max-eval-users {args.max_eval_users}" if args.max_eval_users is not None else ""
    script = f'''#!/bin/bash
set -euo pipefail
mkdir -p /workspace /workspace/data /workspace/out
while [ ! -d /workspace/esasrec ]; do sleep 3; done
cd /workspace/esasrec
python -m pip install -q -r requirements.txt
if [ ! -f {args.ratings_path} ]; then
  curl -L {args.ratings_url} -o /workspace/data/ml-20m.zip
  unzip -o /workspace/data/ml-20m.zip -d /workspace/data
fi
python -c 'import torch; assert torch.cuda.is_available(); print("cuda_device", torch.cuda.get_device_name(0))'
python -m esasrec.train --ratings-path {args.ratings_path} --workdir {args.workdir_remote} --device cuda --epochs {args.epochs} --batch-size {args.batch_size} --max-len {args.max_len} --emb-dim {args.emb_dim} --n-heads {args.n_heads} --n-blocks {args.n_blocks} --n-neg {args.n_neg} --num-workers {args.num_workers} --amp --log-every {args.log_every} {max_eval_flag}
'''
    path = Path("/tmp/onstart_esasrec_ml20m.sh")
    path.write_text(script)
    path.chmod(0o755)
    return path


def parse_test_metrics(logs: str) -> dict[str, float] | None:
    marker = "test: recall@10="
    for line in logs.splitlines()[::-1]:
        if marker in line:
            tail = line.split(marker, 1)[1]
            rec_s, ndcg_s = tail.split(" ndcg@10=")
            return {"recall@10": float(rec_s.strip()), "ndcg@10": float(ndcg_s.strip())}
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", type=str, default=os.getenv("VAST_API_KEY", ""))
    p.add_argument("--repo", type=str, default="/workspace/esasrec")
    p.add_argument("--image", type=str, default="pytorch/pytorch")
    p.add_argument("--gpu-name", type=str, default="RTX_4090")
    p.add_argument("--disk", type=int, default=60)
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--keep", action="store_true")
    p.add_argument("--insecure-ssl", action="store_true")
    p.add_argument("--no-proxy", action="store_true")
    p.add_argument("--search-only", action="store_true")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--emb-dim", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--n-neg", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-eval-users", type=int, default=None)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ratings-url", type=str, default="https://files.grouplens.org/datasets/movielens/ml-20m.zip")
    p.add_argument("--ratings-path", type=str, default="/workspace/data/ml-20m/ratings.csv")
    p.add_argument("--workdir-remote", type=str, default="/workspace/out")
    args = p.parse_args()

    if not args.api_key:
        raise RuntimeError("api key is required")

    offer = search_offer(args.api_key, args.insecure_ssl, args.no_proxy, args.gpu_name)
    ask_id = int(offer["id"])
    print(json.dumps({"selected_offer": {"id": ask_id, "gpu_name": offer.get("gpu_name"), "dph": offer.get("dph_total", offer.get("dph")), "inet_down": offer.get("inet_down"), "inet_down_cost": offer.get("inet_down_cost"), "reliability": offer.get("reliability")}}, indent=2))
    if args.search_only:
        return

    onstart = write_onstart(args)
    instance_id = create_instance(args.api_key, ask_id, args.image, args.disk, args.insecure_ssl, args.no_proxy, str(onstart))
    print(json.dumps({"instance_id": instance_id}))
    try:
        inst = wait_running(args.api_key, instance_id, args.timeout, args.insecure_ssl, args.no_proxy)
        print(json.dumps({"running": {"id": instance_id, "ssh_host": inst.get("ssh_host"), "ssh_port": inst.get("ssh_port")}}, indent=2))
        copy_repo(args.api_key, instance_id, Path(args.repo), args.insecure_ssl, args.no_proxy)

        self_check = execute(args.api_key, instance_id, "ls -l /workspace", args.insecure_ssl, args.no_proxy)
        print(f"self_check_ok=1\n{self_check}")

        t0 = time.time()
        last_logs = ""
        while time.time() - t0 < 60 * 90:
            logs = get_logs(args.api_key, instance_id, args.insecure_ssl, args.no_proxy, 400)
            if logs != last_logs:
                print(logs[-4000:])
                last_logs = logs
            if "train_finished" in logs:
                metrics = parse_test_metrics(logs)
                if metrics:
                    ref = {"recall@10": 0.329, "ndcg@10": 0.197}
                    print(json.dumps({
                        "test_metrics": metrics,
                        "article_esasrec_ml20m": ref,
                        "delta": {k: round(metrics[k] - ref[k], 6) for k in ref},
                    }, indent=2))
                break
            time.sleep(30)
        else:
            raise TimeoutError("training did not finish in 90 minutes")
    finally:
        if not args.keep:
            destroy(args.api_key, instance_id, args.insecure_ssl, args.no_proxy)
            print(json.dumps({"destroyed": instance_id}))


if __name__ == "__main__":
    main()
