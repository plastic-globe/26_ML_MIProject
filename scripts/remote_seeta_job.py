from __future__ import annotations

import argparse
import os
import posixpath
import stat
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / ".vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import paramiko


HOST = "connect.westb.seetacloud.com"
PORT = 20963
USER = "root"
REMOTE_ROOT = "/root/MI_project"
REMOTE_PYTHON = "/root/miniconda3/bin/python3"


def connect() -> paramiko.SSHClient:
    password = os.environ.get("SEETA_PASSWORD")
    if not password:
        raise SystemExit("SEETA_PASSWORD is not set")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        port=PORT,
        username=USER,
        password=password,
        look_for_keys=False,
        allow_agent=False,
        timeout=30,
    )
    return client


def run(client: paramiko.SSHClient, command: str, timeout: int | None = None) -> int:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    del stdin
    for line in iter(stdout.readline, ""):
        print(line, end="")
    err = stderr.read().decode("utf-8", errors="replace")
    if err:
        print(err, end="", file=sys.stderr)
    return stdout.channel.recv_exit_status()


def mkdir_p(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = []
    cur = remote_dir
    while cur not in ("", "/"):
        parts.append(cur)
        cur = posixpath.dirname(cur)
    for path in reversed(parts):
        try:
            sftp.stat(path)
        except FileNotFoundError:
            sftp.mkdir(path)


def upload_file(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    mkdir_p(sftp, posixpath.dirname(remote))
    print(f"upload {local} -> {remote}")
    sftp.put(str(local), remote)


def upload_tree(sftp: paramiko.SFTPClient, local_dir: Path, remote_dir: str) -> None:
    mkdir_p(sftp, remote_dir)
    for path in local_dir.rglob("*"):
        rel = path.relative_to(local_dir).as_posix()
        remote = posixpath.join(remote_dir, rel)
        if path.is_dir():
            mkdir_p(sftp, remote)
        else:
            upload_file(sftp, path, remote)


def download_tree(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for item in sftp.listdir_attr(remote_dir):
        remote_path = posixpath.join(remote_dir, item.filename)
        local_path = local_dir / item.filename
        if stat.S_ISDIR(item.st_mode):
            download_tree(sftp, remote_path, local_path)
        else:
            print(f"download {remote_path} -> {local_path}")
            sftp.get(remote_path, str(local_path))


def setup(client: paramiko.SSHClient) -> None:
    commands = [
        f"mkdir -p {REMOTE_ROOT}/outputs/code {REMOTE_ROOT}/outputs/qwen_cpu_mmlu3000_locate_steer_improve /root/.cache/huggingface",
        f"{REMOTE_PYTHON} --version",
        (
            "cd /root && "
            f"{REMOTE_PYTHON} -m pip install --upgrade --quiet "
            "'transformers>=4.57' pandas tqdm accelerate safetensors sentencepiece"
        ),
    ]
    for command in commands:
        code = run(client, command, timeout=600)
        if code != 0:
            raise SystemExit(f"remote command failed with exit code {code}: {command}")


def upload(client: paramiko.SSHClient, input_file: str, output_dir: str) -> None:
    local_input = ROOT / input_file
    local_output_dir = ROOT / output_dir
    with client.open_sftp() as sftp:
        upload_file(sftp, local_input, f"{REMOTE_ROOT}/{Path(input_file).as_posix()}")
        upload_file(
            sftp,
            ROOT / "outputs" / "code" / "run_qwen3000_cpu_suite.py",
            f"{REMOTE_ROOT}/outputs/code/run_qwen3000_cpu_suite.py",
        )
        if local_output_dir.exists():
            upload_tree(sftp, local_output_dir, f"{REMOTE_ROOT}/{Path(output_dir).as_posix()}")
        else:
            mkdir_p(sftp, f"{REMOTE_ROOT}/{Path(output_dir).as_posix()}")


def upload_script(client: paramiko.SSHClient) -> None:
    with client.open_sftp() as sftp:
        upload_file(
            sftp,
            ROOT / "outputs" / "code" / "run_qwen3000_cpu_suite.py",
            f"{REMOTE_ROOT}/outputs/code/run_qwen3000_cpu_suite.py",
        )


def start(
    client: paramiko.SSHClient,
    device: str,
    input_file: str,
    output_dir: str,
    max_examples: int,
    batch_size: int,
    patch_batch_size: int,
    train_size: int,
    layers: str,
    alphas: str,
    mitigation_modes: str,
    stages: str,
    model_alias: str,
    model_name: str,
) -> None:
    log_name = Path(output_dir).name + "_full_run.log"
    model_name_arg = f"--model_name {model_name} " if model_name else ""
    command = (
        f"cd {REMOTE_ROOT} && "
        "export HF_HOME=/root/.cache/huggingface && "
        "export HF_ENDPOINT=https://hf-mirror.com && "
        f"setsid {REMOTE_PYTHON} outputs/code/run_qwen3000_cpu_suite.py "
        f"--model_alias {model_alias} "
        f"{model_name_arg}"
        f"--input_file {input_file} "
        f"--output_dir {output_dir} "
        f"--max_examples {max_examples} "
        f"--batch_size {batch_size} "
        f"--patch_batch_size {patch_batch_size} "
        f"--train_size {train_size} "
        f"--device {device} "
        f"--layers {layers} "
        f"--alphas={alphas} "
        f"--mitigation_modes {mitigation_modes} "
        f"--stages {stages} "
        f"> {log_name} 2>&1 < /dev/null & echo $!"
    )
    code = run(client, command, timeout=30)
    if code != 0:
        raise SystemExit(f"failed to start remote job: {code}")


def status(client: paramiko.SSHClient, output_dir: str) -> None:
    log_name = Path(output_dir).name + "_full_run.log"
    command = (
        f"cd {REMOTE_ROOT} && "
        "echo '--- ps ---' && "
        "ps -eo pid,etime,cmd | grep run_qwen3000_cpu_suite.py | grep -v grep || true && "
        "echo '--- log tail ---' && "
        f"tail -n 80 {log_name} 2>/dev/null || true && "
        "echo '--- outputs ---' && "
        f"ls -lh {output_dir} 2>/dev/null | tail -n 40"
    )
    code = run(client, command, timeout=60)
    if code != 0:
        raise SystemExit(f"status failed: {code}")


def download(client: paramiko.SSHClient, output_dir: str) -> None:
    log_name = Path(output_dir).name + "_full_run.log"
    with client.open_sftp() as sftp:
        download_tree(
            sftp,
            f"{REMOTE_ROOT}/{Path(output_dir).as_posix()}",
            ROOT / output_dir,
        )
        try:
            sftp.get(f"{REMOTE_ROOT}/{log_name}", str(ROOT / "outputs" / log_name))
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["setup", "upload", "upload-script", "start", "status", "download", "exec", "all"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--command", default="")
    parser.add_argument("--input-file", default="outputs/qwen_mmlu_3000.csv")
    parser.add_argument("--output-dir", default="outputs/qwen_cpu_mmlu3000_locate_steer_improve")
    parser.add_argument("--max-examples", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patch-batch-size", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--layers", default="20,21,22,23")
    parser.add_argument("--alphas", default="-4,-2,-1,0,1,2,4")
    parser.add_argument("--mitigation-modes", default="none,truth_priority,anti_sycophancy,verify_then_answer,counter_opinion_check")
    parser.add_argument("--stages", default="behavior,logit_lens,steer,patching,improve,elimination")
    parser.add_argument("--model-alias", default="qwen2.5-0.5b")
    parser.add_argument("--model-name", default="")
    args = parser.parse_args()

    client = connect()
    try:
        if args.action in {"setup", "all"}:
            setup(client)
        if args.action in {"upload", "all"}:
            upload(client, args.input_file, args.output_dir)
        if args.action == "upload-script":
            upload_script(client)
        if args.action in {"start", "all"}:
            start(
                client,
                args.device,
                args.input_file,
                args.output_dir,
                args.max_examples,
                args.batch_size,
                args.patch_batch_size,
                args.train_size,
                args.layers,
                args.alphas,
                args.mitigation_modes,
                args.stages,
                args.model_alias,
                args.model_name,
            )
        if args.action == "status":
            status(client, args.output_dir)
        if args.action == "download":
            download(client, args.output_dir)
        if args.action == "exec":
            if not args.command:
                raise SystemExit("--command is required for exec")
            code = run(client, args.command, timeout=300)
            if code != 0:
                raise SystemExit(f"remote exec failed: {code}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
