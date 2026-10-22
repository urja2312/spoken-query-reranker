"""Kaggle job: Part 3 of the reranker study (MODE set per kernel copy; see pipeline_part3.sh)."""

import glob
import os
import shutil
import subprocess
import tarfile

MODE = "C"   # C, E, H or S — see benchmark/pipeline_part3.sh
W = "/kaggle/working"
trees = glob.glob("/kaggle/input/**/repo/benchmark/pipeline_part3.sh", recursive=True)
archives = glob.glob("/kaggle/input/**/bundle.tar.gz", recursive=True)
if trees:
    shutil.copytree(trees[0].rsplit("/benchmark/", 1)[0], f"{W}/repo")
elif archives:
    with tarfile.open(archives[0]) as tar:
        tar.extractall(W, filter="data")
else:
    raise SystemExit("bundle not found under /kaggle/input")
for junk in glob.glob(f"{W}/repo/**/._*", recursive=True):
    os.remove(junk) if os.path.isfile(junk) else shutil.rmtree(junk, ignore_errors=True)
bench = f"{W}/repo/benchmark"
# Checkpoints live outside /kaggle/working (20 GB output cap); only results and logs are kept.
os.makedirs("/tmp/models", exist_ok=True)
shutil.rmtree(f"{bench}/models", ignore_errors=True)
os.symlink("/tmp/models", f"{bench}/models")


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    env = {**os.environ, "PY": "python -I", "MODE": MODE, "TOKENIZERS_PARALLELISM": "false"}
    proc = subprocess.Popen(["bash", "-c", cmd], cwd=bench, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait():
        raise SystemExit(f"command failed ({proc.returncode}): {cmd}")


sh("nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /kaggle/working /tmp | tail -2")
sh("pip install -q 'sentence-transformers==6.1.0' faster-whisper piper-tts soundfile 2>&1 | tail -2; "
   "python -c 'import sentence_transformers as s, faster_whisper as f;print(\"st\",s.__version__,\"fw\",f.__version__)'")
try:
    sh("mkdir -p logs; set -o pipefail; bash pipeline_part3.sh 2>&1 | tee -a logs/pipeline_part3.log")
finally:
    out = f"{W}/out"
    os.makedirs(out, exist_ok=True)
    for sub in ("results", "logs"):
        if os.path.isdir(f"{bench}/{sub}"):
            shutil.copytree(f"{bench}/{sub}", f"{out}/{sub}", dirs_exist_ok=True)
    keep = (glob.glob(f"{bench}/cache/*/results_*.json") + glob.glob(f"{bench}/cache/*/transcripts_*.json")
            + glob.glob(f"{bench}/cache/*/conditions*.json") + glob.glob(f"{bench}/cache/trainset*.json")
            + glob.glob(f"{bench}/cache/heysquad/*.json") + glob.glob(f"{bench}/cache/slue/whisper_real.json"))
    for path in keep:
        rel = os.path.relpath(path, bench)
        os.makedirs(os.path.dirname(f"{out}/{rel}"), exist_ok=True)
        shutil.copy(path, f"{out}/{rel}")
    for name in ("heysquad", "slue"):
        if os.path.isdir(f"{bench}/data/{name}"):
            shutil.copytree(f"{bench}/data/{name}", f"{out}/data/{name}", dirs_exist_ok=True)
    shutil.rmtree(f"{W}/repo", ignore_errors=True)
    print("\noutputs:", sorted(os.listdir(W)), flush=True)
