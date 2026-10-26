"""Run one batch TTS job on a Kaggle GPU kernel.

    python runner/run_job.py <work_dir> <out_dir>

work_dir holds job.json and optional refs/<name>.wav + refs/<name>.txt.
Logs are kept minimal on purpose.
"""
import base64
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

AQUI = pathlib.Path(__file__).resolve().parent
SLUG = "tts-batch"
TEMPO_MAX = 150 * 60


def sh(cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(">>", cmd[1], cmd[2], "rc", r.returncode, flush=True)
    if check and r.returncode != 0:
        raise SystemExit("command failed")
    return (r.stdout + r.stderr).lower()


def main(work, out):
    work, out = pathlib.Path(work), pathlib.Path(out)
    job = json.loads((work / "job.json").read_text(encoding="utf-8"))
    refs = {}
    for wav in sorted((work / "refs").glob("*.wav")) if (work / "refs").exists() else []:
        refs[wav.stem] = {"audio_b64": base64.b64encode(wav.read_bytes()).decode(),
                          "texto": wav.with_suffix(".txt").read_text(encoding="utf-8").strip()}
    user = os.environ["KAGGLE_USERNAME"].strip()

    k = pathlib.Path("/tmp/kernel")
    shutil.rmtree(k, ignore_errors=True)
    k.mkdir()
    code = (AQUI / "tts_job.py").read_text(encoding="utf-8")
    code = code.replace("__JOB_B64__", base64.b64encode(json.dumps(job).encode()).decode())
    code = code.replace("__REFS_B64__", base64.b64encode(json.dumps(refs).encode()).decode())
    (k / "job.py").write_text(code, encoding="utf-8")
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{SLUG}", "title": SLUG, "code_file": "job.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": job.get("maquina", "NvidiaTeslaT4"),
        "dataset_sources": [], "competition_sources": [], "kernel_sources": []}))

    sh(["kaggle", "kernels", "push", "-p", str(k)])
    t0, estado = time.time(), "tempo"
    time.sleep(60)
    while time.time() - t0 < TEMPO_MAX:
        s = sh(["kaggle", "kernels", "status", f"{user}/{SLUG}"], check=False)
        if "complete" in s:
            estado = "ok"
            break
        if "error" in s or "cancel" in s:
            estado = "erro"
            break
        time.sleep(45)
    out.mkdir(parents=True, exist_ok=True)
    sh(["kaggle", "kernels", "output", f"{user}/{SLUG}", "-p", str(out), "-o"], check=False)
    if estado != "ok" or not (out / "relatorio.json").exists():
        (out / "FALHOU.txt").write_text(f"estado={estado}\n")
    print("done", estado, int(time.time() - t0), "s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
