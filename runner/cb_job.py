# Batch TTS job with the Chatterbox language pack (runs inside a Kaggle GPU kernel).
# Placeholders are filled by run_job.py. Each phase runs in its own process.
# A: Chatterbox for voices that already have a reference.
# B: (only if needed) Qwen3 VoiceDesign creates references for new voices + "design" lines.
# C: Chatterbox for the new voices; D: Qwen clone fallback for anything still missing.
# E: transcription (word timings + error rate) and report.
import base64
import json
import os
import subprocess
import sys
import time

JOB = json.loads(base64.b64decode("__JOB_B64__"))
REFS = json.loads(base64.b64decode("__REFS_B64__"))
OUT = "/kaggle/working"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def sh(c):
    log(">>", c)
    return subprocess.run(c, shell=True).returncode


def fase(nome, codigo):
    p = f"/tmp/{nome}.py"
    open(p, "w").write(codigo)
    return sh(f"{sys.executable} {p}")


json.dump(JOB, open("/tmp/job.json", "w"))
os.makedirs("/tmp/refs", exist_ok=True)
import soundfile as sf  # noqa: E402
import io  # noqa: E402
for nome, r in REFS.items():
    x, sr = sf.read(io.BytesIO(base64.b64decode(r["audio_b64"])), dtype="float32")
    sf.write(f"/tmp/refs/{nome}.wav", x, sr)
    open(f"/tmp/refs/{nome}.txt", "w").write(r["texto"])

FALAS = JOB["falas"]
TAKES = int(JOB.get("takes", 3))
FALTANDO = sorted({f["quem"] for f in FALAS if f.get("modo", "clone") == "clone" and f["quem"] not in REFS})
PRECISA_QWEN = bool(FALTANDO) or any(f.get("modo") == "design" for f in FALAS)

sh("pip install -q --no-deps 'git+https://github.com/resemble-ai/chatterbox.git@master' 2>&1 | tail -3")
sh("pip install -q librosa s3tokenizer conformer==0.3.2 omegaconf pyloudnorm pykakasi spacy-pkuseg "
   "faster-whisper 'git+https://github.com/resemble-ai/Perth.git@master' 2>&1 | tail -3")

CB = r'''
import json, os, sys, shutil, torch, torchaudio as ta
from huggingface_hub import snapshot_download
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
job = json.load(open("/tmp/job.json"))
so = set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else None
base = snapshot_download("ResembleAI/chatterbox")
extra = snapshot_download(job.get("chatterbox_pack", "ResembleAI/Chatterbox-Multilingual-pt-br"))
merged = "/tmp/cb"
shutil.copytree(base, merged, dirs_exist_ok=True)
shutil.copytree(extra, merged, dirs_exist_ok=True)
sts = [f for f in os.listdir(extra) if f.endswith(".safetensors")]
t3 = max(sts, key=lambda f: os.path.getsize(os.path.join(extra, f)))
model = ChatterboxMultilingualTTS.from_local(merged, "cuda", t3_model=t3)
ex0, cfg0 = job.get("exagero", 0.5), job.get("cfg", 0.5)
for f in job["falas"]:
    if f.get("modo", "clone") != "clone":
        continue
    if so is not None and f["quem"] not in so:
        continue
    ref = f"/tmp/refs/{f['quem']}.wav"
    if not os.path.exists(ref):
        continue
    for k in range(int(job.get("takes", 3))):
        torch.manual_seed(2000 + k)
        wav = model.generate(f["texto"], language_id="pt", audio_prompt_path=ref,
                             exaggeration=float(f.get("exagero", ex0)), cfg_weight=float(f.get("cfg", cfg0)))
        ta.save(f"/kaggle/working/{f['id']}_t{k + 1}.wav", wav, model.sr)
    print("ok", f["id"], flush=True)
'''

QWEN = r'''
import json, torch, numpy as np, soundfile as sf
from qwen_tts import Qwen3TTSModel
job = json.load(open("/tmp/job.json"))
faltando = json.load(open("/tmp/faltando.json"))
pers = job.get("personagens", {})
REF_TXT = "Olha, eu vou te falar uma coisa, e você vai me ouvir com atenção, tá bom? Porque eu não vou repetir."
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cuda:0",
                                  dtype=torch.float32, attn_implementation="sdpa")
novas = []
for quem in faltando:
    torch.manual_seed(7)
    wavs, sr = m.generate_voice_design(text=REF_TXT, language="Portuguese", instruct=pers[quem]["descricao"])
    w = np.asarray(wavs[0], dtype=np.float32)
    sf.write(f"/kaggle/working/novo_ref_{quem}.wav", w, sr)
    sf.write(f"/tmp/refs/{quem}.wav", w, sr)
    open(f"/tmp/refs/{quem}.txt", "w").write(REF_TXT)
    novas.append({"quem": quem, "arquivo": f"novo_ref_{quem}.wav", "texto": REF_TXT})
for f in job["falas"]:
    if f.get("modo") != "design":
        continue
    instr = (pers.get(f["quem"], {}).get("descricao", "") + " " + f.get("emocao", "")).strip()
    for k in range(int(job.get("takes", 3))):
        torch.manual_seed(2000 + k)
        wavs, sr = m.generate_voice_design(text=f["texto"], language="Portuguese", instruct=instr)
        sf.write(f"/kaggle/working/{f['id']}_t{k + 1}.wav", np.asarray(wavs[0], dtype=np.float32), sr)
json.dump(novas, open("/tmp/novas.json", "w"))
'''

FALLBACK = r'''
import json, os, torch, numpy as np, soundfile as sf
from qwen_tts import Qwen3TTSModel
job = json.load(open("/tmp/job.json"))
m = None
for f in job["falas"]:
    if f.get("modo", "clone") != "clone":
        continue
    for k in range(int(job.get("takes", 3))):
        out = f"/kaggle/working/{f['id']}_t{k + 1}.wav"
        if os.path.exists(out):
            continue
        if m is None:
            m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0",
                                              dtype=torch.float32, attn_implementation="sdpa")
            prompts = {}
        if f["quem"] not in prompts:
            a, sr = sf.read(f"/tmp/refs/{f['quem']}.wav", dtype="float32")
            prompts[f["quem"]] = m.create_voice_clone_prompt(
                ref_audio=(a, sr), ref_text=open(f"/tmp/refs/{f['quem']}.txt").read())
        torch.manual_seed(2000 + k)
        wavs, sr = m.generate_voice_clone(text=f["texto"], language="Portuguese", voice_clone_prompt=prompts[f["quem"]])
        sf.write(out, np.asarray(wavs[0], dtype=np.float32), sr)
        print("fallback qwen", f["id"], k + 1, flush=True)
'''

WHISPER = r'''
import json, os, re, unicodedata, subprocess, numpy as np, soundfile as sf
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
job = json.load(open("/tmp/job.json"))
try:
    w = WhisperModel("medium", device="cuda", compute_type="float16")
except Exception:
    w = WhisperModel("small", device="cpu", compute_type="int8")
def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", t).split()
def erro(ref, hip):
    a, b = norm(ref), norm(hip)
    if not a:
        return 0.0
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]; d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev = cur
    return round(d[len(b)] / len(a), 3)
itens = []
for f in job["falas"]:
    for k in range(int(job.get("takes", 3))):
        nome = f"{f['id']}_t{k + 1}.wav"
        p = os.path.join("/kaggle/working", nome)
        if not os.path.exists(p):
            continue
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1: x = x.mean(axis=1)
        g = np.gcd(int(sr), 16000); x16 = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
        segs, _ = w.transcribe(x16, language="pt", word_timestamps=True, vad_filter=False)
        hip, pal = [], []
        for s in segs:
            hip.append(s.text)
            for wd in (s.words or []):
                pal.append([wd.word.strip(), round(wd.start, 3), round(wd.end, 3)])
        hip = " ".join(hip).strip()
        itens.append({"fala": f["id"], "quem": f["quem"], "take": k + 1, "arquivo": nome,
                      "duracao": round(len(x) / sr, 2), "texto": f["texto"], "transcricao": hip,
                      "erro": erro(f["texto"], hip), "palavras": pal})
        subprocess.run(f"ffmpeg -loglevel error -y -i {p} -b:a 128k {p[:-4]}.mp3", shell=True)
novas = json.load(open("/tmp/novas.json")) if os.path.exists("/tmp/novas.json") else []
json.dump({"tipo": "episodio", "motor": "chatterbox", "itens": itens, "novas_vozes": novas},
          open("/kaggle/working/relatorio.json", "w"), ensure_ascii=False, indent=1)
print("itens", len(itens), flush=True)
'''

log("fase A: chatterbox")
fase("cb", CB)
if PRECISA_QWEN:
    json.dump(FALTANDO, open("/tmp/faltando.json", "w"))
    sh("pip install -q qwen-tts 2>&1 | tail -2")
    log("fase B: vozes novas")
    fase("qwen", QWEN)
    if FALTANDO:
        log("fase C: chatterbox nas vozes novas")
        p = "/tmp/cb.py"
        sh(f"{sys.executable} {p} {','.join(FALTANDO)}")
faltam = [f for f in FALAS if f.get("modo", "clone") == "clone"
          and not os.path.exists(f"{OUT}/{f['id']}_t1.wav")]
if faltam:
    log("fase D: reserva qwen para", len(faltam), "falas")
    sh("pip install -q qwen-tts 2>&1 | tail -2")
    fase("fallback", FALLBACK)
log("fase E: transcrição")
fase("whisper", WHISPER)
log("fim")
