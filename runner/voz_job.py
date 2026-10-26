# Voice comparison job (runs inside a Kaggle GPU kernel). Placeholders are filled by run_job.py.
# Phase 1: Qwen3-TTS voice design candidates. Phase 2: Chatterbox (language pack) cloning.
# Phase 3: transcription of every file (clarity score).
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
    subprocess.run(c, shell=True)


json.dump(JOB, open("/tmp/job.json", "w"))
os.makedirs("/tmp/refs", exist_ok=True)
for nome, r in REFS.items():
    open(f"/tmp/refs/{nome}.ogg", "wb").write(base64.b64decode(r["audio_b64"]))
    open(f"/tmp/refs/{nome}.txt", "w").write(r["texto"])

# ---------------- fase 1: Qwen3 VoiceDesign ----------------
sh("pip install -q -U qwen-tts soundfile 2>&1 | tail -2")
FASE1 = r'''
import json, torch, soundfile as sf, numpy as np, subprocess
from qwen_tts import Qwen3TTSModel
job = json.load(open("/tmp/job.json"))
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cuda:0",
                                  dtype=torch.float32, attn_implementation="sdpa")
for c in job.get("design", []):
    torch.manual_seed(c.get("seed", 7))
    wavs, sr = m.generate_voice_design(text=c["ref_texto"], language="Portuguese", instruct=c["descricao"])
    sf.write(f"/kaggle/working/{c['id']}_ref.wav", np.asarray(wavs[0], dtype=np.float32), sr)
    open(f"/tmp/refs/{c['id']}.txt", "w").write(c["ref_texto"])
    sf.write(f"/tmp/refs/{c['id']}.wav", np.asarray(wavs[0], dtype=np.float32), sr)
    for k, t in enumerate(c["falas"]):
        torch.manual_seed(100 + k)
        wavs, sr = m.generate_voice_design(text=t, language="Portuguese", instruct=c["descricao"])
        sf.write(f"/kaggle/working/qwen_{c['id']}_{k + 1}.wav", np.asarray(wavs[0], dtype=np.float32), sr)
    print("qwen ok", c["id"], flush=True)
'''
open("/tmp/fase1.py", "w").write(FASE1)
sh(f"{sys.executable} /tmp/fase1.py")

# ---------------- fase 2: Chatterbox pt-BR ----------------
sh("pip install -q chatterbox-tts 2>&1 | tail -3")
FASE2 = r'''
import json, os, shutil, glob, torch, torchaudio as ta, soundfile as sf
from huggingface_hub import snapshot_download
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
job = json.load(open("/tmp/job.json"))
pack = job.get("chatterbox_pack", "ResembleAI/Chatterbox-Multilingual-pt-br")
base = snapshot_download("ResembleAI/chatterbox")
extra = snapshot_download(pack)
print("arquivos do pack:", os.listdir(extra), flush=True)
for f in glob.glob(os.path.join(extra, "*.md")):
    print(open(f).read()[:1500], flush=True)
merged = "/tmp/cb"
shutil.copytree(base, merged, dirs_exist_ok=True)
shutil.copytree(extra, merged, dirs_exist_ok=True)
sts = [f for f in os.listdir(extra) if f.endswith(".safetensors")]
t3 = max(sts, key=lambda f: os.path.getsize(os.path.join(extra, f))) if sts else None
print("t3:", t3, flush=True)
try:
    model = ChatterboxMultilingualTTS.from_local(merged, "cuda", t3_model=t3)
except Exception as e:
    print("pack falhou, usando v3:", e, flush=True)
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")
# referências: vozes atuais (.ogg) e vozes novas do Qwen (.wav)
for c in job["clonar"]:
    ref = c["ref"]
    cand = [f"/tmp/refs/{ref}.wav", f"/tmp/refs/{ref}.ogg"]
    caminho = next((p for p in cand if os.path.exists(p)), None)
    if caminho is None:
        print("sem ref", ref, flush=True)
        continue
    if caminho.endswith(".ogg"):
        x, sr = sf.read(caminho, dtype="float32")
        caminho = caminho[:-4] + "_conv.wav"
        sf.write(caminho, x, sr)
    for k, t in enumerate(c["falas"]):
        torch.manual_seed(200 + k)
        wav = model.generate(t, language_id="pt", audio_prompt_path=caminho,
                             exaggeration=c.get("exaggeration", 0.5), cfg_weight=c.get("cfg", 0.5))
        ta.save(f"/kaggle/working/cb_{c['id']}_{k + 1}.wav", wav, model.sr)
    print("chatterbox ok", c["id"], flush=True)
'''
open("/tmp/fase2.py", "w").write(FASE2)
sh(f"{sys.executable} /tmp/fase2.py")

# ---------------- fase 3: transcrição (clareza) ----------------
sh("pip install -q faster-whisper 2>&1 | tail -2")
FASE3 = r'''
import json, glob, os, re, unicodedata, numpy as np, soundfile as sf
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
job = json.load(open("/tmp/job.json"))
textos = {}
for c in job.get("design", []):
    for k, t in enumerate(c["falas"]):
        textos[f"qwen_{c['id']}_{k + 1}.wav"] = t
for c in job["clonar"]:
    for k, t in enumerate(c["falas"]):
        textos[f"cb_{c['id']}_{k + 1}.wav"] = t
try:
    w = WhisperModel("medium", device="cuda", compute_type="float16")
except Exception:
    w = WhisperModel("small", device="cpu", compute_type="int8")
def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", t).split()
def wer(a, b):
    a, b = norm(a), norm(b)
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]; d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev = cur
    return round(d[len(b)] / max(1, len(a)), 3)
itens = []
for nome, texto in sorted(textos.items()):
    p = os.path.join("/kaggle/working", nome)
    if not os.path.exists(p):
        continue
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1: x = x.mean(axis=1)
    g = np.gcd(int(sr), 16000); x16 = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
    segs, info = w.transcribe(x16, language="pt")
    hip = " ".join(s.text for s in segs).strip()
    itens.append({"arquivo": nome, "texto": texto, "transcricao": hip, "erro": wer(texto, hip), "duracao": round(len(x) / sr, 2)})
    os.system(f"ffmpeg -loglevel error -y -i {p} -b:a 128k {p[:-4]}.mp3")
json.dump({"tipo": "comparar", "itens": itens}, open("/kaggle/working/relatorio.json", "w"), ensure_ascii=False, indent=1)
print("fim", len(itens), flush=True)
'''
open("/tmp/fase3.py", "w").write(FASE3)
sh(f"{sys.executable} /tmp/fase3.py")
log("fim")
