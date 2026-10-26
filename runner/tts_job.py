# Batch TTS job (runs inside a Kaggle GPU kernel). Placeholders are filled by run_job.py.
import base64
import io
import json
import os
import re
import subprocess
import time
import unicodedata

JOB = json.loads(base64.b64decode("__JOB_B64__"))
REFS = json.loads(base64.b64decode("__REFS_B64__"))
OUT = "/kaggle/working"
T0 = time.time()


def sh(c):
    print(">>", c, flush=True)
    subprocess.run(c, shell=True, check=True)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


sh("pip install -q -U qwen-tts faster-whisper soundfile 2>&1 | tail -3")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from qwen_tts import Qwen3TTSModel  # noqa: E402

NGPU = torch.cuda.device_count()
DTYPE = {"float32": torch.float32, "float16": torch.float16,
         "bfloat16": torch.bfloat16}[JOB.get("dtype", "float32")]
LINGUA = JOB.get("lingua", "Portuguese")
TAKES = int(JOB.get("takes", 3))
log("GPUs:", NGPU, "dtype:", DTYPE)


def carregar(nome, gpu):
    log("carregando", nome, "na GPU", gpu)
    return Qwen3TTSModel.from_pretrained(
        nome, device_map=f"cuda:{gpu}", dtype=DTYPE, attn_implementation="sdpa")


def salvar(nome, wav, sr):
    wav = np.asarray(wav, dtype=np.float32)
    caminho = os.path.join(OUT, nome + ".wav")
    sf.write(caminho, wav, sr)
    subprocess.run(f"ffmpeg -loglevel error -y -i {caminho} -b:a 128k {caminho[:-4]}.mp3",
                   shell=True, check=False)
    return nome + ".wav", len(wav) / sr


# ---------- transcrição (para checar o texto e pegar o tempo de cada palavra) ----------
WHISPER = None


def whisper():
    global WHISPER
    if WHISPER is None:
        from faster_whisper import WhisperModel
        try:
            WHISPER = WhisperModel("medium", device="cuda", compute_type="float16")
        except Exception as e:  # cudnn incompatível etc.
            log("whisper na GPU falhou, indo pra CPU:", e)
            WHISPER = WhisperModel("small", device="cpu", compute_type="int8")
    return WHISPER


def transcrever(caminho):
    try:
        # passa o áudio já carregado (16 kHz): evita o decodificador do PyAV, incompatível no Kaggle
        from scipy.signal import resample_poly
        audio, sr0 = sf.read(os.path.join(OUT, caminho), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        g = np.gcd(int(sr0), 16000)
        audio = resample_poly(audio, 16000 // g, int(sr0) // g).astype(np.float32)
        segs, _ = whisper().transcribe(audio, language="pt",
                                       word_timestamps=True, vad_filter=False)
        palavras, texto = [], []
        for s in segs:
            texto.append(s.text)
            for w in (s.words or []):
                palavras.append([w.word.strip(), round(w.start, 3), round(w.end, 3)])
        return " ".join(texto).strip(), palavras
    except Exception as e:
        log("transcrição falhou:", e)
        return "", []


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", t).split()


def erro_palavras(ref, hip):
    a, b = norm(ref), norm(hip)
    if not a:
        return 0.0
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return round(d[len(b)] / len(a), 3)


relatorio = {"tipo": JOB["tipo"], "itens": []}

if JOB["tipo"] == "casting":
    m = carregar("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", 0)
    for c in JOB["candidatos"]:
        for k, texto in enumerate(c["textos"]):
            torch.manual_seed(1000 + k)
            wavs, sr = m.generate_voice_design(text=texto, language=LINGUA, instruct=c["descricao"])
            nome, dur = salvar(f"{c['id']}_{k + 1}", wavs[0], sr)
            hip, pal = transcrever(nome)
            item = {"id": c["id"], "quem": c["quem"], "n": k + 1, "arquivo": nome,
                    "duracao": round(dur, 2), "texto": texto, "transcricao": hip,
                    "erro": erro_palavras(texto, hip), "palavras": pal}
            relatorio["itens"].append(item)
            log(item["id"], item["n"], f"{dur:.1f}s erro={item['erro']}", hip)

elif JOB["tipo"] == "episodio":
    falas = JOB["falas"]
    precisa_clone = any(f.get("modo", "clone") == "clone" for f in falas)
    precisa_design = any(f.get("modo") == "design" for f in falas)
    clone = carregar("Qwen/Qwen3-TTS-12Hz-1.7B-Base", 0) if precisa_clone else None
    design = carregar("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", 1 if NGPU > 1 else 0) if precisa_design else None
    prompts = {}
    for quem, r in REFS.items():
        audio, sr_ref = sf.read(io.BytesIO(base64.b64decode(r["audio_b64"])), dtype="float32")
        prompts[quem] = clone.create_voice_clone_prompt(ref_audio=(audio, sr_ref), ref_text=r["texto"])
        log("voz de referência pronta:", quem)
    pers = JOB.get("personagens", {})
    for f in falas:
        for k in range(TAKES):
            torch.manual_seed(2000 + k)
            if f.get("modo", "clone") == "clone":
                wavs, sr = clone.generate_voice_clone(text=f["texto"], language=LINGUA,
                                                      voice_clone_prompt=prompts[f["quem"]])
            else:
                desc = pers.get(f["quem"], {}).get("descricao", "")
                instr = (desc + " " + f.get("emocao", "")).strip()
                wavs, sr = design.generate_voice_design(text=f["texto"], language=LINGUA, instruct=instr)
            nome, dur = salvar(f"{f['id']}_t{k + 1}", wavs[0], sr)
            hip, pal = transcrever(nome)
            item = {"fala": f["id"], "quem": f["quem"], "take": k + 1, "arquivo": nome,
                    "duracao": round(dur, 2), "texto": f["texto"], "transcricao": hip,
                    "erro": erro_palavras(f["texto"], hip), "palavras": pal}
            relatorio["itens"].append(item)
            log(f["id"], "take", k + 1, f"{dur:.1f}s erro={item['erro']}", hip)

relatorio["segundos"] = round(time.time() - T0)
with open(os.path.join(OUT, "relatorio.json"), "w", encoding="utf-8") as fh:
    json.dump(relatorio, fh, ensure_ascii=False, indent=1)
log("fim")
