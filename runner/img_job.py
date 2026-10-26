# Batch image job (runs inside a Kaggle GPU kernel). Placeholders are filled by run_job.py.
import base64
import json
import os
import subprocess
import time

JOB = json.loads(base64.b64decode("__JOB_B64__"))
OUT = "/kaggle/working"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


subprocess.run("pip install -q -U diffusers transformers accelerate 2>&1 | tail -2", shell=True)
import torch  # noqa: E402
from diffusers import StableDiffusionXLPipeline  # noqa: E402

pipe = StableDiffusionXLPipeline.from_pretrained(
    JOB.get("modelo", "stabilityai/stable-diffusion-xl-base-1.0"),
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda:0")
pipe.enable_vae_slicing()
estilo = JOB.get("estilo", "")
neg = JOB.get("negativo", "")
rel = {"tipo": "imagens", "itens": []}
for img in JOB["imagens"]:
    for s in img.get("seeds", [1]):
        g = torch.Generator("cuda").manual_seed(int(s))
        im = pipe(prompt=f"{img['prompt']}, {estilo}", negative_prompt=neg,
                  width=img.get("w", 768), height=img.get("h", 1344),
                  num_inference_steps=img.get("passos", 30), guidance_scale=img.get("cfg", 6.5),
                  generator=g).images[0]
        nome = f"{img['id']}_s{s}.png"
        im.save(os.path.join(OUT, nome))
        rel["itens"].append({"id": img["id"], "seed": s, "arquivo": nome})
        log("ok", nome)
rel["segundos"] = round(time.time() - T0)
json.dump(rel, open(os.path.join(OUT, "relatorio.json"), "w"), indent=1)
log("fim")
