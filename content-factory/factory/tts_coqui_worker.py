"""Coqui TTS worker process (keeps torch out of the main process).

stdin JSON: {"model_dir", "phrases": [...], "out_dir", "speed"}
stdout last line JSON: {"files": [...]}
"""
import json
import os
import sys

req = json.loads(sys.stdin.read())
from TTS.api import TTS  # noqa: E402

md = req["model_dir"]
tts = TTS(model_path=os.path.join(md, "model_file.pth"), config_path=os.path.join(md, "config.json"))
try:
    import torch
    if torch.cuda.is_available():
        tts.to("cuda")
except Exception:
    pass
files = []
speed = float(req.get("speed", 1.0))
for i, ph in enumerate(req["phrases"]):
    out = os.path.join(req["out_dir"], f"p{i:03d}.wav")
    try:
        tts.tts_to_file(text=ph, file_path=out, speed=speed)
    except TypeError:
        tts.tts_to_file(text=ph, file_path=out)
    files.append(out)
print(json.dumps({"files": files}), flush=True)
