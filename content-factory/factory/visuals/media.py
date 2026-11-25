"""External media scenes (stock photos / AI images / stock videos) + providers.

Providers (all optional, chosen by config visuals.providers):
  pexels   – free API key, Pexels licence (free commercial use, no attribution required)
  pixabay  – free API key, Pixabay content licence
  nasa     – images-api.nasa.gov, NASA media is generally public domain (check 'center'/'secondary_creator')
  comfyui  – local ComfyUI server on the RTX 5070 Ti (SDXL/Flux), fully original AI images
Every download is logged with source URL + licence into licenses.json of the video.
"""
from __future__ import annotations

import json
import random
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

from .. import secrets
from ..config import cfg
from .procedural import OVERSCAN, Scene

UA = {"User-Agent": "ai-content-factory/1.0 (local video tool)"}


class MediaImage(Scene):
    def build(self):
        img = Image.open(self.p["path"]).convert("RGB")
        img = ImageOps.fit(img, (self.W, self.H), Image.LANCZOS, centering=(0.5, 0.45))
        self.base = np.asarray(img, np.float32)


# ------------------------------------------------------------------ providers
def _download(url: str, dest: Path) -> Path:
    with requests.get(url, stream=True, timeout=60, headers=UA) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    return dest


def pexels(query: str, dest_dir: Path, video: bool = True):
    key = secrets.get("pexels_api_key")
    if not key:
        return None
    if video:
        r = requests.get("https://api.pexels.com/videos/search", headers={"Authorization": key, **UA},
                         params={"query": query, "orientation": "portrait", "per_page": 15, "size": "large"}, timeout=20)
        r.raise_for_status()
        vids = [v for v in r.json().get("videos", []) if 4 <= v.get("duration", 0) <= 40]
        if not vids:
            return None
        v = random.choice(vids[:6])
        files = sorted([f for f in v["video_files"] if f.get("height", 0) >= 1280 and f.get("file_type") == "video/mp4"],
                       key=lambda f: f["height"])
        if not files:
            return None
        p = _download(files[0]["link"], dest_dir / f"pexels_{v['id']}.mp4")
        return {"path": str(p), "type": "video", "source": "pexels", "url": v["url"],
                "author": v.get("user", {}).get("name"), "license": "Pexels License (free to use)"}
    r = requests.get("https://api.pexels.com/v1/search", headers={"Authorization": key, **UA},
                     params={"query": query, "orientation": "portrait", "per_page": 15}, timeout=20)
    r.raise_for_status()
    ph = r.json().get("photos", [])
    if not ph:
        return None
    x = random.choice(ph[:6])
    p = _download(x["src"]["original"], dest_dir / f"pexels_{x['id']}.jpg")
    return {"path": str(p), "type": "image", "source": "pexels", "url": x["url"],
            "author": x.get("photographer"), "license": "Pexels License (free to use)"}


def pixabay(query: str, dest_dir: Path, video: bool = True):
    key = secrets.get("pixabay_api_key")
    if not key:
        return None
    url = "https://pixabay.com/api/videos/" if video else "https://pixabay.com/api/"
    r = requests.get(url, params={"key": key, "q": query[:100], "safesearch": "true", "per_page": 20}, timeout=20, headers=UA)
    r.raise_for_status()
    hits = r.json().get("hits", [])
    if not hits:
        return None
    h = random.choice(hits[:6])
    if video:
        f = h["videos"].get("large") or h["videos"].get("medium")
        if not f or not f.get("url"):
            return None
        p = _download(f["url"], dest_dir / f"pixabay_{h['id']}.mp4")
        typ = "video"
    else:
        p = _download(h.get("largeImageURL") or h["webformatURL"], dest_dir / f"pixabay_{h['id']}.jpg")
        typ = "image"
    return {"path": str(p), "type": typ, "source": "pixabay", "url": h.get("pageURL"),
            "author": h.get("user"), "license": "Pixabay Content License"}


def nasa(query: str, dest_dir: Path, video: bool = False):
    r = requests.get("https://images-api.nasa.gov/search", params={"q": query, "media_type": "image"}, timeout=20, headers=UA)
    r.raise_for_status()
    items = r.json().get("collection", {}).get("items", [])
    items = [i for i in items if i.get("data") and not i["data"][0].get("secondary_creator")]
    if not items:
        return None
    it = random.choice(items[:10])
    nasa_id = it["data"][0]["nasa_id"]
    assets = requests.get(f"https://images-api.nasa.gov/asset/{nasa_id}", timeout=20, headers=UA).json()
    hrefs = [i["href"] for i in assets["collection"]["items"] if i["href"].lower().endswith((".jpg", ".png"))]
    pick = next((h for h in hrefs if "~orig" in h), None) or next((h for h in hrefs if "~large" in h), hrefs[0] if hrefs else None)
    if not pick:
        return None
    p = _download(pick.replace("http://", "https://"), dest_dir / f"nasa_{uuid.uuid4().hex[:8]}.jpg")
    with Image.open(p) as im:
        if min(im.size) < 900:
            return None
    return {"path": str(p), "type": "image", "source": "nasa", "url": f"https://images.nasa.gov/details/{nasa_id}",
            "author": it["data"][0].get("center"), "license": "NASA media usage guidelines (generally public domain, no endorsement)"}


COMFY_WORKFLOW = {
    "3": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 28, "cfg": 5.5, "sampler_name": "dpmpp_2m",
          "scheduler": "karras", "denoise": 1, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1472, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "text, watermark, logo, low quality, blurry, deformed", "clip": ["4", 1]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "cf", "images": ["8", 0]}},
}


def comfyui(prompt: str, dest_dir: Path, video: bool = False):
    base = cfg().get("visuals.comfyui_url").rstrip("/")
    try:
        requests.get(base + "/system_stats", timeout=2).raise_for_status()
    except Exception:
        return None
    wf = json.loads(json.dumps(COMFY_WORKFLOW))
    ckpt = cfg().get("visuals.comfyui_checkpoint")
    if ckpt:
        wf["4"]["inputs"]["ckpt_name"] = ckpt
    wf["6"]["inputs"]["text"] = prompt + ", cinematic lighting, ultra detailed, vertical composition, 35mm film still"
    wf["3"]["inputs"]["seed"] = random.randint(0, 2 ** 31)
    pid = requests.post(base + "/prompt", json={"prompt": wf}, timeout=30).json()["prompt_id"]
    for _ in range(240):
        time.sleep(1.5)
        hist = requests.get(f"{base}/history/{pid}", timeout=10).json()
        if pid in hist:
            imgs = hist[pid]["outputs"]["9"]["images"]
            im = imgs[0]
            data = requests.get(base + "/view", params={"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]}, timeout=30)
            p = dest_dir / f"comfy_{pid[:8]}.png"
            p.write_bytes(data.content)
            return {"path": str(p), "type": "image", "source": "comfyui-local", "url": None, "author": "AI (local SDXL)",
                    "license": "Original AI generation (own)", "prompt": prompt}
    return None


PROVIDERS = {"pexels": pexels, "pixabay": pixabay, "nasa": nasa, "comfyui": comfyui}


def fetch(scene: dict, dest_dir: Path, niche: str) -> dict | None:
    """Try configured providers for a scene. Returns media dict or None (-> procedural)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    order = cfg().get("visuals.providers", ["procedural"])
    for name in order:
        if name == "procedural":
            return None
        if name == "nasa" and niche != "space":
            continue
        fn = PROVIDERS.get(name)
        if not fn:
            continue
        query = scene.get("image_prompt") if name == "comfyui" else scene.get("stock_query")
        if not query:
            continue
        try:
            res = fn(query, dest_dir, video=name in ("pexels", "pixabay"))
            if res:
                return res
        except Exception:
            continue
    return None
