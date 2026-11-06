"""One-time download of the THIRD-PARTY local model files. Run while online; afterwards the app
works fully offline. Nothing here is an API: these are weight files stored in ./models.

  python scripts/download_models.py            # pose baseline + ASR + ImageNet backbone
  python scripts/download_models.py --asr      # only the speech model

Each file is checked against a known hash prefix and recorded in models/registry.json as
'pretrained_baseline' with its origin and licence.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from saree_ai import registry  # noqa: E402

FILES = {
    "pose": {
        "url": "https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth",
        "dest": "models/pose_baseline/keypointrcnn_resnet50_fpn_coco-fc266e95.pth",
        "sha256_prefix": "fc266e95",
        "registry": {"task": "pose_baseline", "status": "pretrained_baseline",
                     "origin": "torchvision.models.detection.keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights.COCO_V1",
                     "licence": "torchvision code BSD-3-Clause; weights trained by the PyTorch team on COCO 2017 "
                                "(annotations CC BY 4.0, images under individual Flickr licences). Not trained by us; "
                                "we claim no ownership.",
                     "trained_from": "pretrained (third party)"},
    },
    "backbone": {
        "url": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
        "dest": "models/torch_home/hub/checkpoints/mobilenet_v3_large-8738ca79.pth",
        "sha256_prefix": "8738ca79",
        "registry": None,  # training-time initialisation only
    },
    "asr": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "dest": "models/asr/vosk-model-small-en-us-0.15.zip",
        "sha256_prefix": None,
        "unzip_to": "models/asr",
        "registry": {"task": "asr", "status": "pretrained_baseline",
                     "origin": "Vosk small English model 0.15 (Alpha Cephei)", "licence": "Apache-2.0",
                     "trained_from": "pretrained (third party)"},
    },
    "asr_te": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-te-0.42.zip",
        "dest": "models/asr/vosk-model-small-te-0.42.zip",
        "sha256_prefix": None,
        "unzip_to": "models/asr",
        "registry": {"task": "asr_te", "status": "pretrained_baseline",
                     "origin": "Vosk small Telugu model 0.42 (Alpha Cephei); published WER 87.9 on Fleurs (open dictation)",
                     "licence": "Apache-2.0", "trained_from": "pretrained (third party)"},
    },
    # Telugu speech output. Runtime piper-tts is GPL-3.0-or-later. The voice was trained on IIT Madras
    # IndicTTS, whose licence (www.iitm.ac.in/donlab/indictts/downloads/license.pdf) we have NOT been
    # able to read: treat as research/personal use only until verified.
    "tts_te_config": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/te/te_IN/maya/medium/te_IN-maya-medium.onnx.json",
        "dest": "models/tts/te_IN-maya-medium.onnx.json",
        "sha256_prefix": None,
        "registry": None,
    },
    "tts_te": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/te/te_IN/maya/medium/te_IN-maya-medium.onnx",
        "dest": "models/tts/te_IN-maya-medium.onnx",
        "sha256_prefix": None,
        "registry": {"task": "tts_te", "status": "pretrained_baseline",
                     "origin": "Piper voice te_IN-maya-medium (rhasspy/piper-voices), fine-tuned from en_US lessac on IIT Madras IndicTTS",
                     "licence": "voice repo MIT; training data IndicTTS licence UNVERIFIED (PDF unreachable 2026-11-06) - "
                                "research/personal use only until checked; runtime piper-tts GPL-3.0-or-later",
                     "trained_from": "pretrained (third party)"},
    },
}


def fetch(name, spec):
    dest = ROOT / spec["dest"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {spec['url']}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(spec["url"], tmp)
        tmp.replace(dest)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    if spec["sha256_prefix"] and not h.startswith(spec["sha256_prefix"]):
        dest.unlink()
        raise SystemExit(f"{name}: hash mismatch ({h[:8]} != {spec['sha256_prefix']}); file removed")
    if spec.get("unzip_to"):
        with zipfile.ZipFile(dest) as z:
            for m in z.namelist():  # zip-slip guard
                if Path(m).is_absolute() or ".." in Path(m).parts:
                    raise SystemExit(f"unsafe path in archive: {m}")
            z.extractall(ROOT / spec["unzip_to"])
    if spec["registry"]:
        registry.register(spec["registry"]["task"], {**spec["registry"], "file": spec["dest"], "source_url": spec["url"],
                                                     "version": Path(spec["dest"]).stem})
    print(f"{name}: ok  sha256={h[:16]}…  -> {spec['dest']}")


def main():
    ap = argparse.ArgumentParser()
    for k in FILES:
        ap.add_argument(f"--{k}", action="store_true")
    a = ap.parse_args()
    chosen = [k for k in FILES if getattr(a, k)] or list(FILES)
    for k in chosen:
        fetch(k, FILES[k])


if __name__ == "__main__":
    main()
