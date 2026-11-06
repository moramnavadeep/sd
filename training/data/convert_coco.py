"""Convert a COCO-format export (e.g. from CVAT or Label Studio, both self-hostable) into our
manifest. Polygon categories are mapped to our region classes; a 'person' category with
keypoints supplies landmarks.

  python training/data/convert_coco.py --coco export/annotations.json --images export/images \
      --out data/real_v1 --participant-map participants.csv --consent-map consent.csv

participants.csv : file_name,participant_id[,video_id,frame_index]
consent.csv      : participant_id,consent_id      (REQUIRED for real data; rows without consent are skipped)
Per-image attributes (lighting, fabric, step_label, correction labels, pleat quality...) are read
from COCO image-level attributes if your tool exports them (CVAT: "attributes" on a whole-image
tag), using exactly the field names in saree_ai/schema/sample.schema.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from saree_ai.constants import SCHEMA_VERSION  # noqa: E402

CATEGORY_MAP = {  # COCO category name -> our region class (edit to match your label set)
    "person_skin": "person", "skin": "person", "hair": "person",
    "blouse": "other_clothing", "petticoat": "other_clothing", "other_clothing": "other_clothing",
    "saree": "saree_body", "saree_body": "saree_body", "waist_wrap": "waist_wrap", "pleats": "pleats", "pallu": "pallu",
}
IMAGE_ATTRS = {"lighting": "capture", "camera_angle": "capture", "camera_height": "capture", "body_position": "capture",
               "primary_color": "saree", "darkness": "saree", "fabric": "saree", "pattern": "saree", "transparency": "saree",
               "drape_style": "saree"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True); ap.add_argument("--images", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--participant-map", required=True); ap.add_argument("--consent-map", required=True)
    ap.add_argument("--license", default="project-internal, participant consent")
    a = ap.parse_args()
    coco = json.loads(Path(a.coco).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    pmap = {r[0]: r for r in csv.reader(open(a.participant_map, encoding="utf-8")) if r and r[0] != "file_name"}
    consent = {r[0]: r[1] for r in csv.reader(open(a.consent_map, encoding="utf-8")) if len(r) >= 2 and r[0] != "participant_id"}
    out = ROOT / a.out; (out / "images").mkdir(parents=True, exist_ok=True)
    anns = {}
    for an in coco["annotations"]:
        anns.setdefault(an["image_id"], []).append(an)
    rows, skipped = [], 0
    for im in coco["images"]:
        fn = im["file_name"]
        if fn not in pmap:
            skipped += 1; continue
        pid = pmap[fn][1]
        if pid not in consent:
            print(f"skip {fn}: no consent record for participant {pid}"); skipped += 1; continue
        dst = f"images/{Path(fn).name}"
        shutil.copy2(Path(a.images) / fn, out / dst)
        r = {"id": Path(fn).stem, "schema_version": SCHEMA_VERSION, "image": dst, "width": im["width"], "height": im["height"],
             "video_id": pmap[fn][2] if len(pmap[fn]) > 2 and pmap[fn][2] else None,
             "frame_index": int(pmap[fn][3]) if len(pmap[fn]) > 3 and pmap[fn][3] else None,
             "participant_id": pid,
             "source": {"type": "real", "consent_id": consent[pid], "license": a.license, "collector": None, "generator": None},
             "capture": {}, "saree": {}, "mask": None, "landmarks": None, "regions": [], "pleats": None,
             "step_label": None, "step_label_certain": True, "correction_labels": [], "annotator": None, "review_status": "draft"}
        attrs = im.get("attributes", {}) or {}
        for k, grp in IMAGE_ATTRS.items():
            if k in attrs: r[grp][k] = attrs[k]
        for k in ("step_label", "review_status"):
            if k in attrs: r[k] = attrs[k]
        if "correction_labels" in attrs:
            v = attrs["correction_labels"]; r["correction_labels"] = v if isinstance(v, list) else [x for x in str(v).split(",") if x]
        for an in anns.get(im["id"], []):
            name = cats[an["category_id"]]
            if an.get("keypoints") and len(an["keypoints"]) == 51:
                kp = an["keypoints"]; r["landmarks"] = [[kp[3 * i], kp[3 * i + 1], kp[3 * i + 2]] for i in range(17)]
            cls = CATEGORY_MAP.get(name)
            if cls and isinstance(an.get("segmentation"), list):
                for poly in an["segmentation"]:
                    pts = [[poly[i], poly[i + 1]] for i in range(0, len(poly), 2)]
                    if len(pts) >= 3:
                        r["regions"].append({"class": cls, "polygon": pts,
                                             "occluded_by_hand": bool((an.get("attributes") or {}).get("occluded_by_hand", False))})
            if cls == "pleats":
                at = an.get("attributes") or {}
                x, y, w, h = an["bbox"]
                r["pleats"] = {"visible": True, "roi": [x, y, x + w, y + h], "fold_lines": [],
                               "count": int(at["count"]) if at.get("count") not in (None, "") else None,
                               "count_certain": bool(at.get("count_certain", True)), "quality": at.get("quality"),
                               "secured": at.get("secured") if isinstance(at.get("secured"), bool) else None}
            if name == "pleat_fold" and an.get("segmentation"):   # polyline exported as a degenerate polygon
                seg = an["segmentation"][0]
                r.setdefault("pleats", {"visible": True, "roi": None, "fold_lines": []})
                r["pleats"].setdefault("fold_lines", []).append([[seg[i], seg[i + 1]] for i in range(0, len(seg), 2)])
        rows.append(r)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} samples, skipped {skipped}. Next: prepare_dataset.py rasterize --dir {a.out}, then validate_dataset.py")


if __name__ == "__main__":
    main()
