import json
from pathlib import Path

def is_thermal_image(img: dict) -> bool:
    """
    Decide whether an image is thermal based on common Roboflow fields.
    Adjust rules if your naming differs.
    """
    fn = (img.get("file_name") or "").lower()
    extra_name = (img.get("extra", {}).get("name") or "").lower()

    # Common patterns: "thermal_..." or contains "/thermal/" etc.
    return ("thermal" in fn) or ("thermal" in extra_name)

def filter_coco_keep_bw(input_path: Path, output_path: Path, prune_unused_categories: bool = False):
    data = json.loads(input_path.read_text(encoding="utf-8"))

    images = data.get("images", [])
    anns = data.get("annotations", [])
    cats = data.get("categories", [])
    licenses = data.get("licenses", [])
    info = data.get("info", {})

    kept_images = [img for img in images if not is_thermal_image(img)]
    kept_image_ids = {img["id"] for img in kept_images}

    kept_anns = [ann for ann in anns if ann.get("image_id") in kept_image_ids]

    if prune_unused_categories and cats:
        used_cat_ids = {ann.get("category_id") for ann in kept_anns if "category_id" in ann}
        kept_cats = [c for c in cats if c.get("id") in used_cat_ids]
    else:
        kept_cats = cats

    out = {
        "info": info,
        "licenses": licenses,
        "images": kept_images,
        "annotations": kept_anns,
        "categories": kept_cats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    removed_images = len(images) - len(kept_images)
    removed_anns = len(anns) - len(kept_anns)
    print(f"[{input_path.name}] kept images={len(kept_images)} (removed {removed_images}), "
          f"kept anns={len(kept_anns)} (removed {removed_anns}) -> {output_path}")

def main():
    """
    Expected structure (common Roboflow export):
      dataset/
        train/_annotations.coco.json
        valid/_annotations.coco.json
        test/_annotations.coco.json

    Run:
      python filter_bw_only.py /path/to/dataset
    """

    root = Path(r"C:\Users\Johnny\Desktop\deer_detector\mmdetection\data")
    # Common file names; we’ll try these first, but also fall back to searching.
    split_files = [
        root / "train" / "_annotations.coco.json",
        root / "val" / "_annotations.coco.json",
        root / "test" / "_annotations.coco.json",
    ]

    # If some don’t exist, try to discover any COCO jsons under train/valid/test.
    discovered = []
    for split in ["train", "valid", "test"]:
        split_dir = root / split
        if split_dir.exists():
            discovered.extend(split_dir.glob("*.coco.json"))
            discovered.extend(split_dir.glob("*.json"))

    # Use explicit list if present, otherwise discovered candidates
    candidates = [p for p in split_files if p.exists()]
    if not candidates:
        # Narrow discovered to those that look like COCO annotation files
        candidates = [p for p in discovered if p.is_file() and "coco" in p.name.lower()]

    if not candidates:
        print("Could not find COCO annotation JSONs under train/valid/test.")
        raise SystemExit(1)

    for in_path in candidates:
        out_path = in_path.with_name(in_path.stem + ".bw_only.json")
        filter_coco_keep_bw(in_path, out_path, prune_unused_categories=False)

if __name__ == "__main__":
    main()
