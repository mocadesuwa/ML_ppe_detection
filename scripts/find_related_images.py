"""Suggest visual scene pairs for human grouping; never assign dataset splits."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import IMAGE_SUFFIXES, sha256, write_json
from src.prepare_dataset import difference_hash


def candidates(raw: Path, output: Path, dhash_distance: int = 8, phash_distance: int = 10) -> dict:
    import numpy as np
    from scipy.fft import dctn
    from PIL import Image, ImageDraw, ImageFont

    raw, output = raw.resolve(), output.resolve()
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 64 for v in (dhash_distance, phash_distance)):
        raise ValueError("Hash distances must be integers between 0 and 64")
    if output.exists() or raw == output or raw in output.parents or output in raw.parents:
        raise ValueError("Use a new evidence directory separate from the source")
    records = []
    for path in sorted((raw / "images").iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        with Image.open(path) as opened:
            small = np.asarray(opened.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
            low = dctn(small, type=2, norm="ortho")[:8, :8].ravel()
            bits = low > np.median(low[1:])
            bits[0] = False
            phash = sum(int(bit) << i for i, bit in enumerate(bits))
            records.append({"filename": path.name, "sha256": sha256(path), "dhash": difference_hash(opened), "phash": phash})
    pairs = []
    for i, left in enumerate(records):
        for right in records[:i]:
            dh = (left["dhash"] ^ right["dhash"]).bit_count()
            ph = (left["phash"] ^ right["phash"]).bit_count()
            if dh <= dhash_distance or ph <= phash_distance:
                pairs.append({"left": right["filename"], "right": left["filename"], "dhash_distance": dh, "phash_distance": ph})
    pairs.sort(key=lambda p: (min(p["dhash_distance"], p["phash_distance"]), p["left"], p["right"]))
    output.mkdir(parents=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    sheets = []
    for start in range(0, len(pairs), 6):
        sheet = Image.new("RGB", (1000, 1320), "#17232b")
        draw = ImageDraw.Draw(sheet)
        for row, pair in enumerate(pairs[start:start + 6]):
            draw.text((10, row * 220 + 4), f"pair {start + row + 1}: dHash={pair['dhash_distance']}, pHash={pair['phash_distance']}", fill="white", font=font)
            for column, name in enumerate((pair["left"], pair["right"])):
                with Image.open(raw / "images" / name) as opened:
                    picture = opened.convert("RGB")
                picture.thumbnail((470, 165))
                sheet.paste(picture, (column * 500 + 10, row * 220 + 50))
                draw.text((column * 500 + 10, row * 220 + 27), name, fill="white", font=font)
        filename = f"pairs_{start // 6 + 1:02d}.png"
        sheet.save(output / filename)
        sheets.append(filename)
    result = {"records": records, "pairs": pairs, "sheets": sheets, "dhash_distance": dhash_distance, "phash_distance": phash_distance,
              "limitations": "Hash candidates are incomplete scene retrieval, not person identity or automatic review. Visually review before setting groups."}
    write_json(output / "candidates.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="dataset/derived/source_v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dhash-distance", type=int, default=8)
    parser.add_argument("--phash-distance", type=int, default=10)
    args = parser.parse_args()
    report = candidates(Path(args.raw), Path(args.output), args.dhash_distance, args.phash_distance)
    print(json.dumps({"images": len(report["records"]), "candidate_pairs": len(report["pairs"]), "sheets": len(report["sheets"])}))
