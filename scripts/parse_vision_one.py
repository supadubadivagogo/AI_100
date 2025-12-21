import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HAN = "\uD55C\uC2DD"  # 한식
POP = "\uD31D\uC5C5"  # 팝업
YANG = "\uC591\uC2DD"  # 양식
SALADBAR = "\uC0D0\uB7EC\uB4DC\uBC14"  # 샐러드바
SALAD = "\uC0D0\uB7EC\uB4DC"  # 샐러드
VEGAN = "\uBE44\uAC74"  # 비건
BURGER = "\uBC84\uAC70"  # 버거
DELI = "\uB378\uB9AC"  # 델리
RICE = "\uB77C\uC774\uC2A4"  # 라이스
NUDEL = "\uB204\uB4E4"  # 누들

KCAL_RE = re.compile(r"(\d{3,4})\s*kca?l", re.IGNORECASE)
NUM_RE = re.compile(r"\d{3,4}")
PAREN_NUM_RE = re.compile(r"\((\d{1,4})\)")
HANGUL_RE = re.compile(r"[가-힣]")
EN_MULTI_RE = re.compile(r"[A-Za-z]{2,}")
SHORT_TOKEN_MAX = 3
BRAND_TOKENS = {"OURHOME", "ONSI", "CHO"}
ROW_BAND_SCALE = 0.08
ROW_BAND_MIN = 3
ROW_BAND_MAX = 10
ROW_BAND_OVERRIDES = {
    ("\uC11D\uC2DD", "\uD55C\uC2DDB"): {"top_expand": 25, "bottom_expand": 8},
    ("\uC911\uC2DD", "\uB77C\uC774\uC2A4&\uB204\uB4E4"): {
        "top_expand": 0,
        "bottom_expand": -12,
    },
    ("\uC911\uC2DD", "\uC0D0\uB7EC\uB4DC\uBC14"): {"top_expand": 0, "bottom_expand": 12},
    ("\uC911\uC2DD", "\uC0D0\uB7EC\uB4DC"): {"top_expand": -8, "bottom_expand": 0},
    ("\uC11D\uC2DD", "\uC0D0\uB7EC\uB4DC\uBC14"): {"top_expand": -8, "bottom_expand": 0},
}


def load_words(vision_path: Path):
    obj = json.loads(vision_path.read_text(encoding="utf-8"))
    page = obj["responses"][0]["fullTextAnnotation"]["pages"][0]
    words = []
    max_x = 0
    max_y = 0
    for block in page["blocks"]:
        for para in block["paragraphs"]:
            for w in para["words"]:
                text = "".join(s["text"] for s in w["symbols"])
                box = w["boundingBox"]["vertices"]
                xs = [v.get("x", 0) for v in box]
                ys = [v.get("y", 0) for v in box]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                max_x = max(max_x, x1)
                max_y = max(max_y, y1)
                words.append(
                    {"text": text, "x": (x0 + x1) / 2, "y": (y0 + y1) / 2}
                )
    return words, max_x, max_y


def kmeans_1d(xs):
    xs_np = np.array(xs, dtype=float)
    centers = np.quantile(xs_np, [0.1, 0.3, 0.5, 0.7, 0.9])
    for _ in range(30):
        dists = np.abs(xs_np[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for i in range(5):
            members = xs_np[labels == i]
            if len(members):
                new_centers[i] = members.mean()
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return np.sort(centers)


def column_bounds(words, height):
    xs = []
    top_words = [w for w in words if w["y"] <= height * 0.12]
    top_words.sort(key=lambda w: (w["y"], w["x"]))
    lines = []
    cur = []
    cur_y = None
    for w in top_words:
        if cur_y is None or abs(w["y"] - cur_y) > 12:
            if cur:
                lines.append(cur)
            cur = [w]
            cur_y = w["y"]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)

    # Prefer day tokens (e.g., "06" followed by "일", or "1/6").
    for line in lines:
        line_sorted = sorted(line, key=lambda w: w["x"])
        for idx, w in enumerate(line_sorted):
            text = w["text"]
            m = re.fullmatch(r"(\d{1,2})일", text)
            if m:
                day = int(m.group(1))
                if 1 <= day <= 31:
                    xs.append(w["x"])
                continue
            m = re.fullmatch(r"(\d{1,2})\s*/\s*(\d{1,2})", text)
            if m:
                day = int(m.group(2))
                if 1 <= day <= 31:
                    xs.append(w["x"])
                continue
            if re.fullmatch(r"\d{1,2}", text):
                if idx + 1 < len(line_sorted) and line_sorted[idx + 1]["text"] == "일":
                    day = int(text)
                    if 1 <= day <= 31:
                        xs.append(w["x"])
    if len(xs) < 5:
        xs = []
        for w in words:
            if w["y"] > height * 0.12:
                continue
            for d in re.findall(r"\d+", w["text"]):
                val = int(d)
                if 10 <= val <= 31:
                    xs.append(w["x"])
    centers = kmeans_1d(xs)
    col_width = float(centers[1] - centers[0])
    bounds = [(c - col_width / 2, c + col_width / 2) for c in centers]
    left_boundary = max(0, centers[0] - col_width / 2)
    return bounds, left_boundary


def row_bounds(words, height, left_boundary):
    left_words = [w for w in words if w["x"] < left_boundary and w["y"] > height * 0.05]
    left_words.sort(key=lambda w: (w["y"], w["x"]))
    lines = []
    cur = []
    cur_y = None
    for w in left_words:
        if cur_y is None or abs(w["y"] - cur_y) > 12:
            if cur:
                lines.append(cur)
            cur = [w]
            cur_y = w["y"]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)

    line_items = []
    for line in lines:
        line_sorted = sorted(line, key=lambda w: w["x"])
        text = "".join([w["text"] for w in line_sorted])
        y = sum(w["y"] for w in line_sorted) / len(line_sorted)
        line_items.append({"y": y, "text": text})

    anchors = {}
    list_anchors = {k: [] for k in ["hanB", "saladbar", "salad", "burger"]}
    ra_y = []
    nu_y = []
    for it in line_items:
        t = it["text"]
        y = it["y"]
        if (HAN + "A") in t:
            anchors["hanA"] = y
        if (HAN + "B") in t:
            list_anchors["hanB"].append(y)
        if (POP + "A") in t:
            anchors["popA"] = y
        if (POP + "B") in t:
            anchors["popB"] = y
        if YANG in t:
            anchors["yang"] = y
        if SALADBAR in t:
            list_anchors["saladbar"].append(y)
        if SALAD in t and SALADBAR not in t:
            list_anchors["salad"].append(y)
        if VEGAN in t:
            anchors["vegan"] = y
        if BURGER in t or DELI in t:
            list_anchors["burger"].append(y)
        if RICE in t:
            ra_y.append(y)
        if NUDEL in t:
            nu_y.append(y)

    anchors["hanB_lunch"] = list_anchors["hanB"][0]
    anchors["hanB_dinner"] = list_anchors["hanB"][1]
    anchors["saladbar_lunch"] = list_anchors["saladbar"][0]
    anchors["saladbar_dinner"] = list_anchors["saladbar"][1]
    anchors["salad_lunch"] = list_anchors["salad"][0]
    anchors["salad_dinner"] = list_anchors["salad"][1]
    anchors["burger_lunch"] = list_anchors["burger"][0]
    anchors["burger_dinner"] = list_anchors["burger"][1]
    if ra_y or nu_y:
        anchors["rice"] = sum(ra_y + nu_y) / len(ra_y + nu_y)

    row_keys = [
        ("\uC911\uC2DD", "\uD55C\uC2DDA", "hanA"),
        ("\uC911\uC2DD", "\uD55C\uC2DDB", "hanB_lunch"),
        ("\uC911\uC2DD", "\uD31D\uC5C5A", "popA"),
        ("\uC911\uC2DD", "\uD31D\uC5C5B", "popB"),
        ("\uC911\uC2DD", "\uC591\uC2DD", "yang"),
        ("\uC911\uC2DD", "\uC0D0\uB7EC\uB4DC\uBC14", "saladbar_lunch"),
        ("\uC911\uC2DD", "\uC0D0\uB7EC\uB4DC", "salad_lunch"),
        ("\uC911\uC2DD", "\uBE44\uAC74", "vegan"),
        ("\uC911\uC2DD", "\uBC84\uAC70&\uB378\uB9AC", "burger_lunch"),
        ("\uC911\uC2DD", "\uB77C\uC774\uC2A4&\uB204\uB4E4", "rice"),
        ("\uC11D\uC2DD", "\uD55C\uC2DDB", "hanB_dinner"),
        ("\uC11D\uC2DD", "\uC0D0\uB7EC\uB4DC\uBC14", "saladbar_dinner"),
        ("\uC11D\uC2DD", "\uC0D0\uB7EC\uB4DC", "salad_dinner"),
        ("\uC11D\uC2DD", "\uBC84\uAC70&\uB378\uB9AC", "burger_dinner"),
    ]

    rows = [(meal, corner, anchors[key]) for (meal, corner, key) in row_keys]
    rows.sort(key=lambda r: r[2])
    bounds = {}
    for idx, (meal, corner, y) in enumerate(rows):
        if idx == 0:
            next_y = rows[idx + 1][2]
            top = max(0, y - (next_y - y) / 2)
        else:
            prev_y = rows[idx - 1][2]
            top = (prev_y + y) / 2
        if idx == len(rows) - 1:
            prev_y = rows[idx - 1][2]
            bottom = min(height, y + (y - prev_y) / 2)
        else:
            next_y = rows[idx + 1][2]
            bottom = (y + next_y) / 2
        bounds[(meal, corner)] = (top, bottom)
    return bounds


def clean_line(line):
    original = line
    line = line.strip()
    reasons = []
    if "•" in line or "?" in line:
        reasons.append("bullet_garbage")
        line = line.replace("•", "").replace("?", "")
    if "_" in line:
        reasons.append("underscore_garbage")
        line = line.replace("_", "")
    line = line.replace("·", "").replace("|", "").strip()
    if PAREN_NUM_RE.search(line):
        reasons.append("paren_numeric")
        line = PAREN_NUM_RE.sub("", line).strip()
    if line == "-":
        reasons.append("dash_only")
        line = ""
    if line and line.upper() in BRAND_TOKENS:
        reasons.append("brand")
    if EN_MULTI_RE.search(line):
        reasons.append("english_removed")
        line = EN_MULTI_RE.sub("", line).strip()
    if re.fullmatch(r"\d{1,2}", line or ""):
        reasons.append("short_numeric")
        line = ""
    if re.fullmatch(r"\d+", line or ""):
        reasons.append("numeric_only")
        line = ""
    return original, line, reasons


def nearest_row_key(y, row_centers):
    return min(row_centers.items(), key=lambda kv: abs(y - kv[1]))[0]


def compute_row_margins(row_bounds):
    margins = {}
    for key, (top, bottom) in row_bounds.items():
        height = max(1.0, bottom - top)
        margin = int(round(height * ROW_BAND_SCALE))
        margin = max(ROW_BAND_MIN, min(ROW_BAND_MAX, margin))
        margins[key] = margin
    return margins


def compute_row_bands(row_bounds, row_margins):
    bands = {}
    for key, (top, bottom) in row_bounds.items():
        margin = row_margins.get(key, ROW_BAND_MIN)
        band_top = top + margin
        band_bottom = bottom - margin
        override = ROW_BAND_OVERRIDES.get(key)
        if override:
            band_top = max(0, band_top - override.get("top_expand", 0))
            band_bottom = band_bottom + override.get("bottom_expand", 0)
        bands[key] = (band_top, band_bottom)
    return bands


def assign_row_key(y, row_bounds, row_centers, row_bands):
    candidates = []
    for key, (band_top, band_bottom) in row_bands.items():
        if band_top <= y <= band_bottom:
            candidates.append((key, abs(y - row_centers[key])))
    if len(candidates) == 1:
        return candidates[0][0]
    if candidates:
        candidates.sort(key=lambda kv: kv[1])
        return candidates[0][0]
    return nearest_row_key(y, row_centers)


def build_column_lines(words, x0, x1, y_min, y_max):
    col_words = [
        w for w in words if x0 <= w["x"] <= x1 and y_min <= w["y"] <= y_max
    ]
    col_words.sort(key=lambda w: (w["y"], w["x"]))
    lines = []
    cur = []
    cur_y = None
    for w in col_words:
        y = w["y"]
        if cur_y is None or abs(y - cur_y) > 12:
            if cur:
                lines.append(cur)
            cur = [w]
            cur_y = y
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    line_items = []
    for line in lines:
        line_sorted = sorted(line, key=lambda t: t["x"])
        text = "".join([t["text"] for t in line_sorted])
        y = sum(t["y"] for t in line_sorted) / len(line_sorted)
        line_items.append({"text": text, "y": y})
    return line_items


def parse_cell(line_items, kcal_tokens, y0, y1):
    text_lines = [li["text"] for li in line_items]
    raw = "\n".join(text_lines)
    kcal = None
    m = list(KCAL_RE.finditer(raw))
    if m:
        kcal = int(m[-1].group(1))
    if kcal is None:
        candidates = []
        for t in kcal_tokens:
            if t["y"] < y0 or t["y"] > y1 + 24:
                continue
            m2 = KCAL_RE.search(t["text"])
            if m2:
                candidates.append((t["y"], t["x"], int(m2.group(1))))
                continue
            m3 = NUM_RE.search(t["text"])
            if m3:
                val = int(m3.group(0))
                if 300 <= val <= 2000:
                    candidates.append((t["y"], t["x"], val))
        if candidates:
            candidates.sort()
            kcal = candidates[-1][2]
    cleaned_lines = []
    suspects = []
    for li in line_items:
        ln = li["text"]
        if KCAL_RE.search(ln):
            ln = KCAL_RE.sub("", ln).strip()
        if not ln:
            continue
        original, cleaned, reasons = clean_line(ln)
        if reasons or original != cleaned:
            log_reasons = reasons[:] if reasons else ["text_changed"]
            suspects.append(
                {
                    "original": original,
                    "cleaned": cleaned,
                    "reasons": log_reasons,
                }
            )
        if cleaned:
            cleaned_lines.append({"text": cleaned, "y": li["y"]})

    # Merge lines starting with '&' into previous line
    merged = []
    i = 0
    while i < len(cleaned_lines):
        item = cleaned_lines[i]
        text = item["text"]
        if text.startswith("&") and merged:
            merged[-1]["text"] = merged[-1]["text"] + text
            merged[-1]["reasons"].append("merge_amp")
            i += 1
            continue
        if (
            len(text) <= SHORT_TOKEN_MAX
            and (i + 1) < len(cleaned_lines)
            and not HANGUL_RE.search(text)
        ):
            next_item = cleaned_lines[i + 1]
            merged.append(
                {
                    "text": text + next_item["text"],
                    "y": next_item["y"],
                    "reasons": ["merge_short"],
                    "original_parts": [text, next_item["text"]],
                }
            )
            i += 2
            continue
        merged.append({"text": text, "y": item["y"], "reasons": []})
        i += 1

    if not merged:
        return [], kcal, "", [], [], suspects

    main = merged[0]["text"]
    sides = [m["text"] for m in merged[1:]]
    for m in merged:
        if m["reasons"]:
            suspects.append(
                {
                    "original": m["text"],
                    "cleaned": m["text"],
                    "reasons": m["reasons"],
                    "original_parts": m.get("original_parts"),
                }
            )
    menus = [main] + sides
    return menus, kcal, main, sides, merged, suspects


def remove_menu(record, text):
    record["_items"] = [l for l in record["_items"] if l["text"] != text]


def dedup_adjacent(records, row_centers):
    by_date = {}
    for r in records:
        key = (r["meal_time"], r["corner"])
        by_date.setdefault(r["date"], {})[key] = r
    order = [k for k, _ in sorted(row_centers.items(), key=lambda kv: kv[1])]
    for date, items in by_date.items():
        for i in range(len(order) - 1):
            c1 = order[i]
            c2 = order[i + 1]
            r1 = items.get(c1)
            r2 = items.get(c2)
            if not r1 or not r2:
                continue
            texts1 = {l["text"]: l for l in r1["_items"]}
            texts2 = {l["text"]: l for l in r2["_items"]}
            dup = set(texts1.keys()) & set(texts2.keys())
            for text in dup:
                y1 = texts1[text]["y"]
                y2 = texts2[text]["y"]
                row_gap = abs(row_centers[c1] - row_centers[c2])
                if abs(y1 - y2) > row_gap * 0.5:
                    continue
                d1 = abs(y1 - row_centers[c1])
                d2 = abs(y2 - row_centers[c2])
                if d1 <= d2:
                    remove_menu(r2, text)
                else:
                    remove_menu(r1, text)


def split_salad_mixed(records, review):
    by_key = {}
    for r in records:
        by_key[(r["date"], r["meal_time"], r["corner"])] = r
    for r in records:
        if r["corner"] not in ("\uC0D0\uB7EC\uB4DC", "\uC0D0\uB7EC\uB4DC\uBC14"):
            continue
        salad = by_key.get((r["date"], r["meal_time"], "\uC0D0\uB7EC\uB4DC"))
        saladbar = by_key.get(
            (r["date"], r["meal_time"], "\uC0D0\uB7EC\uB4DC\uBC14")
        )
        if not saladbar:
            continue
        def split_salad_text(text):
            split_idx = -1
            if "\uAE40\uCE58" in text:
                split_idx = text.rfind("\uAE40\uCE58") + len("\uAE40\uCE58")
                reason = "salad_split_by_kimchi"
            elif "\uBC25" in text:
                split_idx = text.rfind("\uBC25") + len("\uBC25")
                reason = "salad_split_by_rice"
            else:
                split_idx = text.find("\uC0D0\uB7EC\uB4DC")
                reason = "salad_split_by_salad"
            prefix = text[:split_idx].strip() if split_idx > 0 else ""
            suffix = text[split_idx:].strip() if split_idx > 0 else text.strip()
            return prefix, suffix, reason

        # Handle lines that belong to salad but bled into saladbar
        if r["corner"] == "\uC0D0\uB7EC\uB4DC\uBC14" and salad:
            new_bar = []
            for item in r["_items"]:
                m = item["text"]
                if "\uC0D0\uB7EC\uB4DC" in m and ("\uBC25" in m or "\uAE40\uCE58" in m):
                    prefix, suffix, reason = split_salad_text(m)
                    if prefix:
                        new_bar.append({"text": prefix, "y": item["y"]})
                    if suffix:
                        salad["_items"].append({"text": suffix, "y": item["y"]})
                    review.append(
                        {
                            "original": m,
                            "cleaned": suffix,
                            "reasons": ["salad_split_move_to_salad", reason],
                            "prefix": prefix,
                            "suffix": suffix,
                            "moved_to": "\uC0D0\uB7EC\uB4DC",
                            "date": r["date"],
                            "meal_time": r["meal_time"],
                            "corner": r["corner"],
                        }
                    )
                else:
                    new_bar.append(item)
            r["_items"] = new_bar
        # Handle salad lines that should move up to saladbar
        if r["corner"] == "\uC0D0\uB7EC\uB4DC":
            new_salad = []
            for item in r["_items"]:
                m = item["text"]
                if "\uC0D0\uB7EC\uB4DC" in m and ("\uBC25" in m or "\uAE40\uCE58" in m):
                    prefix, suffix, reason = split_salad_text(m)
                    if prefix:
                        saladbar["_items"].append({"text": prefix, "y": item["y"]})
                    if suffix:
                        new_salad.append({"text": suffix, "y": item["y"]})
                    review.append(
                        {
                            "original": m,
                            "cleaned": suffix,
                            "reasons": ["salad_split_move_to_saladbar", reason],
                            "prefix": prefix,
                            "suffix": suffix,
                            "moved_to": "\uC0D0\uB7EC\uB4DC\uBC14",
                            "date": r["date"],
                            "meal_time": r["meal_time"],
                            "corner": r["corner"],
                        }
                    )
                else:
                    new_salad.append(item)
            # Drop label-only lines
            new_salad = [m for m in new_salad if m["text"] != "\uC0D0\uB7EC\uB4DC"]
            r["_items"] = new_salad


def drop_saladbar_main(records):
    for r in records:
        if r["corner"] == "\uC0D0\uB7EC\uB4DC\uBC14":
            r["main"] = ""
            r["sides"] = []


def split_mixed_rice_soup(records, review):
    by_key = {}
    for r in records:
        by_key[(r["date"], r["meal_time"], r["corner"])] = r
    soup_re = re.compile(r"(?=([가-힣]{1,6}(\uAD6D|\uD0D5|\uCC0C\uAC1C)))")
    for r in records:
        if r["corner"] != "\uD55C\uC2DDB" or r["meal_time"] != "\uC11D\uC2DD":
            continue
        rice = by_key.get((r["date"], "\uC911\uC2DD", "\uB77C\uC774\uC2A4&\uB204\uB4E4"))
        if not rice:
            continue
        new_menus = []
        for item in r["_items"]:
            m = item["text"]
            if "\uB36E\uBC25" in m:
                matches = list(soup_re.finditer(m))
                if not matches:
                    new_menus.append(item)
                    continue
                candidates = [
                    mt
                    for mt in matches
                    if "\uBC25" not in mt.group(1) and len(mt.group(1)) >= 3
                ]
                match = candidates[-1] if candidates else matches[-1]
                split_idx = match.start(1)
                if split_idx <= 0:
                    new_menus.append(item)
                    continue
                prefix = m[:split_idx].strip()
                suffix = m[split_idx:].strip()
                if "\uB36E\uBC25" not in prefix:
                    new_menus.append(item)
                    continue
                if prefix:
                    rice["_items"].append({"text": prefix, "y": item["y"]})
                if suffix:
                    new_menus.append({"text": suffix, "y": item["y"]})
                review.append(
                    {
                        "original": m,
                        "cleaned": suffix,
                        "reasons": ["rice_soup_split_move_to_rice"],
                        "moved_to": "\uC911\uC2DD:\uB77C\uC774\uC2A4&\uB204\uB4E4",
                        "date": r["date"],
                        "meal_time": r["meal_time"],
                        "corner": r["corner"],
                    }
                )
            else:
                new_menus.append(item)
        r["_items"] = new_menus




def apply_takeout_merge(records, review):
    takeout_corners_lunch = {
        "\uC0D0\uB7EC\uB4DC",
        "\uBE44\uAC74",
        "\uBC84\uAC70&\uB378\uB9AC",
        "\uB77C\uC774\uC2A4&\uB204\uB4E4",
    }
    takeout_corners_dinner = {
        "\uC0D0\uB7EC\uB4DC",
        "\uBC84\uAC70&\uB378\uB9AC",
    }
    for r in records:
        is_takeout = (
            (r["meal_time"] == "\uC911\uC2DD" and r["corner"] in takeout_corners_lunch)
            or (r["meal_time"] == "\uC11D\uC2DD" and r["corner"] in takeout_corners_dinner)
        )
        if not is_takeout or len(r["_items"]) <= 1:
            continue
        orig_items = sorted(r["_items"], key=lambda it: it["y"])
        orig_texts = [it["text"] for it in orig_items]
        merged = "".join(orig_texts)
        keep_y = orig_items[0]["y"]
        r["_items"] = [{"text": merged, "y": keep_y}]
        review.append(
            {
                "original": "|".join(orig_texts),
                "cleaned": merged,
                "reasons": ["takeout_merge"],
                "date": r["date"],
                "meal_time": r["meal_time"],
                "corner": r["corner"],
            }
        )


def finalize_records(records):
    for r in records:
        r["_items"].sort(key=lambda it: it["y"])
        r["menus"] = [it["text"] for it in r["_items"]]
        r["main"] = r["menus"][0] if r["menus"] else ""
        r["sides"] = r["menus"][1:] if len(r["menus"]) > 1 else []


def parse_one(vision_path: Path, output_path: Path, review_path: Path):
    words, width, height = load_words(vision_path)
    col_bounds, left_boundary = column_bounds(words, height)
    rows = row_bounds(words, height, left_boundary)
    records = []
    review = []
    row_centers = {k: (v[0] + v[1]) / 2 for k, v in rows.items()}
    row_margins = compute_row_margins(rows)
    row_bands = compute_row_bands(rows, row_margins)
    y_min = min(v[0] for v in rows.values()) - 6
    y_max = max(v[1] for v in rows.values()) + 6
    start_date = datetime.strptime("2025-01-13", "%Y-%m-%d")
    for col_idx, (x0, x1) in enumerate(col_bounds):
        date = (start_date + timedelta(days=col_idx)).strftime("%Y-%m-%d")
        col_lines = build_column_lines(words, x0, x1, y_min, y_max)
        row_lines = {k: [] for k in rows.keys()}
        for li in col_lines:
            key = assign_row_key(li["y"], rows, row_centers, row_bands)
            row_lines[key].append(li)
        for key, (y0, y1) in rows.items():
            line_items = row_lines.get(key, [])
            if not line_items:
                continue
            kcal_tokens = [
                w
                for w in words
                if x0 <= w["x"] <= x1 and y0 <= w["y"] <= y1 + 24
            ]
            menus, kcal, main, sides, merged_lines, suspects = parse_cell(
                line_items, kcal_tokens, y0, y1
            )
            if not menus and kcal is None:
                continue
            records.append(
                {
                    "date": date,
                    "meal_time": key[0],
                    "corner": key[1],
                    "menus": menus,
                    "main": main,
                    "sides": sides,
                    "_items": [{"text": m["text"], "y": m["y"]} for m in merged_lines],
                    "kcal": kcal,
                }
            )
            if suspects:
                for s in suspects:
                    s.update(
                        {
                            "date": date,
                            "meal_time": key[0],
                            "corner": key[1],
                        }
                    )
                    review.append(s)
    # Drop dinner rows for Friday (no dinner service)
    filtered = []
    for r in records:
        dt = datetime.strptime(r["date"], "%Y-%m-%d")
        if dt.weekday() == 4 and r["meal_time"] == "\uC11D\uC2DD":
            continue
        filtered.append(r)
    dedup_adjacent(filtered, row_centers)
    split_salad_mixed(filtered, review)
    split_mixed_rice_soup(filtered, review)
    apply_takeout_merge(filtered, review)
    finalize_records(filtered)
    drop_saladbar_main(filtered)
    for r in filtered:
        r.pop("_items", None)
    output_path.write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    review_path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return filtered


if __name__ == "__main__":
    out = Path("records_2025-01-13.json")
    review_out = Path("ocr_review_2025-01-13.json")
    recs = parse_one(Path("vision_2025-01-13.json"), out, review_out)
    print(f"records: {len(recs)}")
    missing = [r for r in recs if r["kcal"] is None]
    print(f"missing kcal: {len(missing)}")
