import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


SUFFIXES = ["조림", "볶음", "무침", "구이"]
Q1_CORNERS = {"한식A", "한식B", "팝업A", "팝업B", "양식"}
Q2_CORNERS = {"한식A", "한식B", "양식", "팝업A", "팝업B"}
Q3_REGIONS = ["태국", "베트남", "나가사키", "안동", "전주"]
Q4_MENUS = [
    "덴가스떡볶이",
    "돈코츠라멘",
    "마라탕면",
    "수제남산왕돈까스",
    "탄탄면",
]
Q5_LUNCH_CORNERS = {
    "한식A",
    "한식B",
    "양식",
    "팝업A",
    "팝업B",
    "샐러드",
    "비건",
    "라이스&누들",
    "버거&델리",
}
Q5_DINNER_CORNERS = {"한식B", "샐러드", "버거&델리"}


def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def iter_record_files(records_dir: Path, pattern: str):
    return sorted(records_dir.glob(pattern))


def load_records(records_dir: Path, pattern: str):
    combined = []
    for path in iter_record_files(records_dir, pattern):
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            item = dict(item)
            item["source"] = path.name
            combined.append(item)
    return combined


def write_integrated_view(records, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "records_all.json"
    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = output_dir / "records_all.csv"
    fieldnames = [
        "date",
        "meal_time",
        "corner",
        "main",
        "sides",
        "menus",
        "kcal",
        "source",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            menus = r.get("menus") or []
            sides = r.get("sides")
            if sides is None:
                sides = menus[1:] if len(menus) > 1 else []
            writer.writerow(
                {
                    "date": r.get("date"),
                    "meal_time": r.get("meal_time"),
                    "corner": r.get("corner"),
                    "main": r.get("main") or (menus[0] if menus else ""),
                    "sides": "|".join(sides),
                    "menus": "|".join(menus),
                    "kcal": r.get("kcal"),
                    "source": r.get("source"),
                }
            )
    return json_path, csv_path


def analyze_q1(records):
    start = datetime.strptime("2025-01-13", "%Y-%m-%d")
    end = datetime.strptime("2025-01-17", "%Y-%m-%d")
    counts = {sfx: 0 for sfx in SUFFIXES}
    for r in records:
        if r.get("meal_time") != "중식":
            continue
        corner = r.get("corner")
        if corner not in Q1_CORNERS:
            continue
        date_str = r.get("date")
        if not date_str:
            continue
        dt = parse_date(date_str)
        if dt is None:
            continue
        if dt < start or dt > end:
            continue
        sides = r.get("sides")
        if sides is None:
            menus = r.get("menus") or []
            sides = menus[1:] if len(menus) > 1 else []
        for side in sides:
            for suffix in SUFFIXES:
                if side.endswith(suffix):
                    counts[suffix] += 1
                    break

    ordering = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    order_str = " > ".join([k for k, _ in ordering])
    return {
        "counts": counts,
        "order": order_str,
    }


def analyze_q2(records):
    sums = {c: 0 for c in Q2_CORNERS}
    counts = {c: 0 for c in Q2_CORNERS}
    for r in records:
        if r.get("meal_time") != "중식":
            continue
        corner = r.get("corner")
        if corner not in Q2_CORNERS:
            continue
        date_str = r.get("date")
        if not date_str:
            continue
        dt = parse_date(date_str)
        if dt is None or dt.year != 2025 or dt.month != 1:
            continue
        kcal = r.get("kcal")
        if kcal is None:
            continue
        sums[corner] += kcal
        counts[corner] += 1

    averages = {}
    for corner in Q2_CORNERS:
        if counts[corner] > 0:
            averages[corner] = sums[corner] / counts[corner]
        else:
            averages[corner] = None

    ordering = sorted(
        [(c, v) for c, v in averages.items() if v is not None],
        key=lambda kv: (-kv[1], kv[0]),
    )
    order_str = " > ".join([k for k, _ in ordering])
    return {
        "averages": averages,
        "counts": counts,
        "order": order_str,
    }


def analyze_q3(records):
    counts = {r: 0 for r in Q3_REGIONS}
    for r in records:
        date_str = r.get("date")
        dt = parse_date(date_str)
        if dt is None or dt.year != 2025 or dt.month not in (1, 2):
            continue
        menus = r.get("menus") or []
        for menu in menus:
            for region in Q3_REGIONS:
                if region in menu:
                    counts[region] += 1
    selected = [r for r in Q3_REGIONS if counts[r] >= 2]
    return {
        "counts": counts,
        "selected": selected,
    }


def analyze_q4(records):
    found = {m: [] for m in Q4_MENUS}
    for r in records:
        date_str = r.get("date")
        dt = parse_date(date_str)
        if dt is None or dt.year != 2025 or dt.month not in (1, 2):
            continue
        menus = r.get("menus") or []
        kcal = r.get("kcal")
        if kcal is None:
            continue
        for menu in menus:
            for target in Q4_MENUS:
                if target in menu:
                    found[target].append(kcal)

    resolved = {}
    for target in Q4_MENUS:
        kcals = sorted(set(found[target]))
        resolved[target] = {
            "kcals": kcals,
            "chosen_kcal": max(kcals) if kcals else None,
        }

    ordering = sorted(
        [(m, info["chosen_kcal"]) for m, info in resolved.items() if info["chosen_kcal"]],
        key=lambda kv: (-kv[1], kv[0]),
    )
    order_str = " > ".join([k for k, _ in ordering])
    return {
        "menus": resolved,
        "order": order_str,
    }


def analyze_q5(records):
    by_date = {}
    for r in records:
        date_str = r.get("date")
        if not date_str:
            continue
        dt = parse_date(date_str)
        if dt is None or dt.year != 2025 or dt.month != 2:
            continue
        by_date.setdefault(date_str, []).append(r)

    results = []
    for date_str in sorted(by_date.keys()):
        dt = parse_date(date_str)
        if dt is None:
            continue
        day_records = by_date[date_str]
        lunch = [
            r
            for r in day_records
            if r.get("meal_time") == "중식"
            and r.get("corner") in Q5_LUNCH_CORNERS
            and r.get("kcal") is not None
        ]
        if dt.weekday() == 4:
            if not lunch:
                continue
            lunch_sorted = sorted(lunch, key=lambda r: (r["kcal"], r["corner"]))
            choice = lunch_sorted[0]
            results.append({"id": date_str, "lunch": choice["corner"]})
            continue

        dinner = [
            r
            for r in day_records
            if r.get("meal_time") == "석식"
            and r.get("corner") in Q5_DINNER_CORNERS
            and r.get("kcal") is not None
        ]
        if not lunch or not dinner:
            continue

        best = None
        for l in lunch:
            for d in dinner:
                total = l["kcal"] + d["kcal"]
                diff = abs(total - 1550)
                candidate = (diff, total, l["corner"], d["corner"])
                if best is None or candidate < best[0]:
                    best = (candidate, l, d)
        if best:
            _, l, d = best
            results.append(
                {"id": date_str, "lunch": l["corner"], "dinner": d["corner"]}
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build integrated view and analyze menu questions."
    )
    parser.add_argument(
        "--records-dir",
        default="data/records",
        help="Directory with records_*.json files.",
    )
    parser.add_argument(
        "--pattern",
        default="records_*.json",
        help="Glob pattern for records files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/analysis",
        help="Directory for integrated view and analysis outputs.",
    )
    parser.add_argument(
        "--skip-view",
        action="store_true",
        help="Skip writing integrated view files.",
    )
    args = parser.parse_args()

    records_dir = Path(args.records_dir)
    records = load_records(records_dir, args.pattern)
    output_dir = Path(args.output_dir)

    if not args.skip_view:
        json_path, csv_path = write_integrated_view(records, output_dir)
        print(f"wrote: {json_path}")
        print(f"wrote: {csv_path}")

    q1 = analyze_q1(records)
    q1_path = output_dir / "q1_result.json"
    q1_path.write_text(
        json.dumps(q1, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"q1 order: {q1['order']}")
    print(f"wrote: {q1_path}")

    q2 = analyze_q2(records)
    q2_path = output_dir / "q2_result.json"
    q2_path.write_text(
        json.dumps(q2, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"q2 order: {q2['order']}")
    print(f"wrote: {q2_path}")

    q3 = analyze_q3(records)
    q3_path = output_dir / "q3_result.json"
    q3_path.write_text(
        json.dumps(q3, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"q3 selected: {', '.join(q3['selected']) if q3['selected'] else 'none'}")
    print(f"wrote: {q3_path}")

    q4 = analyze_q4(records)
    q4_path = output_dir / "q4_result.json"
    q4_path.write_text(
        json.dumps(q4, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"q4 order: {q4['order']}")
    print(f"wrote: {q4_path}")

    q5 = analyze_q5(records)
    q5_path = output_dir / "q5_result.json"
    q5_path.write_text(
        json.dumps(q5, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote: {q5_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
