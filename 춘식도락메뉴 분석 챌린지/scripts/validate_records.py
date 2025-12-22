import argparse
import json
from pathlib import Path


RULES = {
    ("중식", "한식A"): [6],
    ("중식", "한식B"): [6],
    ("중식", "팝업A"): [5, 6],
    ("중식", "팝업B"): [5],
    ("중식", "양식"): [5],
    ("중식", "샐러드바"): [4],
    ("중식", "샐러드"): [1],
    ("중식", "비건"): [1],
    ("중식", "버거&델리"): [1],
    ("중식", "라이스&누들"): [1],
    ("석식", "한식B"): [6],
    ("석식", "샐러드바"): [3],
    ("석식", "샐러드"): [1],
    ("석식", "버거&델리"): [1],
}


def iter_record_files(input_path, pattern):
    path = Path(input_path)
    if path.is_file():
        return [path]
    return sorted(path.glob(pattern))


def validate_records(records, source):
    violations = []
    for record in records:
        meal_time = record.get("meal_time")
        corner = record.get("corner")
        expected = RULES.get((meal_time, corner))
        if not expected:
            continue
        menus = record.get("menus") or []
        actual = len(menus)
        if actual not in expected:
            violations.append(
                {
                    "date": record.get("date"),
                    "meal_time": meal_time,
                    "corner": corner,
                    "actual_count": actual,
                    "expected_counts": expected,
                    "menus": menus,
                    "source": source,
                }
            )
    return violations


def write_report(violations, output_path):
    output_path.write_text(
        json.dumps(violations, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate menu counts in parsed records."
    )
    parser.add_argument(
        "--input",
        default="data/records",
        help="Record file or directory.",
    )
    parser.add_argument(
        "--pattern",
        default="records_*.json",
        help="Glob pattern when --input is a directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/validation",
        help="Directory for validation reports.",
    )
    args = parser.parse_args()

    record_files = iter_record_files(args.input, args.pattern)
    if not record_files:
        print("No record files found.")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_violations = 0
    for record_path in record_files:
        data = json.loads(record_path.read_text(encoding="utf-8"))
        violations = validate_records(data, record_path.name)
        report_name = record_path.name.replace("records_", "validation_")
        report_path = output_dir / report_name
        write_report(violations, report_path)
        total_violations += len(violations)
        print(
            f"{record_path.name}: records={len(data)} violations={len(violations)} -> {report_path}"
        )

    print(f"total violations: {total_violations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
