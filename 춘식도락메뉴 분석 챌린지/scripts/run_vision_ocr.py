import argparse
import base64
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib import error, request

VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate?key="


def build_payload(image_bytes):
    content = base64.b64encode(image_bytes).decode("ascii")
    return {
        "requests": [
            {
                "image": {"content": content},
                "features": [{"type": "TEXT_DETECTION"}],
                "imageContext": {"languageHints": ["ko", "en"]},
            }
        ]
    }


def call_vision_api(api_key, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        VISION_ENDPOINT + api_key,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_api_key(cli_key):
    if cli_key:
        return cli_key
    return os.environ.get("VISION_API_KEY")


def iter_images(input_path, pattern):
    path = Path(input_path)
    if path.is_file():
        return [path]
    return sorted(path.glob(pattern))


def fill_transparency_white(image_bytes):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to process transparent PNGs. Install it with: pip install Pillow"
        ) from exc

    img = Image.open(BytesIO(image_bytes))
    has_alpha = img.mode in ("LA", "RGBA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if not has_alpha:
        return image_bytes

    if img.mode != "RGBA":
        img = img.convert("RGBA")
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    background.alpha_composite(img)
    img = background.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="Create Google Vision OCR JSON files from menu images."
    )
    parser.add_argument(
        "--input",
        default="ai_top_100_menu/menu_images",
        help="Image file or directory.",
    )
    parser.add_argument(
        "--pattern",
        default="*.png",
        help="Glob pattern when --input is a directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write vision_YYYY-MM-DD.json files.",
    )
    parser.add_argument("--api-key", default=None, help="Vision API key.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between requests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned outputs without calling the API.",
    )
    args = parser.parse_args()

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        print("Missing API key. Use --api-key or set VISION_API_KEY.", file=sys.stderr)
        return 2

    images = iter_images(args.input, args.pattern)
    if not images:
        print("No images found.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(images):
        stem = image_path.stem
        out_path = output_dir / f"vision_{stem}.json"
        if out_path.exists() and not args.overwrite:
            print(f"skip: {out_path}")
            continue
        if args.dry_run:
            print(f"plan: {image_path} -> {out_path}")
            continue
        image_bytes = image_path.read_bytes()
        image_bytes = fill_transparency_white(image_bytes)
        payload = build_payload(image_bytes)
        try:
            response = call_vision_api(api_key, payload, args.timeout)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"error: {image_path} ({exc.code}) {body}", file=sys.stderr)
            return 3
        except error.URLError as exc:
            print(f"error: {image_path} ({exc})", file=sys.stderr)
            return 4

        out_path.write_text(
            json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote: {out_path}")
        if args.sleep and idx < len(images) - 1:
            time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
