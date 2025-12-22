import argparse
import base64
import json
import os
import sys
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
                "imageContext": {"languageHints": ["en"]},
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
    key = os.environ.get("GCP_VISION_API_KEY")
    if key:
        return key
    if os.name != "nt":
        return None
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ) as reg_key:
            value, _ = winreg.QueryValueEx(reg_key, "GCP_VISION_API_KEY")
            return value
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Run Google Vision OCR on an image.")
    parser.add_argument(
        "--image",
        default=str(Path("문제") / "ai_top_100_crypto.png"),
        help="Image path to OCR.",
    )
    parser.add_argument(
        "--output",
        default=str(Path("문제") / "gcp_vision_ocr.json"),
        help="Output JSON file.",
    )
    parser.add_argument("--api-key", default=None, help="Vision API key.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout.")
    args = parser.parse_args()

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        print("Missing API key. Set GCP_VISION_API_KEY or use --api-key.", file=sys.stderr)
        return 2

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    payload = build_payload(image_path.read_bytes())
    try:
        response = call_vision_api(api_key, payload, args.timeout)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"error: {exc.code} {body}", file=sys.stderr)
        return 3
    except error.URLError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
