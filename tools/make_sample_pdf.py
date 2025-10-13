from __future__ import annotations

import argparse
from pathlib import Path


def make_text_pdf(path: Path) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception as e:
        raise SystemExit(
            "reportlab is required for --out-text; install with: pip install reportlab"
        ) from e

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawString(72, 740, "Oscillink Ingest Benchmark PDF (text)")
    for i in range(1, 20):
        c.drawString(72, 740 - 20 * i, f"Line {i}: The quick brown fox jumps over the lazy dog {i}.")
    c.showPage()
    c.save()


def make_ocr_pdf(path: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        raise SystemExit(
            "Pillow is required for --out-ocr; install with: pip install Pillow"
        ) from e

    path.parent.mkdir(parents=True, exist_ok=True)
    # Create a white image with black text so text is not embedded (forces OCR)
    img = Image.new("RGB", (1654, 2339), color="white")  # ~A4 at 200 DPI
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    draw.text((72, 72), "Oscillink Ingest Benchmark PDF (image-only)", fill="black", font=font)
    y = 120
    for i in range(1, 40):
        draw.text((72, y), f"OCR Line {i}: The quick brown fox jumps over the lazy dog {i}.", fill="black", font=font)
        y += 28
    # Save as PDF (image-only)
    img.save(str(path), "PDF")


def main() -> int:
    ap = argparse.ArgumentParser(description="Create sample PDFs for ingest benchmarks")
    ap.add_argument("--out-text", type=Path, help="Write a text-embedded PDF to this path")
    ap.add_argument("--out-ocr", type=Path, help="Write an image-only PDF (forces OCR) to this path")
    args = ap.parse_args()

    if not args.out_text and not args.out_ocr:
        ap.error("Provide at least one of --out-text or --out-ocr")

    if args.out_text:
        make_text_pdf(args.out_text)
        print(f"Wrote text PDF: {args.out_text}")
    if args.out_ocr:
        make_ocr_pdf(args.out_ocr)
        print(f"Wrote OCR PDF:  {args.out_ocr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
