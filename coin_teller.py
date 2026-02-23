"""
Digital Coin Teller
Group: MissingSabungeros | CIS301
Members: Gabriel Manjares, Franz Leyno, Kester Preferosa, Eljohn De Robles

Detects and classifies Philippine Peso coins from a static image using:
  - Grayscale & Gaussian Blur (noise reduction)
  - Hough Circle Transform (shape detection)
  - Pixel-to-mm calibration (size measurement)
  - Diameter-based classification (coin lookup table)

SCOPE & LIMITATIONS:
  - Supports NEW GENERATION CURRENCY (NGC) series coins only.
    Old series coins (pre-BSP redesign) share near-identical diameters
    with each other and with NGC coins, making diameter-only classification
    unreliable. Mixed old/new sets WILL produce misclassifications.
  - ₱0.25 (25-centavo) coins are NOT supported — out of active circulation.
  - Overlapping coins cannot be separated by Hough Circle Transform.
  - Requires clear, even lighting and a plain solid-colored background.

SUPPORTED COINS (NGC Series):
  ₱1  — 23.0 mm — Nickel-plated steel
  ₱5  — 25.0 mm — Nickel-plated steel
  ₱10 — 27.0 mm — Bimetallic
  ₱20 — 30.0 mm — Bimetallic ring (bronze outer, nickel center)
"""

import cv2
import numpy as np
import argparse
import sys
from collections import defaultdict

# ─────────────────────────────────────────────
# PHILIPPINE PESO COIN SPECIFICATIONS (in mm)
# NGC (New Generation Currency) series ONLY.
# Old series coins are a known limitation — see module docstring.
# ─────────────────────────────────────────────
COIN_SPECS = {
    "₱1":  {"diameter_mm": 23.0, "tolerance_mm": 1.0, "color": (192, 192, 192)},   # nickel-plated steel, silver
    "₱5":  {"diameter_mm": 25.0, "tolerance_mm": 1.0, "color": (192, 192, 192)},   # nickel-plated steel, silver
    "₱10": {"diameter_mm": 27.0, "tolerance_mm": 1.0, "color": (192, 192, 192)},   # bimetallic, silver
    "₱20": {"diameter_mm": 30.0, "tolerance_mm": 1.8, "color": (0,   200, 100)},   # bimetallic ring, wider tolerance for calibration drift
}

# Default fallback tolerance if a coin entry is missing tolerance_mm.
# Per-coin tolerances in COIN_SPECS take priority.
# ₱20 uses 1.8mm to account for calibration drift on the larger diameter.
TOLERANCE_MM = 1.0

# ─────────────────────────────────────────────
# COIN VALUES
# ─────────────────────────────────────────────
COIN_VALUES = {"₱1": 1, "₱5": 5, "₱10": 10, "₱20": 20}


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Could not load image: {path}")
        sys.exit(1)
    return img


def preprocess(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale and apply Gaussian blur."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    return blurred


def detect_circles(blurred: np.ndarray, img_height: int) -> np.ndarray | None:
    """
    Apply Hough Circle Transform.
    minRadius / maxRadius are set as a fraction of image height
    to be somewhat scale-invariant.
    """
    min_r = int(img_height * 0.04)   # ~4% of height
    max_r = int(img_height * 0.22)   # ~22% of height

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(img_height * 0.08),   # coins must be this far apart
        param1=60,    # upper Canny threshold
        param2=100,    # accumulator threshold (lower = more detections)
        minRadius=min_r,
        maxRadius=max_r,
    )
    return circles


def calibrate(circles: np.ndarray, ref_diameter_mm: float = 25.0) -> float:
    """
    Calibration: use the MEDIAN detected circle radius as the reference.
    Assumes the reference coin (default: ₱5 NGC = 25mm) is present.
    Returns pixels-per-mm ratio.

    For best accuracy, pass --ref-coin with the denomination of a coin
    you know is in the image.

    NOTE: Calibration assumes the majority of coins in the image match
    the reference denomination. For mixed-denomination sets, use a single
    known reference coin placed prominently in the frame.
    """
    radii = circles[0, :, 2]
    median_radius_px = float(np.median(radii))
    ref_radius_mm = ref_diameter_mm / 2.0
    px_per_mm = median_radius_px / ref_radius_mm
    print(f"[CALIBRATION] Median radius: {median_radius_px:.1f}px | "
          f"Reference coin: {ref_diameter_mm}mm | "
          f"Scale: {px_per_mm:.3f} px/mm")
    return px_per_mm


def classify_coin(diameter_mm: float) -> str:
    """
    Match a measured diameter to a known NGC denomination.
    Each coin uses its own tolerance_mm from COIN_SPECS (falls back to
    the global TOLERANCE_MM if not set). Returns 'Unknown' if no match.
    Old series coins and ₱0.25 will return 'Unknown'.
    """
    best_match = "Unknown"
    best_diff = float("inf")
    for denomination, specs in COIN_SPECS.items():
        tol = specs.get("tolerance_mm", TOLERANCE_MM)
        diff = abs(diameter_mm - specs["diameter_mm"])
        if diff < best_diff and diff <= tol:
            best_diff = diff
            best_match = denomination
    return best_match


def annotate_and_summarize(
    img: np.ndarray,
    circles: np.ndarray,
    px_per_mm: float,
) -> tuple[np.ndarray, dict, float]:
    """Draw circles and labels; return annotated image + tally."""
    output = img.copy()
    tally = defaultdict(int)

    circles_rounded = np.round(circles[0, :]).astype(int)

    for (x, y, r) in circles_rounded:
        diameter_mm = (r * 2) / px_per_mm
        denomination = classify_coin(diameter_mm)
        tally[denomination] += 1

        color = COIN_SPECS.get(denomination, {}).get("color", (200, 200, 200))

        # Draw outer circle
        cv2.circle(output, (x, y), r, color, 3)
        # Draw center dot
        cv2.circle(output, (x, y), 4, (0, 0, 255), -1)

        # Label: denomination + measured mm
        label_denom = denomination
        label_size  = f"{diameter_mm:.1f}mm"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, r / 60)
        thickness  = 2

        # Center text above the coin center
        (tw, th), _ = cv2.getTextSize(label_denom, font, font_scale, thickness)
        cv2.putText(output, label_denom,
                    (x - tw // 2, y - 8),
                    font, font_scale, (255, 255, 255), thickness + 2)   # outline
        cv2.putText(output, label_denom,
                    (x - tw // 2, y - 8),
                    font, font_scale, color, thickness)

        (tw2, th2), _ = cv2.getTextSize(label_size, font, font_scale * 0.75, 1)
        cv2.putText(output, label_size,
                    (x - tw2 // 2, y + th2 + 6),
                    font, font_scale * 0.75, (200, 200, 200), 1 + 1)
        cv2.putText(output, label_size,
                    (x - tw2 // 2, y + th2 + 6),
                    font, font_scale * 0.75, (255, 255, 255), 1)

    # Calculate total
    total = sum(COIN_VALUES.get(d, 0) * count for d, count in tally.items())
    return output, dict(tally), total


def draw_summary_box(img: np.ndarray, tally: dict, total: float) -> np.ndarray:
    """Overlay a summary panel in the top-left corner."""
    lines = ["=== COIN SUMMARY ==="]
    for denom in sorted(tally.keys(), key=lambda d: COIN_VALUES.get(d, 0)):
        count = tally[denom]
        value = COIN_VALUES.get(denom, 0) * count
        lines.append(f"  {denom} x{count}  =  ₱{value}")
    lines.append("─" * 22)
    lines.append(f"  TOTAL  =  ₱{total:.0f}")

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 1
    line_h     = 28
    padding    = 12

    box_w = 260
    box_h = line_h * len(lines) + padding * 2

    # Semi-transparent dark overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    for i, line in enumerate(lines):
        y = 10 + padding + line_h * i + 18
        color = (0, 255, 180) if "TOTAL" in line else (240, 240, 240)
        cv2.putText(img, line, (18, y), font, font_scale, color, thickness)

    return img


def print_console_summary(tally: dict, total: float, px_per_mm: float) -> None:
    print("\n" + "=" * 40)
    print("       DIGITAL COIN TELLER — RESULTS")
    print("=" * 40)
    for denom in sorted(tally.keys(), key=lambda d: COIN_VALUES.get(d, 0)):
        count = tally[denom]
        value = COIN_VALUES.get(denom, 0) * count
        print(f"  {denom:5s}  ×{count:>3}  =  ₱{value:>6}")
    print("─" * 40)
    unknown = tally.get("Unknown", 0)
    if unknown:
        print(f"  Unknown coins (unclassified): {unknown}")
        print(f"  [NOTE] Unknown coins may be old series or non-NGC denominations.")
    print(f"  TOTAL COIN COUNT : {sum(tally.values())}")
    print(f"  TOTAL VALUE      : ₱{total:.0f}")
    print("=" * 40 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Digital Coin Teller — Philippine Peso coin detector\n"
            "Supports NGC (New Generation Currency) series coins only:\n"
            "  ₱1 (23mm), ₱5 (25mm), ₱10 (27mm), ₱20 (30mm)\n\n"
            "Known limitations:\n"
            "  - Old series coins will likely be misclassified\n"
            "  - ₱0.25 coins are not supported (out of circulation)\n"
            "  - Overlapping coins cannot be detected separately"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--image", required=True,
        help="Path to the input image (JPG or PNG)"
    )
    parser.add_argument(
        "-r", "--ref-coin", default="₱5",
        choices=list(COIN_SPECS.keys()),
        help="Denomination of the reference coin used for calibration (default: ₱5 NGC = 25mm)"
    )
    parser.add_argument(
        "-o", "--output", default="result.jpg",
        help="Path to save the annotated output image (default: result.jpg)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Skip showing the result window (useful for headless environments)"
    )
    args = parser.parse_args()

    # 1. Load
    img = load_image(args.image)
    h, w = img.shape[:2]
    print(f"[INFO] Image loaded: {w}×{h} px")
    print(f"[INFO] Mode: NGC series only | Tolerance: ±{TOLERANCE_MM}mm")

    # 2. Preprocess
    blurred = preprocess(img)

    # 3. Detect circles
    circles = detect_circles(blurred, h)
    if circles is None:
        print("[WARNING] No circles detected. Try adjusting lighting or background.")
        sys.exit(0)
    print(f"[INFO] Detected {len(circles[0])} circle(s).")

    # 4. Calibrate scale
    ref_diameter_mm = COIN_SPECS[args.ref_coin]["diameter_mm"]
    px_per_mm = calibrate(circles, ref_diameter_mm)

    # 5. Classify & annotate
    annotated, tally, total = annotate_and_summarize(img, circles, px_per_mm)

    # 6. Draw summary box on image
    annotated = draw_summary_box(annotated, tally, total)

    # 7. Console output
    print_console_summary(tally, total, px_per_mm)

    # 8. Save result
    cv2.imwrite(args.output, annotated)
    print(f"[INFO] Annotated image saved to: {args.output}")

    # 9. Display
    if not args.no_display:
        cv2.imshow("Digital Coin Teller — Result", annotated)
        print("[INFO] Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


main()