# Digital Coin Teller
**Group:** MissingSabungeros | CIS301  
**Members:** Gabriel Manjares, Franz Leyno, Kester Preferosa, Eljohn De Robles

---

## Supported Coins

This system supports **NGC (New Generation Currency) series coins only.**

| Denomination | Diameter | Material |
|---|---|---|
| ₱1  | 23.0 mm | Nickel-plated steel |
| ₱5  | 25.0 mm | Nickel-plated steel |
| ₱10 | 27.0 mm | Bimetallic |
| ₱20 | 30.0 mm | Bimetallic ring |

> **Note:** Old series coins and ₱0.25 centavo coins are not supported and will show as `Unknown`.

---

## Setup (Do This First)

**1. Make sure Python is installed**  
Download Python 3.10 or higher from https://www.python.org/downloads/  
During installation, check **"Add Python to PATH"**

**2. Clone or download this project folder**  
Put all files in one folder, e.g. `digital_coin_teller/`

**3. Open a terminal inside that folder**  
- Windows: Shift + Right-click the folder → *Open in Terminal*  
- Or just open CMD and `cd` into the folder

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## How to Run

Place your coin image (JPG or PNG) inside the `sample_images/` folder, then run:

```bash
python coin_teller.py -i sample_images/your_image.jpg -r ₱5 -o result.jpg
```

A window will pop up showing the annotated image. Press any key to close it.  
The result is also saved as `result.jpg`.

---

## Image Tips (Important for Accuracy)

**Background**
- Use a **plain dark/black matte surface** — black felt, dark cardboard, or dark cloth work best
- Avoid white or light backgrounds — silver coins (₱1, ₱5, ₱10) blend in and edges become hard to detect
- Avoid patterned or glossy surfaces — patterns cause false circle detections, gloss causes reflections

**Lighting**
- Use **diffused, even lighting** — ceiling light or overcast daylight is ideal
- Avoid direct lamps or flash pointed straight down — coin reflections confuse edge detection
- Tilt your light source slightly off-center to reduce hotspots on metallic surfaces

**Shooting**
- Shoot **top-down** (directly above, not at an angle)
- Coins must **not overlap or stack**
- Recommended image size: **2000–4000px** on the longest side
- Each coin should appear at least **80–100px in diameter** in the image

---

## Arguments

| Argument | What it does | Default |
|---|---|---|
| `-i` | Path to your input image | *(required)* |
| `-r` | Reference coin denomination for calibration | `₱5` |
| `-o` | Output filename for the annotated image | `result.jpg` |
| `--no-display` | Don't open a preview window | off |

**Tip:** The `-r` flag matters. Set it to a denomination you know is in the image for the most accurate size calibration. For example, if your image only has ₱10 and ₱20 coins, pass `-r ₱10`.

---

## Known Limitations

- **Old series coins** — not supported. Pre-NGC coins share near-identical diameters with NGC coins and will misclassify.
- **₱0.25 centavo** — out of circulation, not in the classification table.
- **Overlapping coins** — cannot be separated. Spread coins flat before shooting.
- **Poor lighting** — reflections or shadows will cause missed or false detections.

---

## File Structure

```
digital_coin_teller/
├── coin_teller.py       ← Main script
├── requirements.txt     ← Dependencies
├── README.md            ← This file
└── sample_images/       ← Put your test images here
```