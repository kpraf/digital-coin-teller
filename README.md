# Digital Coin Teller
**Group:** MissingSabungeros | CIS301  
**Members:** Gabriel Manjares, Franz Leyno, Kester Preferosa, Eljohn De Robles

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

- Use a **plain white or dark solid background**
- Coins should **not overlap**
- Good, even lighting — no harsh shadows
- Higher resolution = better results

---

## Arguments

| Argument | What it does | Default |
|---|---|---|
| `-i` | Path to your input image | *(required)* |
| `-r` | Reference coin denomination for calibration | `₱5` |
| `-o` | Output filename for the annotated image | `result.jpg` |
| `--no-display` | Don't open a preview window | off |

---

## File Structure

```
digital_coin_teller/
├── coin_teller.py       ← Main script
├── requirements.txt     ← Dependencies
├── README.md            ← This file
└── sample_images/       ← Put your test images here
```
