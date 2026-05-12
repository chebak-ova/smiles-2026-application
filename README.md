# SMILES 2026: Signal Interference Cancellation

Applicant solution for the **SMILES 2026 Signal Processing Challenge**. This repository implements a robust interference cancellation pipeline for multi-channel RX signals using TX reference data.

## 📖 Task Overview
Given transmitted (`TX`) and received (`RX`) complex baseband signals, the goal is to estimate and subtract structured interference composed of:
1. Nonlinear cross-channel coupling from the transmitter
2. A spatially coherent external interference source

Performance is evaluated by the average SNR improvement (dB) in the interference band.

## 📁 Project Structure
```text
smiles-2026-signal-solution/
├── applicant_solution.py    # Main implementation (your_canceller function)
├── SOLUTION.md              # Detailed methodology, experiments & reproducibility guide
├── requirements.txt         # Python dependencies (numpy, scipy)
├── .gitignore               # Excluded files (cache, challenge.mat, etc.)
└── results.json             # Auto-generated evaluation metrics
```

## 🚀 How to Run
1. **Prepare environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate   # Windows
   pip install -r requirements.txt
