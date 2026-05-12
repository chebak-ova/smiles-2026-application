# SMILES 2026: Signal Interference Cancellation — Solution Report

## 1. Task Understanding
The goal is to estimate and subtract structured interference from received signals (RX) using transmitted signals (TX) as reference. The interference has two components: (1) a nonlinear function of all TX channels jointly (cross-channel coupling), and (2) a spatially coherent external source appearing across all 4 RX channels with different amplitude/phase. The scoring metric is average dB improvement in a narrow interference band.

## 2. Approach
### 2.1 Nonlinear Feature Construction
Instead of assuming linear coupling (as in the baseline), I built an expanded feature matrix from TX:
- **Linear terms**: all 6 TX channels
- **Magnitude-squared**: |tx_i|² to capture AM-AM distortion in power amplifiers
- **Cross-products**: tx_i · conj(tx_j) for selected pairs to model intermodulation products

This follows standard practice in RF interference modeling where nonlinearities generate sum/difference frequencies.

### 2.2 Per-Channel Estimation
For each RX channel, I solved a regularized least-squares problem: w = argmin ||X @ w - rx_c||² + λ||w||²
with λ=1e-4 to prevent overfitting given the high feature dimensionality.

### 2.3 Spatial Coherence Refinement
After removing TX-driven interference, residuals still contain the external component E. Since E is spatially coherent (rank-1 across RX), I:
1. Computed the covariance matrix of residuals across channels
2. Extracted the dominant eigenvector (spatial signature)
3. Projected residuals onto this signature to estimate the temporal waveform of E
4. Subtracted the rank-1 reconstruction from each channel

This two-stage approach separates device-specific nonlinear leakage from external coherent interference.

## 3. Reproducibility
### Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install numpy scipy
