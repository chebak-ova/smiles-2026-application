import json
import numpy as np
from scipy import signal
from task_and_baseline import load_data, compute_score, baseline_canceller


def your_canceller(tx: np.ndarray, rx: np.ndarray, fs: float = 7.68e6) -> np.ndarray:
    """
    Estimate and subtract interference from received signals.
    
    Signal model: rx = s + I + eta, where I = F_c(TX) + E
    - F_c(TX): nonlinear function of transmitted signals (cross-channel coupling)
    - E: spatially coherent external interference (rank-1 across RX channels)
    
    Approach:
    1. Build nonlinear feature matrix from TX: linear terms + cross-products + magnitude squares
    2. Estimate per-channel interference via regularized least squares
    3. Extract spatially coherent residual component via SVD (rank-1 approximation)
    4. Subtract combined estimate from rx
    
    Args:
        tx: (N, 6) complex128 — transmitted signals
        rx: (N, 4) complex128 — received (corrupted) signals
        fs: sampling frequency in Hz
    
    Returns:
        rx_hat: (N, 4) complex128 — cleaned received signals
    """
    N, n_tx = tx.shape
    n_rx = rx.shape[1]
    
    # --- Step 1: Build nonlinear feature matrix from TX ---
    # Linear terms (6 channels)
    features = [tx]
    
    # Magnitude-squared terms (capture AM-AM distortion): |tx_i|^2
    for i in range(n_tx):
        features.append(np.abs(tx[:, i:i+1])**2)
    
    # Cross-product terms (capture intermodulation): tx_i * conj(tx_j) for i<j
    # Limit to strongest pairs to avoid overfitting (6 choose 2 = 15, we take top 6)
    cross_pairs = [(0,1), (0,2), (1,2), (3,4), (3,5), (4,5)]
    for i, j in cross_pairs:
        features.append(tx[:, i:i+1] * np.conj(tx[:, j:j+1]))
    
    # Stack all features: shape (N, n_features)
    X = np.hstack(features)
    n_feat = X.shape[1]
    
    # --- Step 2: Per-channel interference estimation via regularized LS ---
    rx_hat = np.zeros_like(rx)
    reg = 1e-4  # Tikhonov regularization to prevent overfitting
    
    for c in range(n_rx):
        y = rx[:, c]
        
        # Solve: min ||X @ w - y||^2 + reg * ||w||^2
        # Closed form: w = (X'H X + reg*I)^{-1} X'H y
        XtX = X.conj().T @ X
        Xty = X.conj().T @ y
        w = np.linalg.solve(XtX + reg * np.eye(n_feat), Xty)
        
        # Predicted interference from TX-driven nonlinear component
        I_tx = X @ w
        
        # --- Step 3: Extract spatially coherent external component ---
        # Residual after removing TX-driven part should contain E + noise
        residual = y - I_tx
        
        # For external interference E: same source across RX channels with different scaling
        # We'll refine this in a second pass using multi-channel coherence
        rx_hat[:, c] = residual
    
    # --- Step 4: Joint refinement using spatial coherence across RX channels ---
    # Stack residuals: (N, 4) — external interference should be rank-1
    residuals = rx - (np.zeros_like(rx) if False else rx_hat)  # placeholder
    
    # Actually, re-estimate external component from initial residuals
    # Compute covariance across channels and extract dominant eigenvector
    R = np.cov(rx_hat.T, rowvar=False)  # (4, 4) covariance
    evals, evecs = np.linalg.eigh(R)
    
    # Dominant eigenvector = spatial signature of coherent external interference
    v_ext = evecs[:, -1]  # (4,)
    
    # Project residuals onto this signature to estimate temporal waveform of E
    e_temporal = np.zeros(N, dtype=complex)
    for c in range(n_rx):
        e_temporal += np.conj(v_ext[c]) * rx_hat[:, c]
    e_temporal /= np.sum(np.abs(v_ext)**2)
    
    # Subtract rank-1 external component from each channel
    for c in range(n_rx):
        rx_hat[:, c] -= v_ext[c] * e_temporal
    
    return rx_hat


def main():
    """Main entrypoint: load data, run baseline and solution, save results."""
    print("Loading challenge data...")
    tx, rx, fs, fc_tx = load_data("challenge.mat")
    print(f"  TX shape: {tx.shape}, RX shape: {rx.shape}, Fs: {fs/1e6:.2f} MHz")
    
    # Run provided baseline for reference
    print("Computing baseline cancellation...")
    rx_baseline = baseline_canceller(tx, rx)
    baseline_score = compute_score(rx, rx_baseline, fs)
    print(f"  Baseline score: {baseline_score['average_db']:.2f} dB")
    
    # Run applicant solution
    print("Running applicant canceller...")
    rx_hat = your_canceller(tx, rx, fs)
    
    # Score our solution
    your_score = compute_score(rx, rx_hat, fs)
    print(f"  Your score: {your_score['average_db']:.2f} dB")
    print(f"  Per-channel: {[f'{v:.2f}' for v in your_score['per_channel_db']]} dB")
    
    # Save results in required format
    results = {
        "baseline": {
            "per_channel_db": [float(v) for v in baseline_score["per_channel_db"]],
            "average_db": float(baseline_score["average_db"])
        },
        "yours": {
            "per_channel_db": [float(v) for v in your_score["per_channel_db"]],
            "average_db": float(your_score["average_db"])
        }
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results.json")


if __name__ == "__main__":
    main()
