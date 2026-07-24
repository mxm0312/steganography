# Steganography Benchmark Report

**Date:** 2026-07-24T18:24:33  
**OS:** Linux-6.17.0-35-generic-x86_64-with-glibc2.39  
**CPU:** x86_64  
**GPU:** NVIDIA GeForce RTX 5080 (15.5 GB VRAM)  

> **Reproducibility note:** synthetic cosine+noise fixtures, fixed seeds,
> timings = mean over 3 repeat(s) per trial.

---

## Results

| Method | Scenario | Resolution | Frames | Pack fps | Extract fps | Pack MB/s | PSNR stego ↑ | PSNR secret ↑ | Bit-exact | Fill | BPP | Overhead | RAM MB | GPU MB | Device |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|:---|
| lfvsn | tiny | 64×48 | 10 | 16.35 | 42.09 | 0.09 | 35.99 dB | 35.21 dB | — | 1.8% | 0.4424 | -10.7% | 2.7 | 56.7 | cuda |
| lfvsn | small | 320×240 | 30 | 28.80 | 40.13 | 3.61 | 36.63 dB | 39.92 dB | — | 0.1% | 0.0338 | -12.1% | 20.7 | 139.4 | cuda |
| lfvsn | medium | 640×480 | 60 | 9.60 | 10.64 | 4.80 | 36.65 dB | 40.46 dB | — | 0.1% | 0.0166 | -12.1% | 159.2 | 461.4 | cuda |
| lfvsn | large | 1280×720 | 90 | 2.80 | 2.89 | 4.19 | 36.65 dB | 40.62 dB | — | 0.0% | 0.0110 | -12.2% | 714.6 | 1327.0 | cuda |
| lsb | tiny | 64×48 | 10 | 774.75 | 1047.94 | 4.75 | 54.11 dB | — | ✓ | 50.0% | 1.4937 | +0.1% | 0.6 | 0.0 | cpu |
| lsb | small | 320×240 | 30 | 218.23 | 466.23 | 31.18 | 54.15 dB | — | ✓ | 50.0% | 1.4999 | +0.1% | 40.0 | 0.0 | cpu |
| lsb | medium | 640×480 | 60 | 79.79 | 171.86 | 45.44 | 54.15 dB | — | ✓ | 50.0% | 1.5000 | +0.1% | 319.8 | 0.0 | cpu |
| lsb | large | 1280×720 | 90 | 33.06 | 85.57 | 56.46 | 54.15 dB | — | ✓ | 50.0% | 1.5000 | +0.1% | 1438.7 | 0.0 | cpu |
| steganogan | tiny | 64×48 | 10 | — | — | — | — | — | — | — | — | — | — | — | ERROR: pack(): нужно 16575 Б, доступно 3072 Б |
| steganogan | small | 320×240 | 1 | 6.03 | 1.74 | 0.78 | 33.56 dB | — | — | 50.0% | 0.0159 | -11.0% | 9.8 | 94.5 | cuda |
| steganogan | medium | 640×480 | 1 | 3.53 | 0.59 | 1.82 | 33.60 dB | — | — | 50.0% | 0.0184 | -11.2% | 38.8 | 377.5 | cuda |
| steganogan | large | 1280×720 | 1 | 1.48 | 0.20 | 2.28 | 33.62 dB | — | — | 50.0% | 0.0190 | -11.2% | 116.2 | 1132.9 | cuda |
---

## Metric definitions

| Metric | Formula / meaning |
|:---|:---|
| **PSNR stego** | `10·log₁₀(255²/MSE(cover,stego))` — invisibility. >40 dB imperceptible. |
| **PSNR secret** | `10·log₁₀(255²/MSE(orig,recovered))` — recovery fidelity (neural only). |
| **Bit-exact** | Recovered payload is bit-for-bit identical to the original (LSB only). |
| **Fill** | `secret_bytes / usable_capacity` — fraction of container capacity used. |
| **BPP** | `secret_bits / (W × H × frames)` — bits of secret per pixel. |
| **Overhead** | `(stego_size − cover_size) / cover_size` — relative file size increase. |
| **Pack MB/s** | Stego output size (MB) per second of encoding time. |
