cat > README.md <<'EOF'
# RFIC-NeuralSim
### End-to-End Neural Modeling of EM Response and Design Performance for Free-Form RFIC Passives
## Overview

RFIC-NeuralSim is a neural modeling framework for predicting electromagnetic (EM) responses and downstream circuit-level performance metrics of pixelated RFIC passive matching networks.

The framework combines:

- **Frequency-conditioned neural simulator:** Predicts frequency-dependent EM responses (S-parameters) from passive layout geometry.
- **Circuit-conditioned neural solver:** Predicts passive-network performance metrics under varying circuit contexts.
- **Complementary response representation:** Enhances downstream performance prediction beyond the explicit EM-response representation.

The framework is evaluated on dual-metal-layer pixelated passive matching networks for input, interstage, and output matching applications in RFIC power amplifiers.

## Code Availability

This repository accompanies our NeurIPS 2026 paper.

The source code, environment configuration, and instructions for training and evaluation will be provided following finalization of the code release.

## Data Availability

The original RFIC layout-mask dataset and associated proprietary EM-simulation data are not publicly released.

**NeurIPS 2026**

**Authors:** Yuhao Mao*, Chenhao Chu*, Martin Vechev, and Hua Wang

*Equal contribution.

## Citation

Yuhao Mao, Chenhao Chu, Martin Vechev, and Hua Wang,
"End-to-End Neural Modeling of EM Response and Design Performance for Free-Form RFIC Passives,"
Advances in Neural Information Processing Systems (NeurIPS), 2026.

The official publication link and BibTeX entry will be added when available.

## Acknowledgment

This research is a collaboration between the Integrated Devices, Electronics, and Systems (IDEAS) Group and the Secure, Reliable, and Intelligent Systems (SRI) Lab at ETH Zurich.
EOF