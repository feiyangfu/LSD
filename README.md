#  [NeurIPS'25] Learnable Sampler Distillation for Discrete Diffusion Models
This repository officially houses the PyTorch implementation of the paper titled "Learnable Sampler Distillation for Discrete Diffusion Models", which is presented at NeurIPS 2025 as a spotlight.

# Installation

Please refer to [JYS](https://github.com/enkeejunior1/jump-your-steps) for Installation.

# Pipeline
```bash
# 1. Generate teacher data
./train_teacher.sh

# 2. Train the student sampler
./train_student.sh

```
# References
This repository is heavily based on 

- [CTMC](https://github.com/andrew-cr/tauLDR/tree/main)  
- [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)
- [JYS](https://github.com/enkeejunior1/jump-your-steps)



