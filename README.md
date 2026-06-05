# MSR-Net: Multi-Scale Structural Re-parameterization for Real-Time Infrared Small Target Detection

Official PyTorch implementation of **"MSR-Net: Multi-Scale Structural Re-parameterization with Scale-Sensitive Loss for Real-Time Infrared Small Target Detection."**

> A highly parameter-efficient architecture for infrared small target detection (IRSTD) that achieves a strong speed–accuracy trade-off on resource-constrained edge devices. MSR-Net reaches **238 FPS on an NVIDIA Jetson Orin NX**, attains the **highest IoU on IRSTD-1k (70.77%)**, and records the **lowest false-alarm rate on both benchmarks** — while reducing parameters and MACs by **39×** and **177×** respectively compared to ISNet.

<p align="center">
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.8+-blue.svg"></a>
  <a href="#"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg"></a>
  <a href="#license"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

---

## Table of Contents

- [Highlights](#highlights)
- [Method Overview](#method-overview)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Highlights

- **Multi-Scale Convolutional Re-parameterization Block (MSCRB).** A heterogeneous, IRSTD-tailored multi-branch block that couples multi-scale square kernels with ACNet-style asymmetric kernels (`1×K`, `K×1`) to explicitly model directional background clutter (cloud edges, horizon lines, sea clutter). The multi-branch training topology fuses into a **single-path convolution at inference** via structural re-parameterization — enhanced representation with **zero added inference latency**.
- **Progressive Inverse Kernel-Dilation Scaling.** Large, dense kernels in shallow layers preserve fine target structure; small, heavily dilated kernels in deeper layers expand the receptive field for global context — without the quadratic parameter cost of dense large kernels.
- **Scale-Sensitive Gated (SSG) Loss.** Combines a curriculum-gated soft-intersection term, a quadratic false-alarm penalty, and a fractional gradient-modulation term (with a formally bounded gradient) for stable optimization of tiny targets under severe class imbalance.
- **Real-time edge deployment.** 0.025M parameters / 0.691 GMACs after fusion; 4.2 ms latency (238 FPS) on Jetson Orin NX with TensorRT FP16.

---

## Method Overview

MSR-Net adopts a parameter-efficient U-shaped encoder–decoder topology augmented with structural re-parameterization:

1. **Stem** — a 7×7 stride-2 convolution that suppresses high-frequency noise and downsamples.
2. **Hierarchical encoder** — hybrid downsampling (a learnable bottleneck path concatenated with a non-learnable max-pooling path) preserves sub-pixel peak responses while expanding channels.
3. **Semantic Context Bridge (Common Block)** — a cascade of Depthwise → Dilated → Regular → Dilated bottlenecks with a monotonically increasing dilation schedule (`d ∈ {6, 8}`) for long-range context.
4. **Symmetric decoder** — early fusion of deep semantic and shallow spatial features before transposed-convolution upsampling.
5. **Deep supervision** — the SSG Loss is applied to three auxiliary decoder outputs plus the final fused prediction (`Ns = 4`).

During inference, every MSCRB branch (including the identity path) is mathematically folded into one equivalent `K×K` convolution.

---

## Results

All models re-trained from scratch under identical conditions (512×512 inputs, same partitions, optimizer, and schedule). `Fa` is reported in units of ×10⁻⁶ (lower is better).

### Detection performance

| Dataset    | IoU (%) | nIoU (%) | Pd (%) | Fa (↓) |
|------------|:-------:|:--------:|:------:|:------:|
| IRSTD-1k   | **70.77** | 63.60  | 94.61  | **3.20** |
| SISD-Sky   | 70.21   | 64.08    | 66.52  | **10.0** |

MSR-Net achieves the highest IoU and lowest false-alarm rate on IRSTD-1k, and the lowest false-alarm rate on SISD-Sky.

### Efficiency (512×512 input, Jetson Orin NX 16G, TensorRT FP16)

| Model               | MACs (G) | Params (M) | Latency (ms) | FPS |
|---------------------|:--------:|:----------:|:------------:|:---:|
| ISNet               | 122.50   | 0.970      | –            | –   |
| DNA-Net             | 57.13    | 4.700      | –            | –   |
| RepISD              | 25.74    | 0.279      | –            | –   |
| **MSR-Net (Train)** | 0.777    | 0.065      | 7.20         | 138 |
| **MSR-Net (Deploy)**| **0.691**| **0.025**  | **4.20**     | **238** |

vs. ISNet on IRSTD-1k: **+2.39% IoU**, **39× fewer parameters**, **177× fewer MACs**.

> **Note on the precision–recall trade-off.** MSR-Net prioritizes false-alarm suppression, which can raise the miss rate for the dimmest targets on SISD-Sky (Pd = 66.52%); it is not Pareto-dominant on nIoU. This is a configurable operating point — lowering the binarization threshold to 0.3 raises SISD-Sky Pd to 74.20% at Fa = 18.5×10⁻⁶.


---

## Citation

If you find this work useful, please cite:

```bibtex
@article{nguyen2025msrnet,
  title   = {MSR-Net: Multi-Scale Structural Re-parameterization with Scale-Sensitive Loss for Real-Time Infrared Small Target Detection},
  author  = {Nguyen, Truong Danh and Tran, Duc Manh and Nguyen, Viet Hung},
  journal = {<Journal Name>},
  year    = {2025}
}
```

---

## Acknowledgements

This work was supported by the Electro-Optical Center, Viettel High Technology Industries Corporation, Viettel Group, Hanoi, Vietnam. We thank the providers of the **IRSTD-1k** and **SISD-Sky** datasets for enabling reproducible benchmarking. The structural re-parameterization formulation builds on [RepVGG](https://github.com/DingXiaoH/RepVGG), [DBB](https://github.com/DingXiaoH/DiverseBranchBlock), and [ACNet](https://github.com/DingXiaoH/ACNet).

---

## License

This project is released under the MIT License — see the [LICENSE](LICENSE) file for details.
