# Arbitrary Neural Style Transfer (AdaIN) in PyTorch

A high-performance implementation of **Real-Time Arbitrary Neural Style Transfer** using **Adaptive Instance Normalization (AdaIN)** powered by **PyTorch** and deployed via a modern **Flask** web application interface.

---

## 📌 Executive Summary

Traditional Neural Style Transfer (Gatys et al., 2015) relies on iterative optimization via gradient descent for every content-style image pair, taking minutes to generate a single image. Later Feed-Forward Networks (Johnson et al., 2016) achieved real-time speeds but were restricted to a single pre-trained style per model.

This repository implements **Adaptive Instance Normalization (Huang & Belongie, 2017)**, which enables **real-time, arbitrary style transfer** on unseen content and style images in a single forward pass. By normalizing content feature maps to match the channel-wise mean and variance of style feature maps, the model transfers artistic texture dynamically without requiring retrain loops or per-style parameters.

---

## 🧬 Architectural & Mathematical Foundation

```
                                +-------------------+
                                |   Content Image   |
                                +---------+---------+
                                          |
                                          v
                                 [ VGG-19 Encoder ]
                                          |
                                          v  f(c)
+---------------+              +----------+----------+
|  Style Image  | ---> [ VGG ] -> f(s) -> [ AdaIN Layer ] ---> t
+---------------+              +----------+----------+
                                          |
                                          v
                                    [ Decoder ]
                                          |
                                          v
                                  [ Stylized Output ]
```

### 1. Feature Extractor (VGG-19 Encoder)
- **Base Architecture**: Truncated VGG-19 network pre-trained on ImageNet up to the `relu4_1` activation layer.
- **Fixed Weights**: All encoder parameters are frozen ($\nabla_{\theta_E} = 0$).
- **Intermediate Multi-Scale Feature Maps**:
  - `enc_1`: `relu1_1` (64 channels)
  - `enc_2`: `relu2_1` (128 channels)
  - `enc_3`: `relu3_1` (256 channels)
  - `enc_4`: `relu4_1` (512 channels)

---

### 2. Adaptive Instance Normalization (AdaIN)
The core mechanism receives content feature maps $f(c) \in \mathbb{R}^{B \times C \times H_c \times W_c}$ and style feature maps $f(s) \in \mathbb{R}^{B \times C \times H_s \times W_s}$.

AdaIN normalizes $f(c)$ across spatial dimensions $(H, W)$ per channel, removing original content statistics, and scales/shifts the result using style feature statistics:

$$\text{AdaIN}(x, s) = \sigma(s) \left( \frac{x - \mu(x)}{\sigma(x)} \right) + \mu(s)$$

Where the spatial channel-wise mean $\mu(x)$ and standard deviation $\sigma(x)$ are computed as:

$$\mu_c(x) = \frac{1}{HW} \sum_{h=1}^H \sum_{w=1}^W x_{c,h,w}$$

$$\sigma_c(x) = \sqrt{\frac{1}{HW} \sum_{h=1}^H \sum_{w=1}^W (x_{c,h,w} - \mu_c(x))^2 + \epsilon}$$

#### Continuous Style Strength Control ($\alpha$)
The trade-off between content structure and style intensity is controlled continuously using linear interpolation parameter $\alpha \in [0.0, 1.0]$:

$$t = \alpha \cdot \text{AdaIN}(f(c), f(s)) + (1 - \alpha) \cdot f(c)$$

- $\alpha = 0.0$: Preserves original content features.
- $\alpha = 1.0$: Fully transfers target style statistics.

---

### 3. Symmetric Inverted Decoder Network
The decoder $g$ takes normalized target feature representation $t \in \mathbb{R}^{B \times 512 \times H \times W}$ and reconstructs it back into RGB image space $g(t) \in \mathbb{R}^{B \times 3 \times H_{out} \times W_{out}}$.

- **Reflection Padding**: Replaces standard zero-padding (`ReflectionPad2d`) to eliminate boundary edge artifacts.
- **Upsampling**: Uses Nearest Neighbor upsampling (`scale_factor=2`) to increase resolution progressively ($512 \to 256 \to 128 \to 64 \to 3$).
- **No Normalization Layers**: InstanceNorm / BatchNorm are omitted in the decoder to avoid distorting normalized feature statistics.

---

## 🎯 Loss Functions & Training Strategy

The decoder $g$ is trained using a composite loss function penalizing content and style discrepancies:

$$\mathcal{L}_{total} = \mathcal{L}_c + \gamma \mathcal{L}_s$$

### 1. Content Loss ($\mathcal{L}_c$)
Ensures reconstructed image features $f(g(t))$ match target AdaIN feature map $t$ in deep latent space (`relu4_1`):

$$\mathcal{L}_c = \| f(g(t)) - t \|_2^2$$

### 2. Multi-Layer Style Loss ($\mathcal{L}_s$)
Measures feature statistics matching across multiple intermediate encoder layers $i \in \{\text{relu1\_1}, \text{relu2\_1}, \text{relu3\_1}, \text{relu4\_1}\}$:

$$\mathcal{L}_s = \sum_{i=1}^{L} \left( \|\mu(\phi_i(g(t))) - \mu(\phi_i(s))\|_2^2 + \|\sigma(\phi_i(g(t))) - \sigma(\phi_i(s))\|_2^2 \right)$$

### Training Hyperparameters
- **Optimizer**: Adam ($\text{lr} = 10^{-4}$, $\text{lr\_decay} = 5 \times 10^{-5}$)
- **Loss Weights**: Content Weight = $1.0$, Style Weight ($\gamma$) = $5.0$
- **Input Resolution**: Random crop $256 \times 256$ pixels during training
- **Batch Size**: 4 content-style pairs per iteration

---

## 📁 Repository Directory Structure

```
.
├── NST_Code/
│   ├── app.py                      # Flask Web Application & Inference Server
│   ├── train.py                    # Model Training Pipeline
│   ├── vgg_normalised.pth          # Pre-trained VGG-19 Encoder Weights
│   ├── experiment/
│   │   └── final_exp/
│   │       └── decoder_final.pth   # Trained PyTorch Decoder Weights
│   ├── utils/
│   │   ├── models.py               # VGGEncoder & Decoder Module Definitions
│   │   └── utils.py                # AdaIN Math Implementation & Dataset Loaders
│   ├── static/
│   │   └── uploads/                # Upload Directory for Runtime Processing
│   ├── templates/
│   │   └── index.html              # Modern Web Application User Interface
│   ├── style_data/                 # Sample Style References (la_muse.jpg, etc.)
│   └── examples/                   # Pre-compiled Demo Output Artifacts
├── requirements.txt                # Python Dependencies
└── README.md                       # Documentation
```

---

## 🛠️ Installation & Usage Guide

### 1. Prerequisites
- **Python**: 3.10+ (Tested up to Python 3.14)
- **PyTorch**: 2.0+ (CPU or CUDA GPU acceleration)

### 2. Environment Setup
Clone the repository and install required packages:

```bash
git clone https://github.com/shradha-khapra/ai-nst-project.git
cd "Neural Style Transfer"
pip install -r requirements.txt
```

### 3. Run the Web Application
Start the Flask development server:

```bash
python NST_Code/app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

### 4. Run Model Training (Optional)
To train a custom decoder on your own content & style image datasets:

```bash
python NST_Code/train.py \
  --content_dir ./NST_Code/content_data \
  --style_dir ./NST_Code/style_data \
  --epochs 10 \
  --batch_size 4 \
  --style_weight 5.0
```

---

## 🌐 Web Application Features

- **Sticky Top Header**: Clean sticky navigation bar (`fixed-top` with dark glassmorphism).
- **Interactive Mouse Particles**: Dynamic background neural network canvas with mouse force repulsion and multi-hue node physics.
- **Demo Presets**: One-click **"Use Demo Images"** loader for immediate testing (`la_muse.jpg`, `sketch.png`, `picasso_seated_nude_hr.jpg`).
- **Style Strength Range Control**: Real-time slider adjusting alpha strength $\alpha \in [0.0, 1.0]$.
- **High-Resolution Result Output**: Single-click image download for generated artwork.

---