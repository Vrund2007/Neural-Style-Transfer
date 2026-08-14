# Arbitrary Neural Style Transfer (AdaIN) in PyTorch

A high-performance implementation of **Real-Time Arbitrary Neural Style Transfer** using **Adaptive Instance Normalization (AdaIN)** powered by **PyTorch** and deployed via a modern **Flask** web application interface.

---

## 📌 Executive Summary

Traditional Neural Style Transfer relies on iterative optimization via gradient descent for every content-style image pair, taking minutes to generate a single image. Later Feed-Forward Networks achieved real-time speeds but were restricted to a single pre-trained style per model.

This repository implements **Adaptive Instance Normalization (AdaIN)**, which enables **real-time, arbitrary style transfer** on unseen content and style images in a single forward pass. By dynamically adjusting the feature statistics (mean and variance) of the content image to match those of the style image in latent space, the model transfers artistic style without requiring per-style training or iterative optimization loops.

---

## 🏗️ System Architecture & Computer Science Foundations

```
                                +-------------------+
                                |   Content Image   |
                                +---------+---------+
                                          |
                                          v
                                 [ VGG-19 Encoder ]
                                          |
                                          v  Content Features
+---------------+              +----------+----------+
|  Style Image  | ---> [ VGG ] -> Style    -> [ AdaIN Layer ] ---> Target Latent Features
+---------------+              Features       +----------+----------+
                                          |
                                          v
                                    [ Decoder ]
                                          |
                                          v
                                  [ Stylized Output ]
```

### 1. VGG-19 Feature Extractor (Encoder)
- **Deep Convolutional Backbone**: Utilizes a truncated VGG-19 network pre-trained on ImageNet up to the `relu4_1` feature map.
- **Frozen Weights / Zero-Gradient Pass**: The encoder network parameters are kept completely frozen during training to act purely as a multi-scale feature extractor.
- **Hierarchical Feature Maps**: Extracts spatial structures and texture representations across progressive receptive fields (`relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`).

### 2. Adaptive Instance Normalization (AdaIN Module)
- **Latent Feature Alignment**: Normalizes the feature maps of the content image across spatial dimensions to wipe away its original style/texture, then shifts and scales the normalized features using the channel-wise feature statistics of the style image.
- **Dynamic Feature Scaling**: Operates entirely in non-parametric latent space without trainable weights inside the normalization layer itself.
- **Linear Feature Interpolation**: Provides continuous control over style intensity. A linear combination between raw content features and stylized features allows seamless slider adjustments from content-only to full style transfer.

### 3. Inverted Decoder Network
- **Symmetric Up-Sampling Pipeline**: Reconstructs RGB image space from 512-channel latent representations using nearest-neighbor upsampling layers ($512 \to 256 \to 128 \to 64 \to 3$).
- **Boundary Artifact Prevention**: Replaces standard zero-padding with `ReflectionPad2d` to mitigate edge border artifacts during spatial convolutions.
- **Unnormalized Feature Preservation**: Omits internal normalization layers (BatchNorm/InstanceNorm) inside the decoder to prevent distorting the target feature distribution established by AdaIN.

---

## ⚡ Training Pipeline & Performance Characteristics

### Composite Objective Optimization
The decoder network is trained end-to-end to reconstruct images whose latent features match the target AdaIN feature maps (Content Match) and whose multi-scale intermediate feature statistics match the style image (Style Match).

- **Content Feature Matching**: Evaluates structural alignment at deep feature representations (`relu4_1`).
- **Multi-Layer Style Matching**: Evaluates texture alignment across low, mid, and high-level feature activations (`relu1_1` through `relu4_1`).
- **Computational Efficiency**: Inference is executed in a single deterministic forward pass ($\mathcal{O}(1)$ time complexity with respect to style diversity), enabling sub-second style generation on CPU and GPU platforms.

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
│   │   └── utils.py                # AdaIN Implementation & Dataset Loaders
│   ├── static/
│   │   └── uploads/                # Upload Directory for Runtime Processing
│   ├── templates/
│   │   └── index.html              # Web Application User Interface
│   ├── style_data/                 # Sample Style References
│   └── examples/                   # Output Demonstration Artifacts
├── requirements.txt                # Python Dependencies
└── README.md                       # Documentation
```

---

## 🛠️ Installation & Usage Guide

### 1. Prerequisites
- **Python**: 3.10+ (Tested on Python 3.14)
- **PyTorch**: 2.0+ (Supports CPU & CUDA GPU Acceleration)

### 2. Environment Setup
Clone the repository and install required packages:

```bash
git clone https://github.com/Vrund2007/Neural-Style-Transfer.git
cd "Neural Style Transfer"
pip install -r requirements.txt
```

### 3. Run the Web Application
Start the Flask application server:

```bash
cd NST_Code
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

### 4. Run Model Training (Optional)
To train the decoder on custom content and style datasets:

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

- **Interactive User Interface**: Modern design with sticky navigation, particle effects, and dynamic status updates.
- **Preset Loader**: Quick-load demo content and style images for immediate testing.
- **Real-Time Style Strength Control**: Adjust the style interpolation strength smoothly between content and style output.
- **High-Resolution Export**: One-click download for rendered stylized output images.