# 🎭 DeepfakeCatcher — Reproducibility Study

> A dual-backbone, feature-level fusion deep learning model for deepfake detection.  
> Reproducibility study of [Agarwal & Ratha, CVPR 2024](https://openaccess.thecvf.com/CVPR2024).

---

## 📌 Overview

This repository contains the implementation and full research report for a reproducibility study of the **DeepfakeCatcher** framework. Instead of the original score-level fusion of three backbones (VGG-16, MobileNet, Xception), we implement a **feature-level fusion** of two pretrained CNNs:

- **EfficientNet-B4** — captures global texture patterns (1,792-dim feature vector)
- **Xception** — detects cross-channel spatial artifacts (2,048-dim feature vector)

Their outputs are concatenated into a **3,840-dimensional** fused representation, passed through a regularized classification head to produce a binary **Real / Fake** prediction.

---

## 📊 Results

| Metric | Original Paper | Our Implementation | Δ |
|---|---|---|---|
| Accuracy (FF++) | 92.40% | **91.00%** | −1.40% |
| AUC-ROC (FF++ HQ) | 98.01% | **97.49%** | −0.52% |

> A gap of < 2% is considered a successful replication in the deep learning literature.

---

## 🏗️ Architecture

```
Input Image (224×224)
        │
   ┌────┴────┐
   │         │
EfficientNet-B4    Xception
(frozen → fine-tuned → task layers)
   │         │
1,792-dim   2,048-dim
   │         │
   └────┬────┘
        │  Concatenation (3,840-dim)
        │
   FC(1024) → BN → ReLU → Dropout(0.5)
        │
   FC(256) → ReLU → Dropout(0.25)
        │
   Output(2) → Softmax
        │
   Real / Fake
```

---

## 📁 Repository Structure

```
├── notebook/
│   └── deepfake_detection.ipynb   # Main training notebook (Google Colab)
├── report/
│   ├── deepfake_detection_ieee.tex  # IEEE-format LaTeX paper
│   └── figures/                     # Plots and architecture diagrams
├── README.md
└── requirements.txt
```

---

## 🗂️ Dataset

We use the **[FaceForensics++ (FF++)](https://github.com/ondyari/FaceForensics)** benchmark via a [Kaggle subset](https://www.kaggle.com/datasets/hungle3401/faceforensics) (~2.73 GB).

| Split | Real videos | Fake videos |
|---|---|---|
| Train (70%) | ~140 | ~140 |
| Val (15%) | ~30 | ~30 |
| Test (15%) | ~30 | ~30 |

Fake videos include: **DeepFakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter** in both HQ and LQ formats.

> The full FF++ dataset (~200 GB) was not used due to hardware constraints. Results on the full dataset may differ slightly.

---

## ⚙️ Setup

### Requirements

```bash
pip install -r requirements.txt
```

```
torch>=2.0
torchvision>=0.15
efficientnet-pytorch
timm
opencv-python
scikit-learn
matplotlib
numpy
```

### Running the notebook

The notebook is designed for **Google Colab with a T4 GPU**.

1. Upload the notebook to [Google Colab](https://colab.research.google.com)
2. Mount your Google Drive or upload the dataset directly
3. Set the dataset path in the config cell
4. Run all cells — training completes in ~25 minutes (13 epochs with early stopping)

---

## 🧪 Training Configuration

| Component | Value |
|---|---|
| Optimizer | Adam (weight decay 1e-4) |
| Scheduler | ReduceLROnPlateau |
| Early stopping | Patience = 5 |
| Batch size | 32 |
| Initial LR | 1e-4 |
| Epochs trained | 13 |
| Augmentation | RandomHorizontalFlip, ColorJitter |

---

## 📈 Training Dynamics

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | AUC |
|---|---|---|---|---|---|
| 1 | 0.3823 | 82.21% | 0.2355 | 89.00% | 96.50% |
| 4 | 0.0575 | 97.82% | 0.2594 | 90.67% | 97.33% |
| 8 *(best)* | 0.0150 | 99.46% | 0.3811 | **91.00%** | 96.32% |
| 13 | 0.0165 | 99.61% | 0.3792 | 90.67% | 96.61% |

---

## 🔬 Key Design Choices

**Feature-level vs score-level fusion** — concatenating raw feature vectors preserves richer cross-backbone information compared to averaging softmax probabilities.

**EfficientNet-B4 + Xception** — complementary extractors: EfficientNet captures compound-scaled global textures; Xception's depthwise separable convolutions excel at spotting GAN checkerboard artifacts.

**Transfer learning** — ImageNet pretrained weights; only the classifier head is trained from scratch, reducing training time significantly.

**Regularization** — Dropout (0.5, 0.25) + Batch Normalization + Early Stopping prevent memorization of the small training set.

---

## ⚠️ Limitations

- Evaluated on a **single dataset** (FF++ Kaggle subset); cross-dataset generalization is unknown
- **No face detection** applied during preprocessing (hardware constraint) — background noise may affect accuracy
- **No temporal reasoning** — frames are classified independently; subtle motion-based fakes may be missed
- Two backbones instead of three (original paper uses VGG-16, MobileNet, Xception)

---

## 🔮 Future Work

- [ ] Evaluate on Celeb-DF v2 and DFDC for cross-dataset generalization
- [ ] Add MTCNN face detection preprocessing
- [ ] Progressive layer unfreezing to reduce overfitting
- [ ] Replace concatenation with a learned cross-attention fusion
- [ ] Extend to temporal reasoning with 3D CNN or LSTM

---

## 📄 Citation

If you use this work, please cite the original paper:

```bibtex
@inproceedings{agarwal2024deepfakecatcher,
  title     = {Deepfake Catcher: Can a Simple Fusion be Effective and Outperform Complex DNNs?},
  author    = {Agarwal, Shruti and Ratha, Nalini},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
```

---

## 👥 Authors

| Name | Email | Institution |
|---|---|---|
| Abdullah Khan | i232546@isb.nu.edu.pk | FAST NUCES Islamabad |
| Ahmed Ghaffar | i232599@isb.nu.edu.pk | FAST NUCES Islamabad |
| Muhammad Usman | i232512@isb.nu.edu.pk | FAST NUCES Islamabad |

---

## 📜 License

This project is for academic and research purposes only. The FaceForensics++ dataset is subject to its own [terms of use](https://github.com/ondyari/FaceForensics/blob/master/LICENSE).
