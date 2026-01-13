# 🚦 Traffic Sign Spatial Recognition & XAI
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
> **Comparative analysis of spatial invariance in Neural Networks and Explainable AI (XAI) for Autonomous Driving perception systems.**

This project quantifies the impact of **Inductive Bias** in Computer Vision architectures. It explores the catastrophic failure of **Dense Networks (MLP)** regarding **Spatial Invariance** compared to Convolutional Networks (CNN). The pipeline includes a custom "Shifted Dataset" generation to stress-test the models and utilizes **Grad-CAM** to audit the decision-making process, ensuring the model focuses on semantic features (pictograms) rather than contextual noise.

---

## 💡 Project Overview

In **Autonomous Driving**, a perception system must be robust against camera vibrations or misalignment. A stop sign is a stop sign, regardless of its position in the frame. This project empirically proves why the lack of spatial invariance in MLPs makes them unsuitable for raw pixel processing.

**Key Technical Capabilities:**

* **🛡️ Robustness Auditing:** Implementation of a custom data pipeline to create `ds2` (Shifted Dataset), rigorously testing translational invariance.
* **📊 Architecture Benchmarking:** A/B testing between MLP (Flattened input) and CNN (Spatial input) under stress conditions.
* **🧠 Explainable AI (XAI):** Integration of `pytorch-grad-cam` to generate Saliency Maps, verifying that the model learns the object's semantics and not background bias.
* **🔄 Data Augmentation Pipeline:** Using affine transformations to improve generalization on the validation set.

---

📊 The Dataset: GTSRB
The model was trained on the German Traffic Sign Recognition Benchmark (GTSRB), a de-facto standard for multi-class image classification in Computer Vision. It presents a high challenge due to real-world variations such as varying light conditions, motion blur, and physical damage to the signs.

Classes: 43 distinct categories (Speed limits, Stop, Yield, Warnings, etc.).

Volume: >50,000 images (Train/Test split).

Resolution: Varying sizes (resized to 32x32 for this experiment).

<img width="1200" height="1200" alt="image" src="https://github.com/user-attachments/assets/7727de18-1808-421c-88e1-36ebc20af1a6" />


---

## 🏗️ System Architecture

The pipeline demonstrates the complete lifecycle from raw tensor processing to model auditing.

```mermaid
graph LR
    A[GTSRB Dataset] --> B{Data Pipeline}
    B -->|Original Tensor| C[Standard Training]
    B -->|Spatial Shift| D["Stress Test (DS2)"]
    
    subgraph Models [Model Architectures]
        C --> E[MLP Baseline]
        C --> F[CNN / FCNN]
        D --> E
        D --> F
    end
    
    F --> G[Inference]
    G --> H[Performance Metrics]
    G --> I[Grad-CAM XAI]
    
    H --> J((Final Report))
    I --> J
```

---

## 🧪 Experiments & Results

The core experiment involved training models on centered data and testing them on spatially shifted data to measure **Generalization Capability**.

### 1. The "Shift" Hypothesis

We tested the hypothesis that **Dense Networks** overfit to specific pixel coordinates. A synthetic dataset (`ds2`) was created where traffic signs were randomly translated within the 32x32 canvas.

### 2. Quantitative Results

| Model Architecture | Inductive Bias | Test Accuracy (Shifted Data) | Technical Conclusion |
| --- | --- | --- | --- |
| **MLP (Dense)** | None | **~24.2%** ❌ | **Catastrophic Failure.** Flattening the input destroys spatial topology; the model treats shifted pixels as entirely new features. |
| **FCNN / CNN** | Spatial | **~83.5%** ✅ | **Robust.** Convolutional kernels share weights across the input, effectively detecting features regardless of position. |

---

## 🛠️ Tech Stack

* **Deep Learning Framework:** PyTorch (`torch`, `torchvision`, `nn.Module`).
* **Data Manipulation:** NumPy (tensor operations), Pandas (logging).
* **Visualization:** Matplotlib, Seaborn, IPyWidgets (interactive analysis).
* **Interpretability:** `pytorch-grad-cam` (Gradient-weighted Class Activation Mapping).

---

## 🚀 Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/Hugo31810/traffic-sign-spatial-recognition.git
cd traffic-sign-spatial-recognition

```

### 2. Install Dependencies

```bash
pip install torch torchvision matplotlib pandas scikit-learn pytorch-grad-cam opencv-python-headless

```

### 3. Run the Analysis

The project is structured in sequential notebooks for reproducibility:

1. **`01_Dataset_Prep.ipynb`**: Data downloading, preprocessing, and "Shifted Dataset" creation.
2. **`02_MLP_Baseline.ipynb`**: Training the Dense Network and analyzing the high error rate on `ds2`.
3. **`03_CNN_Architecture.ipynb`**: Implementing the robust CNN/FCNN architecture and comparing metrics.
4. **`04_XAI_Visualization.ipynb`**: Running Grad-CAM and filter visualization hooks.


---

## 👨‍💻 Author

**Hugo Salvador Aizpún**

*Degree in Artificial Intelligence*

*Focus: Machine Learnign II*

[GitHub Profile](https://www.google.com/search?q=https://github.com/Hugo31810)
