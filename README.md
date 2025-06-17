# Benchmarking-Adam-Optimizers

## **Adam Variants Comparison: Adam, AdamW, and AdaMax:Mathematical Modeling and Simulation**

---

## 1. Introduction

Optimization algorithms play a central role in training deep neural networks. Among the most widely used is the **Adam optimizer**, which combines the benefits of momentum and RMSProp. However, despite its success, Adam has limitations such as convergence instability and suboptimal generalization.

To address these challenges, several **Adam variants** have been proposed:
- **AdamW**: Separates weight decay from gradient update
- **AdaMax**: Uses the infinity norm instead of the L2 norm for robustness

This project investigates the performance of these optimizers — including custom implementations — across both:
- A **classification task** on MNIST
- **Benchmarking** on Rosenbrock and Powell functions

---

## 2. Optimizer Formulations

The expected loss minimized during training is:

```math
J(\theta) = \mathbb{E}_{(x,y) \sim D} [ \ell(f(\theta; x), y) ]
```

Where:
- \( \theta \): model parameters  
- \( \ell \): loss function (cross-entropy)  
- \( f \): neural network function  
- \( (x, y) \sim D \): input-output data distribution

---

## 3. Dataset & Preprocessing

### 3.1 Dataset: MNIST
- **Samples**: 70,000 grayscale images of handwritten digits (28x28)
- **Split**: 80% train, 10% validation, 10% test
- **Labels**: Digits (0–9)
- **Preprocessing**:
  - Normalization: [0, 1]
  - One-hot encoding of labels

---

## 4. Optimizer Description

| Optimizer   | Description                                                  |
|-------------|--------------------------------------------------------------|
| Adam        | Baseline optimizer with adaptive moments                     |
| AdamW       | Decouples L2 weight decay from the update rule               |
| AdaMax      | Uses L∞ norm for stability with sparse gradients             |

Each optimizer was tested in:
- **TensorFlow's native version**
- **Our custom Python implementation**

---

## 5. Model Architecture

```
Input (28x28x1)
├── Conv2D (64) + ReLU
├── Conv2D (128) + ReLU
├── MaxPooling2D
├── Conv2D (128) + ReLU
├── Conv2D (64) + ReLU
├── MaxPooling2D
├── Flatten
├── Dense (28) + ReLU
└── Dense (10) + Softmax
```

---

## 6. Evaluation Metrics

- **Training Accuracy**
- **Validation Accuracy**
- **Loss (Training + Validation)**
- **Test Accuracy**
- **F1 Score**
- **Confusion Matrix**
- **Loss vs Time / Iterations**
- **3D Convergence Path**

---

## 7. Results Summary

### 7.1 Classification Performance (MNIST)

| Optimizer     | Accuracy (%) | F1 Score | Training Time |
|---------------|--------------|----------|----------------|
| TF Adam       | 99.00        | 0.9900   | 1min 7s        |
| TF AdamW      | 98.97        | 0.9897   | 1min 8s        |
| TF AdaMax     | 99.00        | 0.9900   | 1min 8s        |
| **OurAdam**   | **99.08**    | **0.9908** | 1min 6s      |
| **OurAdamW**  | 98.93        | 0.9893   | 1min 7s        |
| **OurAdaMax** | 98.97        | 0.9896   | 1min 7s        |

### 7.2 Optimization Benchmark Results

| Function     | Optimizer   | Final Loss | Iterations | CPU Time |
|--------------|-------------|------------|------------|----------|
| Rosenbrock   | AdaMax      | Lowest     | Fastest    | Best     |
| Powell       | AdaMax      | Lowest     | Fastest    | Best     |
| Rosenbrock   | Adam        | Higher     | Slower     | Slower   |

### 7.3 Visual Insights
- **AdaMax** shows smoother convergence and better handling of curvature
- **AdamW** improves generalization in early epochs
- **Our implementations** track TensorFlow results with minimal deviation

---

## 8. Simulation Tasks

We evaluated convergence on:
- **Rosenbrock Function**: \( f(x) = (1 - x_1)^2 + 100(x_2 - x_1^2)^2 \)
- **Powell Function**: Highly nonlinear, non-convex 4D benchmark

Metrics used:
- Loss over iterations
- Loss over CPU time
- 2D and 3D convergence path visualizations

---

## 9. Technical Stack

### Languages & Platform
- **Python 3.10** (Google Colab)
- **TensorFlow 2.16.1**
- **Jupyter Notebooks**

### Libraries
- `tensorflow`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## 10. Deliverables

| File | Description |
|------|-------------|
| `README.md` | Full technical documentation |
| `Math_Project_implementation.ipynb` | Jupyter notebook with code |
| `The Paper` | Final scientific report |
| `Presentation.pptx` | Slide deck |

---

## 11. Conclusion

This study demonstrated the trade-offs between different variants of Adam optimizers. While TensorFlow implementations offered speed, custom optimizers provided comparable — and occasionally superior — final metrics.

- **AdamW** improves regularization
- **AdaMax** excels in convergence behavior
- Custom code maintained functional parity with TensorFlow

---

## 12. Future Work

- Add more optimizers (e.g., AMSGrad, Nadam, RAdam)
- Test on complex datasets (CIFAR-10, IMDB)
- Use adaptive learning rate schedulers

---

## 13. References

- Kingma & Ba (2014). *Adam: A Method for Stochastic Optimization*. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)  
- Loshchilov & Hutter (2017). *Decoupled Weight Decay Regularization*. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)  
- Reddi et al. (2019). *On the Convergence of Adam and Beyond*. [arXiv:1904.09237](https://arxiv.org/abs/1904.09237)  
- TensorFlow Docs: [https://www.tensorflow.org](https://www.tensorflow.org)

---
