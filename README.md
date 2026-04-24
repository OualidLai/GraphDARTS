# 🚀 GraphDARTS  
**Contrastive Graph Metric Learning for Unsupervised Differentiable Architecture Search in Structural Health Monitoring**

---

## 📌 Overview
Structural Health Monitoring (SHM) often operates in scenarios where labeled data is unavailable or extremely limited, making unsupervised learning essential for reliable damage detection. While Differentiable Architecture Search (DARTS) has shown strong performance in supervised tasks, its application to unsupervised SHM remains largely unexplored.

**GraphDARTS** introduces a unified framework that combines:
- Unsupervised architecture search  
- Contrastive metric learning  
- Dual graph representations  

The method is specifically designed for **acoustic emission (AE) signal analysis**, enabling robust and scalable damage detection without labeled data.

---

## ❗ Motivation
Existing unsupervised SHM approaches face several key limitations:

- **Architectural bias** from fixed network topologies  
- **Incomplete relational modeling**, focusing only on similarity while ignoring dissimilarity  
- **Scalability issues** in graph-based similarity computation  

GraphDARTS addresses these challenges by jointly learning:
- Feature embeddings  
- Neural architectures  
- Similarity and dissimilarity relationships  

---

## 🧠 Key Contributions

### 🔄 Single-Level Unsupervised Optimization
We reformulate the traditional bilevel DARTS optimization into a **single-level objective**, allowing simultaneous learning of:
- Neural architecture (MLPs)  
- Embedding representations  

This is achieved via a **contrastive graph metric learning objective**.

---

### 🔗 Dual Graph Metric Learning
GraphDARTS constructs two complementary graphs:

- **Similarity Graph**
  - Built using low-rank spectral decomposition of Gram matrices  
  - Captures intrinsic cluster structures  

- **Dissimilarity Graph**
  - Encodes uniform separation constraints  
  - Enhances inter-cluster discrimination  

---

### 📊 Dual-Criterion Cluster Validation
To ensure robust evaluation in unsupervised settings:

- **Silhouette Coefficient** → guides cluster selection  
- **Recall** → used as post-hoc evaluation (ORION-AE protocol)  

This dual strategy resolves ambiguity found in single-metric approaches.

---

### ⚙️ LogDet Rank Optimization
We introduce a **LogDet relaxation** to enforce low-rank structure:

- Encourages compact clusters  
- Improves separation via contrastive Laplacians  

---

## 🏗️ Architecture Search Space
GraphDARTS explores a flexible MLP-based search space:

- Hidden units: **8 → 2048 neurons**  
- Activation functions: ReLU, GELU, Tanh, etc.  
- Regularization: Dropout variations  

Additionally:
- Embeddings are clustered using **Time Series K-Means**  
- Captures temporal dependencies in AE signals  

---

## 📈 Results
GraphDARTS achieves strong performance across **five ORION-AE campaigns**:

- ✅ **Recall:** 1.00 (perfect damage detection)  
- 🎯 **Precision:** 0.188 – 0.833  
- 🔗 **Adjusted Rand Index (ARI):** 0.823 – 0.969  

➡️ Successfully identifies all damage classes and outperforms baseline methods.

---

If you find this work useful, please cite:

@article{LAIADI2026114292,
title = {GraphDARTS: Contrastive graph metric learning for unsupervised differentiable architecture search in Structural Health Monitoring},
journal = {Mechanical Systems and Signal Processing},
volume = {253},
pages = {114292},
year = {2026},
doi = {https://doi.org/10.1016/j.ymssp.2026.114292},
author = {Oualid Laiadi and Ikram Remadna and Mohamed El-Amine Laiadi and Oussama Hadoune and Redouane Drai and Noureddine Zerhouni}
}
