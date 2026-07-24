# Financial Fraud Detection using Graph Neural Networks (GNNs)

## Project Overview
This project applies Modern AI techniques to detect illicit financial behaviors within a network of localized Bitcoin transactions. By leveraging Graph Neural Networks, the model captures the relational dynamics and flow of currency across transaction networks.

## Dataset
* **Elliptic Bitcoin Dataset:** 203,769 transaction nodes and 234,355 edge relationships.
* **Class Imbalance:** Features a ~10:1 ratio of legitimate (42,019) to fraudulent (4,545) transactions, with 157,205 unlabeled nodes serving as network bridges.

## Technologies Used
* **Frameworks:** PyTorch, PyTorch Geometric (PyG)
* **Libraries:** scikit-learn, NumPy
* **Core Concepts:** Graph Convolutional Networks (GCN), Sparse Matrix Operations, Class Imbalance Handling

## Experiments & Repository Structure

This repository contains two iterations of the Graph Neural Network pipeline:

1. **`Fraud_Detection_Baseline.ipynb`**
   * Standard GCN trained using unweighted Cross-Entropy loss.
   * **Overall Accuracy:** 94.13%
   * **Fraud F1-Score:** 0.4318

2. **`Fraud_detection_weighted_loss.ipynb`**
   * GCN trained using a custom weighted Cross-Entropy loss (~9.2x penalty weight on fraud class) to directly penalize false negatives in the minority class.
   * **Overall Accuracy:** 88.11%
   * **Fraud F1-Score:** 0.4001
     
## Key Takeaway
While overall accuracy remains high (>88%), evaluating model performance via the minority-class F1-score provides a realistic assessment of fraud detection capabilities on highly imbalanced financial datasets.
