# Sparse Neural Network with Learnable Gating

This project focuses on designing and training a prunable deep neural network that can automatically identify and remove redundant components during training. The model integrates learnable sigmoid-based gates into fully connected layers, enabling dynamic control over neuron activations and facilitating structured sparsity.

A modified SE-ResNet-56 architecture is used as the backbone, combined with prunable linear layers in the classification head. Each neuron is associated with a trainable gate parameter, and an L1 regularization penalty is applied to these gates to encourage sparsity. During training, the model jointly optimizes classification performance and sparsity, effectively learning which neurons are essential and which can be suppressed.

To improve generalization and robustness, techniques such as CutMix data augmentation, mixed-precision training (AMP), and dual-optimizer strategies (SGD for weights and Adam for gate parameters) are employed. A controlled λ (lambda) scheduling mechanism balances the trade-off between accuracy and sparsity.

The project demonstrates how neural networks can maintain high predictive performance while introducing mechanisms for model compression and efficiency, laying the foundation for deployment in resource-constrained environments.


###  Why L1 Penalty on Sigmoid Gates Encourages Sparsity

In this model, each neuron is controlled by a learnable gate:

$$g = \sigma(s)$$

where $s$ is the trainable gate score and $\sigma$ is the sigmoid function. 
The output of a neuron is scaled by this gate:

$$y = g \cdot f(x)$$

---

###  Role of L1 Regularization

We apply an L1 penalty on the gate values:

$$\mathcal{L}_{\text{sparsity}} = \sum |g|$$

This encourages sparsity because:

* The **L1 penalty** pushes values toward zero.
* Since $g \in (0,1)$, minimizing $g$ pushes it toward 0.
* When $g \approx 0$, the neuron is effectively turned off.
* This leads to **automatic pruning** of unnecessary neurons.

---

###  Important Insight

* Sigmoid gates never reach exact 0, but can get arbitrarily close.
* This results in **soft sparsity**.
* Thresholding (e.g., $g < 0.1$) is used to measure effective pruning.

---

### 🔹 Experimental Results

| Lambda ($\lambda$) | Test Accuracy (%) | 
| :--- | :--- | 
| **0.1** | 93.91 | 
| **1.0** | 93.64 |
| **5.0** | 93.38 | 
