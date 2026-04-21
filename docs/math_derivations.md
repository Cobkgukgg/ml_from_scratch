# Mathematical Derivations

## 1. Linear Regression (OLS)

**Model:** $\hat{y} = X\theta$

**Loss:** $L(\theta) = \frac{1}{2n}\|y - X\theta\|^2$

**Normal Equation:** Minimize by setting $\nabla L = 0$:
$$\nabla_\theta L = \frac{1}{n} X^T(X\theta - y) = 0 \implies \theta^* = (X^TX)^{-1}X^Ty$$

**Gradient Descent Update:**
$$\theta \leftarrow \theta - \alpha \cdot \frac{1}{n}X^T(X\theta - y)$$

**Ridge (L2):** Add $\frac{\lambda}{2}\|\theta\|^2$ → closed form: $\theta^* = (X^TX + \lambda I)^{-1}X^Ty$

**Lasso (L1):** Coordinate descent with soft-thresholding:
$$\theta_j \leftarrow S\!\left(\frac{X_j^T r_j}{n}, \lambda\right), \quad S(z, \gamma) = \text{sign}(z)\max(|z| - \gamma, 0)$$

---

## 2. Logistic Regression

**Model:** $P(y=1|x) = \sigma(x^T\theta)$, where $\sigma(z) = \frac{1}{1+e^{-z}}$

**Log-likelihood:**
$$\ell(\theta) = \frac{1}{n}\sum_{i=1}^n \left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]$$

**Gradient:**
$$\nabla_\theta \ell = \frac{1}{n}X^T(\hat{p} - y)$$

**Multinomial Softmax:** $P(y=k|x) = \frac{e^{x^T w_k}}{\sum_j e^{x^T w_j}}$

**Gradient w.r.t. $W$:** $\nabla_W = \frac{1}{n}(\hat{P} - Y_{enc})^T X$

---

## 3. Decision Tree (CART)

**Gini Impurity:**
$$G(S) = 1 - \sum_{k} p_k^2$$

**Entropy:**
$$H(S) = -\sum_{k} p_k \log_2 p_k$$

**Information Gain:**
$$\text{IG}(S, A) = H(S) - \frac{|S_L|}{|S|}H(S_L) - \frac{|S_R|}{|S|}H(S_R)$$

**Best Split:** exhaustively search all features and thresholds, pick $(j^*, t^*)$ that maximizes IG.

---

## 4. Gradient Boosting

**Additive model:** $F_m(x) = F_{m-1}(x) + \nu h_m(x)$

At step $m$, fit $h_m$ to **pseudo-residuals** (negative gradient of loss):
$$r_i^{(m)} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

**MSE loss:** $r_i = y_i - F_{m-1}(x_i)$

**Log-loss:** $r_i = y_i - \sigma(F_{m-1}(x_i))$

---

## 5. XGBoost (Second-Order Taylor Expansion)

Expand loss around $F_{m-1}$ to second order:
$$L \approx \sum_i \left[g_i f_m(x_i) + \frac{1}{2}h_i f_m(x_i)^2\right] + \Omega(f_m)$$

Where $g_i = \partial_{F}L(y_i, F_{m-1})$, $h_i = \partial^2_{F}L(y_i, F_{m-1})$.

**Optimal leaf weight:** $w_j^* = -\frac{G_j}{H_j + \lambda}$

**Split gain:**
$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G^2}{H+\lambda}\right] - \gamma$$

---

## 6. Support Vector Machine (SVM)

**Primal:**
$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_i\xi_i \quad \text{s.t.} \quad y_i(w^Tx_i+b) \geq 1 - \xi_i, \xi_i \geq 0$$

**Dual:**
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_iy_j K(x_i,x_j)$$

**KKT conditions:** $0 \leq \alpha_i \leq C$, $\sum_i \alpha_i y_i = 0$

**SMO:** Iteratively update two $\alpha_i, \alpha_j$ analytically while holding others fixed.

**RBF kernel:** $K(x, x') = \exp(-\gamma\|x-x'\|^2)$

---

## 7. Principal Component Analysis

**SVD decomposition:** $X - \bar{X} = U \Sigma V^T$

**Principal components:** columns of $V$ (right singular vectors)

**Projection:** $Z = (X - \bar{X}) V_k$ where $V_k$ is first $k$ columns

**Explained variance ratio:** $\text{EVR}_j = \sigma_j^2 / \sum_i \sigma_i^2$

**Whitening:** $Z_{\text{white}} = Z \cdot \text{diag}(\sigma)^{-1}$ → unit variance components

---

## 8. Gaussian Mixture Model (EM)

**Latent variable:** $z_i \in \{1,\ldots,K\}$ indicates component membership.

**E-step (responsibilities):**
$$r_{ik} = \frac{\pi_k \mathcal{N}(x_i|\mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i|\mu_j, \Sigma_j)}$$

**M-step:**
$$N_k = \sum_i r_{ik}, \quad \pi_k = \frac{N_k}{n}, \quad \mu_k = \frac{\sum_i r_{ik}x_i}{N_k}, \quad \Sigma_k = \frac{\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^T}{N_k}$$

---

## 9. t-SNE

**High-dim similarity (Gaussian):**
$$p_{j|i} = \frac{\exp(-\|x_i-x_j\|^2 / 2\sigma_i^2)}{\sum_{k\neq i}\exp(-\|x_i-x_k\|^2/2\sigma_i^2)}$$

**Bandwidth $\sigma_i$:** binary search to match target perplexity $= 2^{H(P_i)}$.

**Low-dim similarity (Student-t, df=1):**
$$q_{ij} = \frac{(1+\|y_i-y_j\|^2)^{-1}}{\sum_{k\neq l}(1+\|y_k-y_l\|^2)^{-1}}$$

**KL divergence loss:** $L = \text{KL}(P\|Q)$

**Gradient:**
$$\frac{\partial L}{\partial y_i} = 4\sum_j (p_{ij} - q_{ij})(y_i - y_j)(1+\|y_i-y_j\|^2)^{-1}$$

---

## 10. Adam Optimizer

**Updates at step $t$:**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = \theta_{t-1} - \frac{\alpha \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Defaults: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$

