from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd
# Load the data set
iris = load_iris()
iris_df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
# Scale the data to have a mean of 0 and a standard deviation of 1
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_df)

from sklearn.decomposition import KernelPCA
# Initialise KernelPCA with different kernels
kpca_linear = KernelPCA(n_components=2, kernel="linear")
kpca_rbf = KernelPCA(n_components=2, kernel="rbf")
kpca_poly = KernelPCA(n_components=2, kernel="poly")
kpca_cos = KernelPCA(n_components=2, kernel="cosine")

# Fit and transform the scaled data
X_linear = kpca_linear.fit_transform(iris_scaled)
X_rbf = kpca_rbf.fit_transform(iris_scaled)
X_poly = kpca_poly.fit_transform(iris_scaled)
X_cos = kpca_cos.fit_transform(iris_scaled)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(15, 15))

axs[0, 0].scatter(X_linear[:, 0], X_linear[:, 1], c=iris["target"])
axs[0, 0].set_title("Linear Kernel")

axs[0, 1].scatter(X_rbf[:, 0], X_rbf[:, 1], c=iris["target"])
axs[0, 1].set_title("RBF Kernel")

axs[1, 0].scatter(X_poly[:, 0], X_poly[:, 1], c=iris["target"])
axs[1, 0].set_title("Polynomial Kernel")

axs[1, 1].scatter(X_cos[:, 0], X_cos[:, 1], c=iris["target"])
axs[1, 1].set_title("Cosine Kernel")

plt.show()

X_rbf_unscaled = KernelPCA(n_components=2, kernel="rbf").fit_transform(iris_df)

plt.scatter(X_rbf_unscaled[:, 0], X_rbf_unscaled[:, 1], c=iris["target"])
plt.title("RBF Kernel without Scaling")
plt.show()
