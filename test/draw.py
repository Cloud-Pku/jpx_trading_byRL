import pickle
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
with open("./test/data_set.pkl", 'rb') as f:
    data_set = np.array(pickle.load(f)).squeeze()
    
with open("./test/label.pkl", 'rb') as f:
    label = np.array(pickle.load(f))

data_set = data_set[:, 17, :]

# data_set = data_set[760:800]
label = label[:, 17]
print(data_set.shape)
print(label.shape)

after_sort = []
for item in label:
    after_sort.append(np.argsort(item))
after_sort = np.array(after_sort).reshape(-1)

X = data_set.reshape(-1, 10)
Y = label.reshape(-1)
print(X.shape, Y.shape)
'''t-SNE'''
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)
norm = plt.Normalize(Y.min(), Y.max())
norm_y = norm(Y)
'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(8, 8))

plt.scatter(X_norm[:, 0], X_norm[:, 1], alpha= 0.5, c=norm_y, cmap='viridis')

plt.xticks([])
plt.yticks([])
plt.savefig("./test/op.png") 
plt.close()