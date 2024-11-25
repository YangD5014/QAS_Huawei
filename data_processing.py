from sklearn.decomposition import PCA
import numpy as np
from torchvision import datasets, transforms# 导入量子门RY
from sklearn.model_selection import train_test_split   
import torch
from mindquantum.algorithm.library import amplitude_encoder
from mindquantum.algorithm.nisq import IQPEncoding





def sample_data(X, y, label, sample_ratio=0.2):
    label_mask = (y == label)
    X_label = X[label_mask]
    y_label = y[label_mask]
    sample_size = int(len(y_label) * sample_ratio)
    sample_indices = np.random.choice(len(y_label), sample_size, replace=False)
    return X_label[sample_indices], y_label[sample_indices]

def filter_3_and_6(data):
    images, labels = data
    mask = (labels == 3) | (labels == 6)
    return images[mask], labels[mask]


def amplitude_param(pixels):
    param_rd = []
    _, parameterResolver = amplitude_encoder(pixels, 6)   
    for _, param in parameterResolver.items():
        param_rd.append(param)
    param_rd = np.array(param_rd)
    return param_rd

def  PCA_data_preprocessing(mnist_dataset:datasets.MNIST,PCA_dim:int=10):
    '''
    将 28*28 的 MNIST 手写数字图像 基于PCA进行压缩
    
    '''
    transform = transforms.Compose([
    transforms.ToTensor()])
    np.random.seed(10)
        
    #mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    filtered_data = filter_3_and_6((mnist_dataset.data, mnist_dataset.targets))
    X_data, y = filtered_data  # X 图像数据 y 标签
    
    X_data_3, y_data_3 = sample_data(X_data, y, label=3, sample_ratio=0.1)
    X_data_6, y_data_6 = sample_data(X_data, y, label=6, sample_ratio=0.1)
    #合并抽样后的数据
    X_sampled = torch.cat((X_data_3, X_data_6), dim=0)
    y_sampled = torch.cat((y_data_3, y_data_6), dim=0)
    
    n_samples = X_sampled.shape[0]
    X_flattened = X_sampled.view(n_samples, -1)  # 将图像展平为一维向量
    X_flattened = X_flattened/255
    pca = PCA(n_components=PCA_dim)
    X_pca = pca.fit_transform(X_flattened)
    
    # 将 PCA 处理后的值缩放到 [0, π] 之间
    X_pca_min = np.min(X_pca)
    X_pca_max = np.max(X_pca)
    X_pca_scaled = np.pi * (X_pca - X_pca_min) / (X_pca_max - X_pca_min)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X_pca_scaled, y_sampled, test_size=0.2, random_state=0, shuffle=True) # 将数据集划分为训练集和测试集
    y_train[y_train==3]=1
    y_train[y_train==6]=0
    y_test[y_test==3]=1
    y_test[y_test==6]=0
    y_train = y_train.numpy()
    y_test = y_test.numpy()
    
    
    return X_train, X_test, y_train, y_test

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
X_train, X_test, y_train, y_test = PCA_data_preprocessing(mnist_dataset,8)

def  PCA_data_preprocessing_micro(mnist_dataset:datasets.MNIST=mnist_dataset,
                                  PCA_dim:int=8,ratio:float=1.0):
    '''
    将 28*28 的 MNIST 手写数字图像 基于PCA进行压缩
    
    '''
    #mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    filtered_data = filter_3_and_6((mnist_dataset.data, mnist_dataset.targets))
    X_data, y = filtered_data  # X 图像数据 y 标签
    
    X_data_3, y_data_3 = sample_data(X_data, y, label=3, sample_ratio=1.0)
    X_data_6, y_data_6 = sample_data(X_data, y, label=6, sample_ratio=1.0)
    #合并抽样后的数据
    X_sampled = torch.cat((X_data_3, X_data_6), dim=0)
    y_sampled = torch.cat((y_data_3, y_data_6), dim=0)
    
    n_samples = X_sampled.shape[0]
    X_flattened = X_sampled.view(n_samples, -1)  # 将图像展平为一维向量
    X_flattened = X_flattened/255
    pca = PCA(n_components=PCA_dim)
    X_pca = pca.fit_transform(X_flattened)
    
    # 将 PCA 处理后的值缩放到 [0, π] 之间
    X_pca_min = np.min(X_pca)
    X_pca_max = np.max(X_pca)
    X_pca_scaled = np.pi * (X_pca - X_pca_min) / (X_pca_max - X_pca_min)
    
    y_sampled[y_sampled==3]=1
    y_sampled[y_sampled==6]=0
    y_sampled[y_sampled==3]=1
    y_sampled[y_sampled==6]=0
    y_sampled = y_sampled.numpy()
    
    return X_pca_scaled,y_sampled

X_full,y_full = PCA_data_preprocessing_micro(mnist_dataset,8,0.95)

    



def getfulldata(mnist_dataset:datasets.MNIST,PCA_dim:int=8):
    filtered_data = filter_3_and_6((mnist_dataset.data, mnist_dataset.targets))
    X_data, y = filtered_data  # X 图像数据 y 标签
    X_flattened = X_data.view(X_data.shape[0], -1)  # 将图像展平为一维向量
    X_flattened = X_flattened/255
    pca = PCA(n_components=PCA_dim)
    X_pca = pca.fit_transform(X_flattened)
    X_pca_min = np.min(X_pca)
    X_pca_max = np.max(X_pca)
    X_pca_scaled = np.pi * (X_pca - X_pca_min) / (X_pca_max - X_pca_min)
    y[y==3]=1
    y[y==6]=0
    y[y==3]=1
    y[y==6]=0
    return X_pca_scaled,y
x_fulldata, y = getfulldata(mnist_dataset,8)

    
    
    
    
    
def amplitude_encoding(X_train:np.array,X_test:np.array):
    trian_params = np.array([amplitude_param(pixels=i.flatten()) for i in X_train])
    test_params  = np.array([amplitude_param(pixels=i.flatten()) for i in X_test])
    
    return trian_params,test_params


    
            

    