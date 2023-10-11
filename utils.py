import numpy as np
import cv2
import matplotlib.pyplot as plt
# from utils import *
import PIL.Image
import torch
import clip
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from torchvision import transforms
import torch
import clip

from ldm.util import instantiate_from_config
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import torch
import torch.nn.functional as F
import json
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import skimage.io as io
from PIL import Image 
import json
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import cv2
from torchvision.utils import save_image, make_grid
from skimage.metrics import structural_similarity as ssim
import lpips
from numpy.linalg import solve
import argparse
transform = transforms.Compose([
        
        transforms.Resize((224, 224), interpolation=BICUBIC),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
transform768 = transforms.Compose([
        
        transforms.Resize((768, 768), interpolation=BICUBIC),
        # transforms.CenterCrop((768, 768)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def visualize(X1, X2, name):
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(X1.shape[0]), np.ones(X2.shape[0])))
    X = TSNE(n_components=2).fit_transform(X)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.savefig(name)
    return

def alignB2A(A, B):
    std_norm_fMRI_list = (B - np.mean(B,axis=0)) / np.std(B,axis=0)  # 白化
    B = std_norm_fMRI_list * np.std(A,axis=0) + np.mean(A,axis=0)
    return B


def get_inter(generated_f, feature_list, idx_sort, w_c=1):
        # |  vg - sum( wi*vi )|   et. sum(wi) = 1
        # == | vg - v0 - sum( wi*vi) |   et. w = [1,w1,...,wn]
        nearnN = 5
        generated_f = np.expand_dims(generated_f.numpy(), 0)
        A_0 = np.expand_dims(feature_list[idx_sort[0],:], 0)
        A_m = A_0
        for i in range(1,nearnN):
            A_m = np.concatenate((A_m,np.expand_dims(feature_list[idx_sort[i],:], 0)), axis=0)
        
        A_0 = np.array(A_0)
        A_m= np.array(A_m).T
        A_m0 = np.concatenate((A_m[:,1:]-A_0.T, np.ones((1,nearnN-1))*10), axis=0)

        A = np.dot(A_m0.T, A_m0)
        b = np.zeros((1, generated_f.shape[1]+1))
        b[0,0:generated_f.shape[1]] = generated_f-A_0

        B = np.dot(A_m0.T, b.T)

        x = solve(A, B)

        xx = np.zeros((nearnN,1))
        xx[0,0] = 1 - x.sum()
        xx[1:,0] = x[:,0]

        
        vec_mu = np.dot(A_m, xx).T * w_c + (1-w_c)* generated_f

        return vec_mu


def euclidean_distance(descriptor1, descriptor2):
    return np.sqrt(np.sum((descriptor1 - descriptor2)**2))

def calculate_sift_distance(imgs1, imgs2):
    dis = 0
    for i in range(imgs1.shape[0]):
        img1 = np.transpose(imgs1[i], (1,2,0))
        img2 = np.transpose(imgs2[i], (1,2,0))

        # 初始化SIFT检测器
        sift = cv2.SIFT_create()
        img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        # 检测并计算图像的SIFT特征
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # 使用BFMatcher匹配器计算特征向量之间的距离
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # 应用比例测试来筛选好的匹配点
        good_matches = []
        for m, n in matches:
            if m.distance < 0.85 * n.distance:
                good_matches.append(m)

        # 计算好的匹配点的SIFT距离
        distances = []
        for match in good_matches:
            descriptor1 = des1[match.queryIdx]
            descriptor2 = des2[match.trainIdx]
            distance = euclidean_distance(descriptor1, descriptor2)
            distances.append(distance)

        # 计算SIFT距离的平均值
        if len(distances)>0:
            dis += np.mean(distances)
    return dis / imgs1.shape[0]

# 输入np格式 0-255
def get_semantic_metrix(imgs1, imgs2, transform, device):
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    net = nn.Sequential(*list(net.children())[:-1]).to(device)
    dis = 0
    for i in range(imgs1.shape[0]):
        img1 = np.transpose(imgs1[i], (1,2,0))
        img2 = np.transpose(imgs2[i], (1,2,0))
        img1 = Image.fromarray(img1.astype('uint8'), 'RGB')
        img2 = Image.fromarray(img2.astype('uint8'), 'RGB')
        img1 = transform(img1)
        img2 = transform(img2)

        feat1 = net(img1.unsqueeze(0).float().to(device))
        feat2 = net(img2.unsqueeze(0).float().to(device))
        dis += torch.norm(feat1 - feat2)
    return dis / imgs1.shape[0]

# 输入np，【3 256 256】
def get_ssim(imgs1, imgs2):
    dis = 0
    for i in range(imgs1.shape[0]):
        img1 = imgs1[i]
        img2 = imgs2[i]
        dis += (ssim(img1[0], img2[0]) + ssim(img1[1], img2[1]) + ssim(img1[2], img2[2])) / 3
    return dis / imgs1.shape[0]


def density_estimation(xtr, xte, k, low_dim, show_img=False):
    # first project by pca to lower dimension
    from sklearn.decomposition import PCA
    pca = PCA(n_components=low_dim)
    pca.fit(xtr)
    low_feats_tr = pca.transform(xtr)
    low_feats_te = pca.transform(xte)

    # scale the tr normal distribution, scale the te with tr's mean and std
    # low_feats_te = (low_feats_te - np.mean(low_feats_tr, axis=0)) / np.std(low_feats_tr, axis=0)
    # low_feats_tr = (low_feats_tr - np.mean(low_feats_tr, axis=0)) / np.std(low_feats_tr, axis=0)
    low_feats_te = (low_feats_te - np.mean(low_feats_te, axis=0)) / np.std(low_feats_te, axis=0)
    low_feats_tr = (low_feats_tr - np.mean(low_feats_tr, axis=0))/ np.std(low_feats_tr, axis=0)
    

    # visual lowfeattr and lowfeatte in two dimension
    if show_img:
        import matplotlib.pyplot as plt
        plt.scatter(low_feats_tr[:, 0], low_feats_tr[:, 1], c='r', label='train')
        plt.scatter(low_feats_te[:, 0], low_feats_te[:, 1], c='b', label='test')
        plt.legend()
        plt.savefig("images/lowdim_feats.png")
        # plt.show()

    from sklearn.neighbors import KernelDensity

    kde = KernelDensity(kernel='gaussian', bandwidth=1.5).fit(low_feats_tr)
    # estimate density by
    kde_density = kde.score_samples(low_feats_te)
    kde_density = np.exp(kde_density)
    return kde_density
    # # concate features and normalize to unit variance and zero mean
    # low_featus = np.concatenate((low_feats_tr, low_feats_te))
    # low_feats = (low_featus - np.mean(low_featus, axis=0)) / np.std(low_featus, axis=0)
    # # use knn to estimate the density p(x) = k / (n * h^d)
    # from sklearn.neighbors import NearestNeighbors
    # neigh = NearestNeighbors(n_neighbors=k)
    
def knn_density_estimation(xtr, xte, k, low_dim= 2, show_img=False):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=low_dim)
    pca.fit(xtr)
    low_feats_tr = pca.transform(xtr)
    low_feats_te = pca.transform(xte)
    
    # # scale the tr normal distribution, scale the te with tr's mean and std
    # low_feats_te = (low_feats_te - np.mean(low_feats_tr, axis=0)) / np.std(low_feats_tr, axis=0)
    # low_feats_tr = (low_feats_tr - np.mean(low_feats_tr, axis=0)) / np.std(low_feats_tr, axis=0)
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=k)
    low_feats = np.concatenate((low_feats_tr, low_feats_te))
    neigh.fit(low_feats_tr)
    distance, indices = neigh.kneighbors(low_feats[-50:])
    # p(x) = k / (n * h^d) 其中体积公式的其他参数会在比值中抵消
    density = k / (low_feats.shape[0] * np.power(distance[:, -1], low_dim)) 
    return density


def normalize(img):
    # if img.shape[-1] == 3:
    #     img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img


# compute the entropy of the distribution
def compute_entropy(p):
    return -np.sum(p * np.log(p + 1e-8), axis=1)

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model




def process_imgs(raw_imgs, transform):

    list = []
    for img in raw_imgs:
        img = img.transpose(1,2,0)
        min, max = img.min(), img.max()
        img = (img - min )/ (max - min)
        img = cv2.GaussianBlur(img,(1, 1),0)* 255
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        img = transform(img)
        list.append(img.unsqueeze(0))
    list = torch.cat(list, dim=0)
    return list
            
def range_loss(input):
    """(Katherine Crowson) - Spherical distance loss"""
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def tv_loss(input: torch.Tensor):
    """(Katherine Crowson) - L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])