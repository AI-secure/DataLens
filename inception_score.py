import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    from tqdm import tqdm
    for i, batch in tqdm(enumerate(dataloader, 0)):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, orig, transform):
        self.transform = transform
        self.orig = orig

    def __getitem__(self, index):
        return self.transform(self.orig[index])

    def __len__(self):
        return len(self.orig)

dtype = torch.cuda.FloatTensor
inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)

def pipeline(datapath):
    import joblib
    import torchvision.transforms as transforms
    data = joblib.load(datapath)
    from sklearn.utils import shuffle
    data = shuffle(data)
    data = data[:10000,:784].reshape(-1, 28, 28, 1)
    data = np.concatenate((data, data, data), axis=3)
    data = np.float32(data)
    transform = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])
    data = MnistDataset(data, transform=transform)
    print(inception_score(data, cuda=True, batch_size=64, resize=True, splits=10))



if __name__ == '__main__':
    # (2.358112431133864, 0.04182949711390127)
    pipeline('/home/ubuntu/disk2/wbxshm/mnist_z_dim_50_topk_200_teacher_4000_sigma_5000_thresh_0.7_pt_30_d_step_2_v2_stochastic_b_1e-5_v2/eps-1.00.data')
