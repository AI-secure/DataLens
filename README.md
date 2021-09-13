# DataLens

This is the official code base for our ACM CCS 2021 paper:

["DataLens: Scalable Privacy Preserving Training via Gradient Compression and Aggregation".](https://arxiv.org/abs/2103.11109)

Boxin Wang, Fan Wu, Yunhui Long, Luka Rimanic, Ce Zhang, Bo Li

## Citation
```
@article{wang2021datalens,
  title={DataLens: Scalable Privacy Preserving Training via Gradient Compression and Aggregation},
  author={Wang, Boxin and Wu, Fan and Long, Yunhui and Rimanic, Luka and Zhang, Ce and Li, Bo},
  journal={arXiv preprint arXiv:2103.11109},
  year={2021}
}
```

## Usage

### Prepare your environment

Download required packages

```shell script
pip install -r requirements.txt
```

### Prepare your data
Please store the training data in `$data_dir`. By default, `$data_dir` is set to `../../data`.

We provide a script to download the MNIST and Fashion Mnist datasets. 

```shell script
python download.py [dataset_name]
```

For MNIST, you can run 

```shell script
python download.py mnist
```

For Fashion-MNIST, you can run 

```shell script
python download.py fashion_mnist
```

For CelebA and Places365 datasets, please refer to their official websites for downloading. 

### Training 

```shell script
python main.py --checkpoint_dir [checkpoint_dir] --dataset [dataset_name] --train
```

For example, to train the Datalens on Fashion-MNIST given eps=1 and delta=1e-5
```shell script
python main.py --checkpoint_dir fmnist_z_dim_50_topk_200_teacher_4000_sigma_5000_thresh_0.7_pt_30_d_step_2_stochastic_1e-5/ \
--topk 200 --signsgd --norandom_proj --shuffle  --teachers_batch 80 --batch_teachers 50 \
--dataset fashion_mnist --train --max_eps 1 --train --thresh 0.7 --sigma 5000 --nopretrain \
--z_dim 50 --nosave_epoch --epoch 300 --save_vote --d_step 2 --pretrain_teacher 10 --stochastic --max_grad 1e-5
```

By default, after it reaches the max epsilon, it will generate 10 batches of 10,000 DP samples as `eps-1.00.data-{i}.pkl` (i=0,...9) in `checkpoint_dir`.

### Training Args

```
main.py:
  --ae: AE model name
    (default: '')
  --batch_size: The size of batch images [64]
    (default: '30')
    (an integer)
  --batch_teachers: Number of teacher models in one batch
    (default: '1')
    (an integer)
  --beta1: Momentum term of adam [0.5]
    (default: '0.5')
    (a number)
  --checkpoint_dir: Directory name to save the checkpoints [checkpoint]
    (default: 'checkpoint')
  --checkpoint_name: checkpoint model name [checkpoint]
    (default: 'checkpoint')
     --[no]crop: True for cropping
    (default: 'false')
  --d_step: steps of the discriminator
    (default: '1')
    (an integer)
  --data_dir: Root directory of dataset [data]
    (default: '../../data')
  --dataset: The name of dataset [cinic, celebA, mnist, lsun, fire-small]
    (default: 'slt')
  --delta: delta for differential privacy
    (default: '1e-05')
    (a number)
  --epoch: Epoch for training teacher models
    (default: '1000')
    (an integer)
  --[no]finetune_ae: Finetune ae
    (default: 'false')
  --g_epoch: Epoch for training the student models
    (default: '500')
    (an integer)
  --g_step: steps of the generator
    (default: '1')
    (an integer)
  --generator_dir: Directory name to save the generator
    (default: 'generator')
  --hid_dim: Dimmension of hidden dim
    (default: '512')
    (an integer)
  --[no]increasing_dim: Increase the projection dimension for each epoch
    (default: 'false')
  --input_height: The size of image to use (will be center cropped). 
    (default: '32')
    (an integer)
  --input_width: The size of image to use (will be center cropped). If None, same value as input_height [None]
    (default: '32')
    (an integer)
  --klevel: Levels of gradient quantization
    (default: '4')
    (an integer)
  --[no]klevelsgd: Apply klevel sgd for gradient agggregation
    (default: 'false')
  --learning_rate: Learning rate of for adam
    (default: '0.001')
    (a number)
  --[no]load_d: True for loading the pretrained models w/ discriminator, False for not load [True]
    (default: 'true')
  --loss: AE reconstruction loss
    (default: 'l1')
  --max_eps: maximum epsilon
    (default: '1.0')
    (a number)
  --max_grad: maximum gradient for signsgd aggregation
    (default: '0.0')
    (a number)
  --[no]mean_kernel: Apply Mean Kernel for gradient agggregation
    (default: 'false')
  --[no]non_private: Do not apply differential privacy
    (default: 'false')
  --orders: rdp orders
    (default: '200')
    (an integer)
  --output_height: The size of the output images to produce [64]
    (default: '32')
    (an integer)
  --output_width: The size of the output images to produce. If None, same value as output_height [None]
    (default: '32')
    (an integer)
  --[no]pca: Apply pca for gradient aggregation
    (default: 'false')
  --pca_dim: principal dimensions for pca
    (default:'10')                                                           
    (a number)
  --[no]pretrain: True for loading the pretrained models, False for not load [True]
    (default: 'true')
  --pretrain_teacher: Pretrain teacher for epochs
    (default: '0')
    (an integer)
  --proj_mat: #/ projection mat
    (default: '1')
    (an integer)
  --[no]random_label: random labels for training data, only used when pretraining some models
    (default: 'false')
  --[no]random_proj: Apply pca for gradient aggregation
    (default: 'true')
  --sample_dir: Directory name to save the image samples [samples]
    (default: 'samples')
  --sample_step: Number of teacher models in one batch
    (default: '10')
    (an integer)
  --[no]save_epoch: Save each epoch per 0.1 eps
    (default: 'false')
  --[no]save_vote: Save voting results
    (default: 'false')
  --[no]shuffle: Evenly distribute dataset
    (default: 'true')
  --sigma: Scale of gaussian noise for gradient aggregation
    (default: '2000.0')
    (a number)
  --sigma_thresh: Scale of gaussian noise for thresh gnmax
    (default: '4500.0')
    (a number)
  --[no]signsgd: Apply sign sgd for gradient agggregation
    (default: 'false')
  --[no]signsgd_dept: Apply sign sgd for gradient agggregation with data dependent bound
    (default: 'false')
  --[no]signsgd_nothresh: Apply sign sgd for gradient agggregation
    (default: 'false')
  --[no]simple_gan: Use fc to build GAN
    (default: 'false')
  --[no]sketchsgd: Apply sketch sgd for gradient agggregation
  (default: 'false')
  --[no]small: Use a smaller discriminator
    (default: 'false')
  --step_size: Step size for gradient aggregation
    (default: '0.0001')
    (a number)
  --[no]stochastic: Apply stochastic sign sgd for gradient agggregation
    (default: 'false')
  --[no]tanh: Use tanh as activation func
    (default: 'false')
  --teacher_dir: Directory name to save the teacher [teacher]
    (default: 'teacher')
  --teachers_batch: Number of batch
    (default: '1')
    (an integer)
  --thresh: threshhold for threshgmax
    (default: '0.5')
    (a number)
  --topk: Number of top k gradients
    (default: '50')
    (an integer)
  --[no]train: True for training, False for testing [False]
    (default: 'false')
  --[no]train_ae: Train ae
    (default: 'false')
  --train_size: The size of train images [np.inf]
    (default: 'inf')
    (a number)
  --[no]wgan: Train wgan
    (default: 'false')
  --y_dim: #/ y dim
    (default: '10')
    (an integer)
  --z_dim: #/ z dim
    (default: '100')
    (an integer)
```


## Generating synthetic samples

```shell script
python main.py --checkpoint_dir [checkpoint_dir] --dataset [dataset_name]
```

## Evaluate the synthetic records 

We train a classifier on synthetic samples and test it on real samples. We put the evaluation script under the `evaluation` folder.

For MNIST,
```shell script
python evaluation/train-classifier-mnist.py --data [DP_data_dir]
```


For Fashion-MNIST,
```shell script
python evaluation/train-classifier-fmnist.py --data [DP_data_dir]
```

For CelebA-Gender,
```shell script
python evaluation/train-classifier-celebA.py --data [DP_data_dir]
```

For CelebA-Hair,
```shell script
python evaluation/train-classifier-hair.py --data [DP_data_dir]
```

The `[DP_data_dir]` is where your generated DP samples are located. In the Fashion-MNIST example above, we have generated 10 bathces of DP samples in `$checkpoint_dir/eps-1.00.data-{i}.pkl` (i=0,...,9). During evaluation, you should run with the prefix of the `data_dir`, where the program will concatenate all of the generated DP samples and use it as the training data. 

```shell script
python evaluation/train-classifier-hair.py --data $checkpoint_dir/eps-1.00.data
```