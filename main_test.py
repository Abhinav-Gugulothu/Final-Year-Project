from dataset import TripletDataset
from model import FaceNet
from sampler import samples
from train import train, load, save
from test import evalulate, test
from util import get_Optimizer, get_Scheduler, get_Sampler

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchsummary import summary

import os

if __name__ == "__main__":
    # config
    BATCH_SIZE=128
    #NUM_WORKERS = multiprocessing.cpu_count()
    NUM_WORKERS = 2
    embedding_size = 512
    # all or None(may not iterate all data in each batch)
    sampler = None
    weight_decay = 1e-3
    lr = 3e-6
    dropout = 0.3
    # resnet, effnet or None(IncepetionResNet)
    model_name = None
    pretrain = True
    # global gem or None(avgerage pooling)
    pool= None
    # Cyclic or Step
    scheduler_name = 'multistep'
    # sgd or None(adam) or rmsprop
    optimizer_type = None
    num_epochs = 20
    eval_every = 1000
    # margin for triplet loss
    margin=2
    # name to open or save the model
    name = 'arcface1.pt'
    load_local_model = False


    os.environ['CUDA_VISIBLE_DEVICES']='2'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device:",device)

    df_eval1 = pd.read_csv('../Datasets/new_eval_same.csv')
    df_eval2 = pd.read_csv('../Datasets/new_eval_diff.csv')
    df_test = pd.read_csv('../Datasets/new_test.csv')

    eval_dataset1 = TripletDataset(df_eval1, mode='eval')
    eval_dataset2 = TripletDataset(df_eval2, mode='eval')
    test_dataset = TripletDataset(df_test, mode='test')

    eval_loader1 = DataLoader(eval_dataset1, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=False)
    eval_loader2 = DataLoader(eval_dataset2, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=False)
    facenet = torch.load('../models/InceptionResNetV1_ArcFace.pt')
    device = torch.device('cpu')


    dist_threshold = evalulate(facenet, eval_loader1, eval_loader2, device)
    test(facenet,test_loader,dist_threshold,device)
