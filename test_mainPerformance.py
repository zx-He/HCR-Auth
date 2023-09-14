

import torch
import random
import torch.optim as optim
import torch.nn as nn
import numpy as np

from DataSet import getDataset, getIllegal, GetSubset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from losses import ContrastiveLoss
from networks import SiameseNet
from trainer import fit


def count_samples_above_threshold(lst, t):
    count = 0
    for sample in lst:
        if sample > t:
            count += 1
    return count

if __name__ == '__main__':


    # set hyperparameters
    seed = 10000
    lr = 0.001
    n_epochs = 40
    batchSize = 8
    trainSampleNum = 10

    distance_le = []
    distance_ile = []
    dataset_path = '/root/project/Paper2/mainDataset'

    for i in range(1, 46):
        embedding_net = torch.load("embedding_net.pth") #load pre-trained base model
        embedding_net.train()
        for name, module in embedding_net.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = 0  # set dropRate=0

        UserID = i
        print("UserID:", UserID)

        Defult_User = getDataset(dataset_path, startUser=46, endUser=61) # get default Dataset
        User = getDataset(dataset_path, startUser=UserID, endUser=UserID + 1) # get legal Dataset
        Illegal_User = getIllegal(dataset_path, startUser=1, endUser=46,
                                  legalUser=UserID)  # get Illegal Dataset

        # divide trainSet and testSet
        train_indices = []
        test_indices = []
        indices = [j for j, label in enumerate(User.labels) if label == UserID]
        random.seed(seed)
        train_indices.extend(random.sample(indices, trainSampleNum))  # select samples to train user-specific model
        test_indices.extend([index for index in indices if index not in train_indices])  # others as test samples
        train_User = GetSubset(User, train_indices, train=True)
        test_User = GetSubset(User, test_indices, train=False)

        cuda = torch.cuda.is_available()
        margin = 1.
        model = SiameseNet(embedding_net)
        if cuda:
            model.cuda()

        # set loss function, optimizer
        loss_fn = ContrastiveLoss(margin)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)
        log_interval = 50
        trainloss = fit(train_User, Defult_User, UserID, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval) # train user-specific model

        # save user-specific model
        # torch.save(embedding_net, "Siamese_User_" + str(UserID) + "_twoEarphone.pth")

        # test user-specific model
        # calculate template
        temp_dataloader = DataLoader(train_User, batch_size=1)
        output_list = []
        with torch.no_grad():
            for data in temp_dataloader:
                sample, target = data
                sample = sample.cuda()
                output = embedding_net(sample)
                output_list.append(output)
        concatenated = torch.cat(output_list, dim=0)
        summed = torch.sum(concatenated, dim=0)
        template = summed / len(output_list)
        template = template.unsqueeze(0)

        # get test dataset
        Illegal_dataloader = DataLoader(Illegal_User, batch_size=1)
        legal_dataloader = DataLoader(test_User, batch_size=1)
        embedding_net.eval()

        # calculate distance
        with torch.no_grad():
            for data in legal_dataloader:
                samples, label = data
                samples = samples.cuda()
                outputs = embedding_net(samples)
                distance = torch.dist(template, outputs)
                distance_le.append(distance)

        with torch.no_grad():
            for data in Illegal_dataloader:
                samples, label = data
                samples = samples.cuda()
                outputs = embedding_net(samples)
                distance = torch.dist(template, outputs)
                distance_ile.append(distance)

    # calculate metric
    thresh = 0.46
    # thresh_list = np.arange(0.3, 0.7, 0.01)
    # for thresh in thresh_list:
    FN = count_samples_above_threshold(distance_le, thresh)
    TP = len(distance_le) - FN
    TN = count_samples_above_threshold(distance_ile, thresh)
    FP = len(distance_ile) - TN
    FAR = FP /(FP+TN)
    FRR = FN /(TP+FN)
    TPR = TP /(TP+FN)
    TNR = TN /(TN+FP)
    BAC = 1/2 * (TPR + TNR)

    message = '\nthresh:{}, trainSampleNum:{}, batchSize: {}, lr: {}, n_epochs: {}, seed: {}, FAR: {}, FRR: {}, BAC: {}'.format(thresh, trainSampleNum, batchSize, lr, n_epochs, seed, FAR, FRR, BAC)
    print(message)







