import numpy as np

from torch.utils.data import Dataset
import torch

class SiamesePair(Dataset):
    """
    For each sample creates randomly a positive or a negative pair
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.train = self.dataset.train

        if self.train:
            self.labels = self.dataset.labels
            self.samples = self.dataset.samples
            self.labels_set = set(self.labels.numpy())
            self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate sample pairs
            self.labels = self.dataset.labels
            self.samples = self.dataset.samples
            self.labels_set = set(self.labels.numpy())
            self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                     for label in self.labels_set}
            random_state = np.random.RandomState(29) #初始化一个随机数生成器
            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.labels[i].item()]),
                               1]
                              for i in range(0, len(self.samples), 2)]
            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.samples), 2)]
            self.test_pairs = positive_pairs + negative_pairs


    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            sample1, label1 = self.samples[index], self.labels[index].item()
            if target == 1: #output positive pair
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1]) #select a sample with the same label

            else: #output negative pair
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label]) #select a sample with different label
            sample2 = self.samples[siamese_index]
        else:
            sample1 = self.samples[self.test_pairs[index][0]]
            sample2 = self.samples[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]
        return (sample1, sample2), target


    def __len__(self):
        return len(self.dataset)



class IncrementalDataset(Dataset):
    """
    obtain sample pairs for incremental learning
    """

    def __init__(self, train_User, Defult_User, UserID, epoch):
        self.train_User = train_User
        self.Defult_User = Defult_User
        self.UserID = UserID
        self.train_User_samples = self.train_User.samples
        self.train_User_labels = self.train_User.labels
        self.Defult_User_samples = self.Defult_User.samples
        self.Defult_User_labels = self.Defult_User.labels
        self.epoch = epoch

    def __getitem__(self, index):
        if index % 2 == 0:
            target = 1
        else:
            target = 0
        # get a legal sample
        sample1, label1 = self.train_User_samples[index], self.train_User_labels[index].item()
        if target == 1:  #output positive pair
            torch.manual_seed(self.epoch+index)
            torch.cuda.manual_seed(self.epoch+index)
            sample2 = self.train_User_samples[torch.randint(0, self.train_User_samples.size(0), (1,))]

        else:  #output negative pair
            torch.manual_seed(self.epoch+index)
            torch.cuda.manual_seed(self.epoch+index)
            sample2 = self.Defult_User_samples[torch.randint(0, self.Defult_User_samples.size(0), (1,))]
        sample2 = torch.reshape(sample2, (1, -1))
        return (sample1, sample2), target

    def __len__(self):
        return len(self.train_User)




