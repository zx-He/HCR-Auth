
import os
import torch
from torch.utils.data import Dataset, Subset


class getDataset(Dataset):
    def __init__(self, root_dir, train=True, startUser = 46, endUser = 61):
        self.root_dir = root_dir
        self.samples = []
        self.labels = []
        self.train = train
        self.startUser = startUser
        self.endUser = endUser

        # get data and label from folder
        for i in range(startUser, endUser):  # UserID
            folder_path = os.path.join(root_dir, str(i))
            for j in range(1, 251):  #SampleID
                file_path = os.path.join(folder_path, str(j) + ".txt")
                with open(file_path, 'r') as file:
                    data = file.read().strip().split('\n')
                self.samples.append(list(map(float, data)))
                self.labels.append(i)
        self.samples = torch.tensor(self.samples)
        self.labels = torch.tensor(self.labels)
        self.samples = torch.unsqueeze(self.samples, 1)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        return torch.tensor(sample), torch.tensor(label)

    def __len__(self):
        return len(self.samples)


class getIllegal(Dataset):
    def __init__(self, root_dir, train=False, startUser=1, endUser=46, legalUser=5):
        self.root_dir = root_dir
        self.samples = []
        self.labels = []
        self.train = train
        self.startUser = startUser
        self.endUser = endUser
        self.legalUser = legalUser

        # get data and label from folder
        if startUser == legalUser:
            for i in range(startUser+1, endUser):  # User-ID
                folder_path = os.path.join(root_dir, str(i))
                for j in range(1, 251):  # Sample-ID
                    file_path = os.path.join(folder_path, str(j) + ".txt")
                    # read data from folder
                    with open(file_path, 'r') as file:
                        data = file.read().strip().split('\n')
                    self.samples.append(list(map(float, data)))
                    self.labels.append(i)
        elif endUser == legalUser+1:
            for i in range(startUser, endUser-1):
                folder_path = os.path.join(root_dir, str(i))
                for j in range(1, 251):
                    file_path = os.path.join(folder_path, str(j) + ".txt")
                    with open(file_path, 'r') as file:
                        data = file.read().strip().split('\n')
                    self.samples.append(list(map(float, data)))
                    self.labels.append(i)
        else:
            for i in range(startUser, legalUser):
                folder_path = os.path.join(root_dir, str(i))
                for j in range(1, 251):
                    file_path = os.path.join(folder_path, str(j) + ".txt")
                    with open(file_path, 'r') as file:
                        data = file.read().strip().split('\n')
                    self.samples.append(list(map(float, data)))
                    self.labels.append(i)
            for i in range(legalUser+1, endUser):
                folder_path = os.path.join(root_dir, str(i))
                for j in range(1, 251):
                    file_path = os.path.join(folder_path, str(j) + ".txt")
                    with open(file_path, 'r') as file:
                        data = file.read().strip().split('\n')
                    self.samples.append(list(map(float, data)))
                    self.labels.append(i)
        self.samples = torch.tensor(self.samples)
        self.labels = torch.tensor(self.labels)
        self.samples = torch.unsqueeze(self.samples, 1)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        return torch.tensor(sample), torch.tensor(label)

    def __len__(self):
        return len(self.samples)



class GetSubset(Subset):
    def __init__(self, dataset, indices, train=True, transform=None):
        super().__init__(dataset, indices)
        self.train = train
        self.transform = transform
        self.samples = dataset.samples[indices]
        self.labels = dataset.labels[indices]

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return torch.tensor(sample), torch.tensor(label)

    def __len__(self):
        return len(self.samples)


# class getIncreSet(Dataset):
#     def __init__(self, train=True, sampleList=None, labelList=None):
#         if labelList is None:
#             self.labels = []
#         if sampleList is None:
#             self.samples = []
#         sampleList = [list(map(float, sublist)) for sublist in sampleList]
#         self.samples = sampleList
#         self.labels = labelList
#         self.train = train
#         self.samples = torch.tensor(self.samples)
#         self.labels = torch.tensor(self.labels)
#         self.samples = torch.unsqueeze(self.samples, 1)
#
#     def __getitem__(self, index):
#         sample = self.samples[index]
#         label = self.labels[index]
#         return torch.tensor(sample), torch.tensor(label)
#
#     def __len__(self):
#         return len(self.samples)


