from SiameseDataset import SiamesePair
from DataSet import getDataset
import torch

from losses import ContrastiveLoss
from networks import SiameseNet, EmbeddingNet

from torch.optim import lr_scheduler
import torch.optim as optim

from preTrainer import fit

if __name__ == '__main__':

    path = '/root/project/Paper2/mainDataset' #DataSet path
    dataset = getDataset(path)
    siamese_train_dataset = SiamesePair(dataset)  # Returns pairs of Inputs and target same/different
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    batchSize = 16
    lr = 0.001
    dropRate = 0.2

    siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batchSize, shuffle=False, **kwargs)
    # loading model
    margin = 1.
    embedding_net = EmbeddingNet(dropRate)
    model = SiameseNet(embedding_net)
    if cuda:
        model.cuda()
    #set up loss function, optimizer
    loss_fn = ContrastiveLoss(margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1)
    n_epochs = 45

    fit(siamese_train_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda)
    #get pre-trained model
    torch.save(embedding_net, "embedding_net.pth")
