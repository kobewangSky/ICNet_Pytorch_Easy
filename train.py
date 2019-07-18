from torch import optim
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import argparse
from dataset import Cityscapesloader

from model import ICnet_model
from collections import deque
import os




LAMBDA1 = 0.16
LAMBDA2 = 0.4
LAMBDA3 = 1.0

LABELWEIGHT = {'label4' : LAMBDA3, 'label8' : LAMBDA2, 'label16' : LAMBDA1}


def train(dataloader, Model, optimizer, losses, lr0 = None, epochs = 10,  testDataset = None, Path_to_save_model = None):

    minloss = 9999

    for epoch in range(epochs):

        if epoch > epochs/2:
            lr = lr0 / 10
        else:
            lr = lr0

        for g in optimizer.param_groups:
            g['lr'] = lr

        for batch_index, (image_batch, labels_batch) in enumerate(dataloader):


            image = image_batch.cuda()
            labels_batch = labels_batch.cuda()

            sub_4, sub_8, sub_16 = Model.forward(image)

            loss = Model.loss( sub_4 , sub_8 , sub_16 , labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            avg_loss = sum(losses) / len(losses)
            print(f'epach = {epoch}, lr = {lr}, index = {batch_index},  loss = {avg_loss:.6f}')

        losstest = 0
        if testDataset != None:
            for test_index, (testimage, testlabel) in enumerate(testDataset):
                testimage = testimage.cuda()
                testlabel = testlabel.cuda()
                sub_4, sub_8, sub_16 = Model.forward(testimage)
                temp = Model.loss(sub_4 , sub_8  , sub_16 , testlabel)
                losstest = losstest + temp.item()
            losstest = losstest / testDataset.__len__()
            print(f'test loss = {losstest}, minloss = {minloss}')
            if losstest < minloss:
                minloss = losstest
                Model.save(Path_to_save_model, 'best')

        Model.save(Path_to_save_model, 'last')






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=400, help = 'number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--resume',action = 'store_true', help='load weight')
    parser.add_argument('--lr', type=float, default=0.0001, help = 'learning rate' )
    parser.add_argument('--save_path', default='./checkpoints', help='save path')
    opt = parser.parse_args()


    def main():

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        traindataset = Cityscapesloader('Cityscapes/', split='train', )
        traindataloader = DataLoader(traindataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

        testdataset = Cityscapesloader('Cityscapes/', split='val', )
        testdataloader = DataLoader(testdataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

        ICnet_ = ICnet_model(19, (512, 1024)).cuda()
        ICnet_.setuplabelweight(LABELWEIGHT)
        if opt.resume == True:
            ICnet_.load('./checkpoints/last.pth')

        optimizer = optim.SGD(ICnet_.parameters(), lr=opt.lr, weight_decay=0.0005)
        #optimizer = optim.Adam(ICnet_.parameters(), lr=opt.lr)

        losses = deque(maxlen=1000)
        train( traindataloader, ICnet_, optimizer, losses, opt.lr, opt.epochs, testDataset= testdataloader, Path_to_save_model= opt.save_path)


    main()