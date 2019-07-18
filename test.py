from torch import optim
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import DataLoader
from dataset import Cityscapesloader
from model import ICnet_model
from collections import deque
import os
import numpy as np
from metrics import Evaluator

from torchsummary import summary


if __name__ == '__main__':
    def main():
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"

        testdataset = Cityscapesloader('Cityscapes/', split='val', )
        testdataloader = DataLoader(testdataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)


        ICnet_ = ICnet_model(19, (512, 1024)).cuda()

        model_path = 'checkpoints/last.pth'
        if os.path.exists(model_path):
            ICnet_.load(model_path)

        summary(ICnet_, ( 3, 512, 1024))

        ICnet_.eval()

        eval = Evaluator(19)
        eval.reset()
        with torch.no_grad():
            for batch_index, (image_batch, labels_batch) in enumerate(testdataloader):
                print(batch_index)
                image = image_batch.cuda()
                labels_batch = np.array(labels_batch)


                output = ICnet_.forward(image).cpu().numpy()

                pred = np.argmax(output, axis=1)



                eval.add_batch(labels_batch, pred)

        Pixel_Accuracy = eval.Pixel_Accuracy()

        print('Pixel_Accuracy = ', Pixel_Accuracy)



    main()