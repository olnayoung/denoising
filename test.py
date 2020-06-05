import torch
import torch.nn as nn
import torch.optim as optim
import glob
import cv2
import numpy as np
import os
import time
import argparse
import random

from network import Net


class testNetwork():
    def __init__(self):
        self.class_num = args.class_num
        self.batch_num = args.batch_num

        self.test_input_path = os.path.join(args.test_path, 'CORD', 'test', 'BERTgrid')
        self.test_label_path = os.path.join(args.test_path, 'CORD', 'test', 'LABELgrid')
        weight_path = os.path.join(args.test_path, 'params', args.weight)

        self.model = Net()
        self.model.cuda()
        self.model.load_state_dict(torch.load(weight_path))
        if args.lossname == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()


    def list_files(self, in_path):
        img_files = []
        for (dirpath, dirnames, filenames) in os.walk(in_path):
            for file in filenames:
                filename, ext = os.path.splitext(file)
                ext = str.lower(ext)
                if ext == '.png':
                    img_files.append(file)
                    # img_files.append(os.path.join(dirpath, file))
        img_files.sort()
        return img_files


    def load_data(self, img_names, label_names):
        assert len(img_names) == len(label_names)

        for idx in range(len(img_names)):
            image = cv2.imread(img_names[idx])
            label = cv2.imread(label_names[idx])

            images = np.zeros((1, self.b_size, self.b_size, 3))
            labels = np.zeros((1, self.b_size, self.b_size, 3))

            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)

            _, hei, wid, _ = image.shape
            for h in range(0, hei, self.b_size):
                for w in range(0, wid, self.b_size):
                    if h+self.b_size < hei:
                        start_h, end_h = h, h+self.b_size
                    else:
                        start_h, end_h = hei-self.b_size-1, hei-1
                    
                    if w+self.b_size < wid:
                        start_w, end_w = w, w+self.b_size
                    else:
                        start_w, end_w = wid-self.b_size-1, wid-1

                    temp_image = image[:, start_h:end_h, start_w:end_w, :]
                    temp_label = label[:, start_h:end_h, start_w:end_w, :]

                    images = np.concatenate((images, temp_image), axis=0)
                    labels = np.concatenate((labels, temp_label), axis=0)

        return images, labels


    def val_epoch(self, input_lists):
        self.model.eval()

        losses = 0

        for batch in range(int(len(input_lists)/self.batch_num)):
            self.optimizer.zero_grad()
            input_list, label_list = [], []

            for num in range(self.batch_num):
                idx = batch*self.batch_num + num
                filename = input_lists[idx]

                input_list.append(self.input_path + '/'+filename)
                label_list.append(self.label_path + '/'+filename)
            
            train_input, train_label = self.load_data(input_list, label_list)

            input_tensor = torch.tensor(train_input, dtype=torch.float).cuda()
            input_tensor = input_tensor.permute(0, 3, 1, 2)
            label_tensor = torch.tensor(train_label, dtype=torch.float).cuda()
            label_tensor = label_tensor.permute(0, 3, 1, 2)

            output = self.model(input_tensor)
            loss = self.criterion(output, label_tensor)
            losses += loss.item()

            return losses / self.batch_num


    # def testMany(self):
    #     test_lists = self.list_files(self.test_input_path)
    #     val_loss, val_acc, val_mic, val_mac = self.val_epoch(test_lists)

    #     print('\tVal Loss: %.3f | Val Acc: %.2f%% | mic: %.2f%% | mac: %.2f%%' % (val_loss, val_acc*100, val_mic*100, val_mac*100))


    def testOne(self, img_path):
        self.model.eval()

        input = cv2.imread(img_path)
        input_tensor = torch.tensor(input, dtype=torch.float).cuda()
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.permute(0,3,1,2)

        output = self.model(input_tensor)
        output = output.permute(0,2,3,1)
        output = output.squeeze(0)
        output = output.cpu().detach().numpy()

        return output


def main():
    result = testNetwork().testOne('/home/ny/pytorch_codes/Data/test/1.png')
    print(result.shape)
    cv2.imwrite('/home/ny/pytorch_codes/result.png', result)

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_path', type=str, default='/home/ny/pytorch_codes/Data')
    parser.add_argument('--weight', type=str, default='27.pt')

    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--class_num', type=int, default=43)
    parser.add_argument('--lossname', type=str, default='L2')

    args = parser.parse_args()
    main()
