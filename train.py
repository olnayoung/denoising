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


class trainNetwork():
    def __init__(self):
        self.class_num = args.class_num
        self.batch_num = args.batch_num
        self.n_epochs = args.n_epochs

        self.input_path = os.path.join(args.train_path, 'train')
        self.label_path = os.path.join(args.train_path, 'train_cleaned')
        self.save_path = os.path.join(args.train_path, 'params')
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        self.l_r = args.l_r
        self.b_size = args.batch_size
        self.model = Net()
        self.model.cuda()

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.l_r)
        elif args.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.l_r, momentum=0.9, weight_decay=0.0001)

        if args.lossname == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()
        elif args.lossname == 'L1':
            self.criterion = nn.L1Loss()
        elif args.lossname == 'L2':
            self.criterion = nn.MSELoss()

    
    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins *60))
        return elapsed_mins, elapsed_secs


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


    def train_epoch(self, input_lists):
        self.model.train()

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

            loss.backward()
            self.optimizer.step()

            return losses / self.batch_num


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


    def train(self):
        file_lists = (self.list_files(self.input_path))
        past_loss = 0

        for epoch in range(self.n_epochs):
            np.random.shuffle(file_lists)

            input_lists = file_lists[:int(len(file_lists)*0.9)]
            val_input_lists = file_lists[int(len(file_lists)*0.9)+1:]

            start_time = time.time()
            train_loss = self.train_epoch(input_lists)
            val_loss = self.val_epoch(val_input_lists)
            end_time = time.time()

            if past_loss < val_loss:
                torch.save(self.model.state_dict(), self.save_path + '/' + str(epoch+1) + '.pt')
                past_loss = val_loss

            mins, secs = self.epoch_time(start_time, end_time)
            print()
            print('--------------------------------------------------------------')
            print('Epoch: %02d | Epoch Time: %dm %ds' % (epoch+1, mins, secs))
            print('\tTrain Loss: %.3f | Val Loss: %.3f ' % (train_loss, val_loss))



def main():
    trainNetwork().train()

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, default='/home/ny/pytorch_codes/Data')

    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--class_num', type=int, default=43)
    parser.add_argument('--l_r', type=float, default=10**(-4))

    parser.add_argument('--lossname', type=str, default='L2')
    parser.add_argument('--optimizer', type=str, default='Adam')


    args = parser.parse_args()
    main()