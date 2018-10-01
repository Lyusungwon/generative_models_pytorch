from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets, transforms
import os
import pandas as pd
import numpy as np
import cv2
import pickle
from PIL import Image

def train_loader(data, data_directory = '/home/sungwonlyu/data/', batch_size = 128, input_h = 128, input_w = 128, cpu_num = 0):
    if data == 'mnist':
        train_dataloader = DataLoader(
            datasets.MNIST(data_directory  + data, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'fashionmnist':
        train_dataloader = DataLoader(
            datasets.FashionMNIST(data_directory  + data, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'svhn':
        train_dataloader = DataLoader(
            datasets.SVHN(data_directory + data, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'lsun':
        train_dataloader = DataLoader(
            datasets.LSUN(data_directory  + data, classes = 'train', transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'cifar10':
        train_dataloader = DataLoader(
            datasets.CIFAR10(data_directory + data, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'alphachu':
        train_dataloader = DataLoader(
            AlphachuDataset(data_directory + data, train=True, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'clevr':
        train_dataloader = DataLoader(
            Clevr(data_directory + data + '/', train=True, 
            transform = transforms.Compose([transforms.Resize((input_h, input_w)),
                                                transforms.ToTensor()])),
            batch_size=batch_size, shuffle=True,
            num_workers = cpu_num)
            # collate_fn = collate_text)
    elif data == 'sortofclevr':
        train_dataloader = DataLoader(
            SortOfClevr(data_directory + data + '/', train=True),
            batch_size=batch_size, shuffle=True)    

    return train_dataloader

def test_loader(data, data_directory = '/home/sungwonlyu/data', batch_size = 128, input_h = 128, input_w = 128, cpu_num = 0):
    if data == 'mnist':
        test_dataloader = DataLoader(
            datasets.MNIST(data_directory + data, train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'fashionmnist':
        test_dataloader = DataLoader(
            datasets.FashionMNIST(data_directory + data, train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'svhn':
        test_dataloader = DataLoader(
            datasets.SVHN(data_directory + data, train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'lsun':
        test_dataloader = DataLoader(
            datasets.LSUN(data_directory + data, classes = 'test', transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'cifar10':
        test_dataloader = DataLoader(
            datasets.CIFAR10(data_directory + data, train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'alphachu':
        test_dataloader = DataLoader(
            AlphachuDataset(data_directory + data, train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    elif data == 'clevr':
        test_dataloader = DataLoader(
            Clevr(data_directory + data + '/', train=False, 
            transform = transforms.Compose([transforms.Resize((input_h, input_w)),
                                                transforms.ToTensor()])),
            batch_size=batch_size, shuffle=True,
            num_workers = cpu_num)
            # collate_fn = collate_text)
    elif data == 'sortofclevr':
        test_dataloader = DataLoader(
            SortOfClevr(data_directory + data + '/', train=False), 
            batch_size=batch_size, shuffle=True)

    return test_dataloader


class AlphachuDataset(Dataset):
    """Alphachu dataset."""
    def __init__(self, root_dir, train = True, transform=None):
        """
            Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.makelist()

    def makelist(self):
        img_list = os.listdir(self.root_dir)
        test = len(img_list) // 10
        if not self.train:
            img_list = img_list[:test]
        else:
            img_list = img_list[test:]
        # timestamps = [int(i[:12]) for i in img_list]
        # sets = [int(i[12:].split('-')[0]) for i in img_list]
        frames = [int(i.split('-')[1].split('.')[0]) for i in img_list]
        img_list = pd.DataFrame(img_list)
        # img_list['timestamps'] = timestamps
        # img_list['sets'] = sets
        img_list['frames'] = frames
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.root_dir + '/' + self.img_list.iloc[idx, 0]
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(np.array(image), 2)
        if self.transform:
            image = self.transform(image)
        frames = self.img_list.iloc[idx, 1]
        # timestamps = self.img_list.iloc[idx, 1]
        # sets = self.img_list.iloc[idx, 2]
        # sample = {'image': image, 'set': sets, 'frame': frames}
        return image, frames

class Clevr(Dataset):
    """Clevr dataset."""
    def __init__(self, root_dir, train = True, transform = None):
        self.root_dir = root_dir
        # self.mode = 'sample'
        self.mode = 'train' if train else 'val'
        self.transform = transform
        self.q_dir = self.root_dir + 'questions/'+ 'CLEVR_{}_questions.json'.format(self.mode)
        self.img_dir = self.root_dir + 'images/'+ '{}/'.format(self.mode)
        if self.mode == 'sample':
            self.img_dir = self.root_dir + 'images/train/'
        self.load_data()

    def make_data(self):
        q_corpus = set()
        a_corpus = set()
        modes = ['train', 'val', 'sample']
        q_list = dict()
        qa_list = defaultdict(list)
        for mode in modes:
            img_dir = self.root_dir + 'images/{}/'.format(mode)
            if mode == 'sample':
                img_dir = self.root_dir + 'images/train/'
            ann_dir = self.root_dir + 'questions/CLEVR_{}_questions.json'.format(mode)
            with open(self.root_dir + ann_dir) as f:
                q_list[mode] = json.load(f)['questions']
            for q_obj in q_list[mode]:
                img_dir = q_obj['image_filename']
                q_text = q_obj['question'].lower()
                q_text = re.sub('\s+', ' ', q_text)
                q_text_without_question_mark = q_text[:-1]
                q_words = q_text_without_question_mark.split(' ')
                q_corpus.update(q_words)
                a_text = q_obj['answer'].lower()
                a_text = re.sub('\s+', ' ', a_text)
                a_corpus.add(a_text)
                qa_list[mode].append((img_dir, q_words, a_text))

        word_to_idx = {"PAD":0, "SOS": 1, "EOS": 2}
        idx_to_word = {0: "PAD", 1: "SOS", 2: "EOS"}
        answer_word_to_idx = dict()
        answer_idx_to_word = dict()
        for idx, word in enumerate(q_corpus, start=3):
            # index starts with 1 because 0 is used as the padded value when batches are
            #  created
            word_to_idx[word] = idx
            idx_to_word[idx] = word

        for idx, word in enumerate(a_corpus):
            answer_word_to_idx[word] = idx
            answer_idx_to_word[idx] = word
        #     # single answer, so no padded values of 0 are created. thus index starts with 0
        data_dict = {'question': {'word_to_idx' : word_to_idx,
                                    'idx_to_word' : idx_to_word},
                        'answer': {'word_to_idx' : answer_word_to_idx,
                                    'idx_to_word' : answer_idx_to_word}}
        with open(self.root_dir + 'data_dict.pkl', 'wb') as file:
            pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
        print('data_dict.pkl saved')

        qa_idx_data = defaultdict(list)
        for mode in modes:
            for img_dir, q_word_list, answer_word in qa_list[mode]:                  
                q = [word_to_idx[word] for word in q_word_list]
                q.insert(0, 1)
                q.append(2)
                q = torch.from_numpy(np.array(q))
                a = answer_word_to_idx[answer_word]
                a = torch.from_numpy(np.array(a)).view(1)
                qa_idx_data[mode].append((img_dir, q, a))
            with open(self.root_dir + 'qa_idx_data_{}.pkl'.format(mode), 'wb') as file:
                pickle.dump(qa_idx_data[mode], file, protocol=pickle.HIGHEST_PROTOCOL)
            print('qa_idx_data_{}.pkl saved'.format(mode))

    def load_data(self):
        with open(self.root_dir + '{}/qa_idx_data_{}.pkl'.format(self.mode, self.mode), 'rb') as file:
            self.qa_idx_data = pickle.load(file)
        with open(self.root_dir + '{}/data_dict.pkl'.format(self.mode), 'rb') as file:
            self.data_dict = pickle.load(file)
        self.word_to_idx = self.data_dict['question']['word_to_idx']
        self.idx_to_word = self.data_dict['question']['idx_to_word']
        self.answer_word_to_idx = self.data_dict['answer']['word_to_idx']
        self.answer_idx_to_word = self.data_dict['answer']['idx_to_word']
        self.q_size = len(self.word_to_idx)
        self.a_size = len(self.answer_word_to_idx)

    def __len__(self):
        return len(self.qa_idx_data)

    def __getitem__(self, idx):
        img_dir, q, a = self.qa_idx_data[idx]
        image = Image.open(self.img_dir + img_dir).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class SortOfClevr(Dataset):
    """SortOfClevr dataset."""
    def __init__(self, root_dir, train = True, transform = None):
        self.root_dir = root_dir
        self.mode = 'train' if train else 'val'
        self.transform = transform
        self.data_dir = self.root_dir + '{}/data.hy'.format(self.mode)
        self.load_data()

    def load_data(self):
        file = h5py.File(self.data_dir, 'r')
        data = []
        for key, val in file.items():
            image = val['image'].value
            image.astype(float)
            image = torch.from_numpy(image.transpose(2,0,1)).to(torch.float)
            question = np.where(val['question'].value)[0]
            question[1] = question[1] - vqa_util.NUM_COLOR
            question = torch.Tensor(question).to(torch.long)
            answer = np.where(val['answer'].value)[0]
            answer = torch.Tensor(answer).to(torch.long)
            data.append((image, question, answer))
        self.data_list = data
        self.idx_to_question = vqa_util.question_type_dict
        self.idx_to_color = vqa_util.color_dict
        self.idx_to_answer = vqa_util.answer_dict
        self.q_size = len(self.idx_to_question)
        self.c_size = len(self.idx_to_color)
        self.a_size = len(self.idx_to_answer)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image, q, a = self.data_list[idx]
        return image, q, a

# def collate_text(list_inputs):
#     list_inputs.sort(key=lambda x:len(x[1]), reverse = True)
#     images = torch.Tensor()
#     questions = []
#     answers = torch.Tensor().to(torch.long)
#     for i, q, a in list_inputs:
#         images = torch.cat([images, i.unsqueeze(0)], 0)
#         questions.append(q)
#         answers = torch.cat([answers, a], 0)
#     questions_packed = pack_sequence(questions)
#     return images, questions_packed, answers