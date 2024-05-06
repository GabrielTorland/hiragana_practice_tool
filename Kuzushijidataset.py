import requests
import os
import numpy as np

from PIL import Image

class Kuzushijidataset():
    """
    class credz to https://github.com/elisiojsj/Kuzushiji-49/blob/master/Kuzushiji-49.ipynb
    """
    
    resources = [
    ("http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz"),
    ("http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz"),
    ("http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz"),
    ("http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz")]
    
    training_file_imgs = "k49-train-imgs.npz"
    training_file_labels = "k49-train-labels.npz"
    test_file_imgs = "k49-test-imgs.npz"
    test_file_labels = "k49-test-labels.npz"
    data_dir = "k49-dataset"
    
   
    def __init__(self, data_dir="k49-dataset", train=True, transform=None, download=True):
        self.data_dir = data_dir
        
        if download:
            self.download(train)
            
        train_data_imgs = os.path.join(self.data_dir, self.training_file_imgs)
        train_data_imgs = np.load(train_data_imgs)
        train_data_imgs = train_data_imgs.f.arr_0

        train_data_labels = os.path.join(self.data_dir, self.training_file_labels)
        train_data_labels = np.load(train_data_labels)
        train_data_labels = train_data_labels.f.arr_0

        test_data_imgs = os.path.join(self.data_dir, self.test_file_imgs)
        test_data_imgs = np.load(test_data_imgs)
        test_data_imgs = test_data_imgs.f.arr_0
        
        test_data_labels = os.path.join(self.data_dir, self.test_file_labels)
        test_data_labels = np.load(test_data_labels)
        test_data_labels = test_data_labels.f.arr_0
        
        self.transform = transform
        
        if train:
            self.data = train_data_imgs
            self.targets = train_data_labels
        else:
            self.data = test_data_imgs
            self.targets = test_data_labels


    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L') # mode='L' - (8-bit pixels, black and white)

        if self.transform:
            img = self.transform(img)
        
        return img, target        
      
    def download(self, train):
        # download the Kuzushiji-49 dataset if it doesn't exist
        if self._check_exists():
            if train:
                print('Train dataset already exists!')
            else:
                print('Test dataset already exists!')
            return

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        for url in self.resources:
            filename = url.rpartition('/')[2]
            print('Downloading: ', filename)
            myfile = requests.get(url, allow_redirects=True)
            open(os.path.join(self.data_dir, filename), 'wb').write(myfile.content)

        print('All files downloaded!')
        

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.data_dir, self.training_file_imgs)) and
                os.path.exists(os.path.join(self.data_dir, self.training_file_labels)) and
                os.path.exists(os.path.join(self.data_dir, self.test_file_imgs)) and
                os.path.exists(os.path.join(self.data_dir, self.test_file_labels)))
