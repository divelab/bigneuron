import numpy as np
import re
import itertools
from collections import Counter
import csv
import scipy
import time
import operator
import h5py 


class data_reader:
    def __init__(self):
        print("Now loading data ==================")
        self.path_train= '/tempspace/hyuan/big_neuron/data/training.txt'
        self.path_test= '/tempspace/hyuan/big_neuron/validate_data/test_1.txt'
        self.height = 160
        self.width = 160
        self.depth = 8
        self.training_files= self.read_data_names(self.path_train)
        self.test_files = self.read_data_names(self.path_test)
        self.training_size = len(self.training_files)
        self.test_size = len(self.test_files)
    #    np.random.shuffle(self.training_files)
        self.train_idx = 0 
        self.test_idx = 0

    def read_data_names(self, path):
        files = []
        with open(path, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                files.append(line)
        return files
        
    def read_random_crop_image(self, path):
    #    a= time.time()
        data = h5py.File(path, 'r')
        img = data['data']
        label = data['label']
        #print('the time diff is ', b-a)
        shape = img.shape
        x  = np.random.randint(low=0, high= shape[0]-self.height+1)
        y  = np.random.randint(low= 0, high = shape[1]-self.width+1) 
        z  = np.random.randint(low= 0, high = shape[2]- self.depth+1)
        mid_x = img[x:x+self.height,y:y+self.width,:]
        mid_y = label[x:x+self.height,y:y+self.width,:]

        res_x = mid_x[:,:,z:z+self.depth]
        res_y = mid_y[:,:,z:z+self.depth]
        return res_x, res_y
       

    def read_whole_img(self, path):
        data = h5py.File(path, 'r')
        img = data['data']
        label = data['label']

        name = path.split('/')[-1]
        return img, label, name


    def gen_index(self):
        self.indexes = np.random.permutation(range(len(self.training_files)))
        self.train_idx = 0


    def next_batch(self, batch_size):
        start = time.time()
        imgs = []
        labels = []
        next_index = self.train_idx + batch_size
        if next_index>= self.training_size:
            np.random.shuffle(self.training_files)
            self.train_idx = 0 
            next_index = self.train_idx + batch_size
        for i in range(self.train_idx, next_index):
      #      r_s =  time.time()
            path = self.training_files[i]
            img, label = self.read_random_crop_image(path)
            imgs.append(img)
            labels.append(label)
        imgs_return = np.array(imgs)
        labels_return = np.array(labels)
        _, count = np.unique(labels_return, return_counts=True)    
        count_total = 0  
        for i in range(count.shape[0]):
            count_total = count_total +count[i]

        count_0 = count_total/ count[0]
        if count.shape[0]==3:
            count_1 = count_total/ count[1]
        else:
            count_1 = 0
        if count.shape[0]==3:
            count_2 = count_total/ count[2]
        else:
            count_2 = count_total/ count[1]
        count_return= np.array([count_0, count_1, count_2])
   #     print('The counts are,', count, 'the total is', count_total, 'the return ratio is ', count_return)
        self.train_idx = next_index
        return imgs_return, labels_return, count_return

    def get_padded_images(self, img, label):

        x =img.shape[0]
        y =img.shape[1]
        z =img.shape[2]
        print(x,' ', y, ' ' ,z)
        padding_x=0
        padding_y=0
        padding_z=0
        if x%8 !=0:
            padding_x =  8-x%8
        if y%8 !=0:
            padding_y = 8- y%8
        if z%8 !=0:
            padding_z = 8- z%8
        if padding_x!=0:
            x_padding = np.ones((padding_x, y , z))
            img = np.concatenate((img, x_padding*0),axis=0)
            label = np.concatenate((label, x_padding*2), axis=0)
            print(img.shape)
        if padding_y!=0:            
            y_padding = np.ones((x+padding_x, padding_y, z))
            img = np.concatenate((img, y_padding*0),axis=1)
            label = np.concatenate((label, y_padding*2), axis=1)
            print(img.shape)
        if padding_z!=0:
            z_padding = np.ones((x+padding_x, y+padding_y, padding_z))
            img = np.concatenate((img, z_padding*0),axis=2)
            label = np.concatenate((label, z_padding*2), axis=2)
            print(img.shape)
        return img, label,x,y,z      

    
    def get_next_test(self):
        images = []
        labels = []
        path = self.test_files[self.test_idx]        
        img, label, name = self.read_whole_img(path)
        image_new, label_new, x, y, z = self.get_padded_images(img, label)
        images.append(image_new)
        labels.append(label_new)    
        imgs_return = np.array(images)
        labels_return = (np.array(labels)).astype(np.int64)
     #   imgs_return = np.transpose(imgs_return, axes=(3,0,1,2))
     #   labels_return = np.transpose(labels_return, axes=(3,0,1,2))
        # imgs_return = imgs_return[:,::]
        # labels_return = labels_return[:::]
        self.test_idx = self.test_idx+1
        return imgs_return, labels_return, name, x, y ,z        

    def reset(self):
        self.test_idx = 0

    def get_random_test(self):
        imgs = []
        labels = []
        for i in range(0, 2):
      #      r_s =  time.time()
            idx = np.random.randint(0,self.test_size)
            path = self.test_files[idx]
            img, label = self.read_random_crop_image(path)
            imgs.append(img)
            labels.append(label)
        imgs_return = np.array(imgs)
        labels_return = np.array(labels)
        _, count = np.unique(labels_return, return_counts=True) 
       # _, count = np.unique(labels_return, return_counts=True)
        count_total = 0  
        for i in range(count.shape[0]):
            count_total = count_total+count[i]

        count_0 = count_total/ count[0]
        if count.shape[0]==3:
            count_1 = count_total/ count[1]
        else:
            count_1 = 0
        if count.shape[0]==3:
            count_2 = count_total/ count[2]
        else:
            count_2 = count_total/ count[1]
        count_return= np.array([count_0, count_1, count_2])
    #    print('The counts are,', count, 'the total is', count_total, 'the return ratio is ', count_return)
        return imgs_return, labels_return, count_return

    def get_patches(self, id, overlap_size):
        imgs = []
        labels = []
        path = self.test_files[id]
        print(path)
        img, label, name = self.read_whole_img(path)
        shape = img.shape
        print('the shape is ', shape)
        patch_ids = self.prepare_validation(shape, overlap_size)
        print('The number of patches is ', len(patch_ids))

        return img, label, name, patch_ids, shape
    
    def get_patches_from_img(self,img,label,ids):
        imgs = []
        labels = []
        patch = ids 
        x, y, z = patch
        img_patch = img[x:x+self.height, y:y+self.width,:]
        img_patch = img_patch[:,:,z:z+self.depth]
        label_patch = label[x:x+self.height, y:y+self.width,:]
        label_patch = label_patch[:,:,z:z+self.depth]            
        imgs.append(img_patch)
        labels.append(label_patch)
         #   print(len(imgs))
        imgs_return = np.array(imgs)
        labels_return = np.array(labels)        
        return imgs_return, labels_return




    def prepare_validation(self, shape, overlap_stepsize):
        """Determine patches for validation."""
        patch_ids = []
        x, y, z = shape

        x_range = list(range(0, x-self.height+1, overlap_stepsize[0]))
        y_range = list(range(0, y-self.width+1, overlap_stepsize[1]))
        z_range = list(range(0, z-self.depth+1, overlap_stepsize[2]))

        if (x-self.height) % overlap_stepsize[0] != 0:
            x_range.append(x-self.height)
        if (y-self.width) % overlap_stepsize[1] != 0:
            y_range.append(y-self.width)
        if (z-self.depth) % overlap_stepsize[2] != 0:
            z_range.append(z-self.depth)

        for d in x_range:
            for h in y_range:
                for w in z_range:
                    patch_ids.append((d, h, w))
        return patch_ids
