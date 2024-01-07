import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import keras
import tensorflow as tf


def load_images(path,categories):
    data_list = list()
    labels=list()          
    for label in  sorted(os.listdir(path)):  
        y=os.path.join(path,label)
        for filename in os.listdir(y):
            pixels = cv2.imread(os.path.join(y,filename))
            pixels=cv2.resize(pixels,(299,299))
            data_list.append(pixels)
            labels.append(categories[label])
    data_list=np.array(data_list,dtype='float32')/255.0
    labels=keras.utils.to_categorical(labels, 36)    
    dataset=tf.data.Dataset.from_tensor_slices(data_list,labels)
    dataset=dataset.shuffle(shuffle_size=1000)
    return dataset

def get_dataset_partitions(ds, ds_size,val_split=0.2, shuffle=True, shuffle_size=1000):
    train_split = 1- val_split 
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    
    return train_ds, val_ds

def data_builder(args):
    train_path=args.train_path
    categories = {
        "0":0,
        "1":1,
        "2":2,
        "3":3,
        "4":4,
        "5":5,
        "6":6,
        "7":7,
        "8":8,
        "9":9,
        "a":10,
        "b":11,
        "c":12,
        "d":13,
        "e":14,
        "f":15,
        "g":16,
        "h":17,
        "i":18,
        "j":19,
        "k":20,
        "l":21,
        "m":22,
        "n":23,
        "o":24,
        "p":25,
        "q":26,
        "r":27,
        "s":28,
        "t":29,
        "u":30,
        "v":31 ,
        "w":32,
        "x":33,
        "y":34,
        "z":35,
    }
    train_dataset=load_images(train_path,categories)
    size_=train_dataset.cardinality().numpy()
    if args.mode=='test':
        train_dataset = train_dataset.prefetch(buffer_size= tf.data.AUTOTUNE)
        train_dataset.batch(1)
        return train_dataset
    if args.val_path=='None':
        train_dataset,val_dataset=get_dataset_partitions(ds=train_dataset,ds_size=size_,val_split=args.val_split)
    else:
        val_dataset=load_images(args.val_path,categories)    

    train_dataset = train_dataset.prefetch(buffer_size= tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size= tf.data.AUTOTUNE)
    train_dataset.batch(args.batch_size)
    return train_dataset,val_dataset
    