import os
import argparse
from model import train,test
from data import data_builder
parser=argparse.ArgumentParser()
parser.add_argument("--train_path",help="Path to the training dataset",default='./data')
parser.add_argument("--val_path",help="Path to the validation dataset",default='None')
parser.add_argument("--lr", help="Learning Rate",type = float, default = 0.001)
parser.add_argument("--batch_size", help="Batch size for training dataset",type = int, default = 4)
parser.add_argument("--mode",help="Choose between train or test mode.", default='train',choices=['train','test'])
parser.add_argument("--path_to_model",help="Path to store trained model. After training model will be stored in checkpoints directory if value not set",default='./checkpoints')
parser.add_argument("--path_to_trained_model",help="Path to load trained model for testing",default='./checkpoints/ASl_Classifier')
parser.add_argument("--epochs",help="Number of epochs to train the model",type=int)
parser.add_argument("--val_split",help="Validation data split ratio. Only to be specified when val_path is None",type=float, default=0.2)
                    
if __name__=='__main__':    
    args=parser.parse_args()
    if args.mode=='train':
        train(args)
    if args.mode=='test':
        test(args)
