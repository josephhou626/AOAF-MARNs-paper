import os
import argparse
from test import *
from train import *
from model.MARNs import *

## 設定parser
parser = argparse.ArgumentParser()
parser.add_argument('--mode',default='train' ,type=str, required=True)
parser.add_argument('--train_path',default='AOAF_results/train/input' ,type=str, required=True)
parser.add_argument('--train_gt', default='AOAF_results/train/gt' ,type=str,required=True)
parser.add_argument('--test_path', default='AOAF_results/test/input',type=str,required=True)
parser.add_argument('--max_epoch', default=80,type=int,required=True)
parser.add_argument('--batch_szie', default=8,type=int,required=True)
parser.add_argument('--device',help='cpu or gpu')
parser.add_argument('--load_model_path', default=None,type=str,required=True)
parser.add_argument('--save_model_path', default="runs/MARNs",type=str,required=True)
parser.add_argument('--save_results', default="runs/MARNs/output",type=str,required=True)
opt = parser.parse_args()


if __name__ == "__main__":

    # device是選擇CPU還是GPU
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {opt.device} now!")

    # 設定model
    model = MARNs()
    print ('[Info]: Number of params: %d' % count_parameters( model ))


    # 開始train 或是 test
    if opt.mode == 'train':
        if opt.load_model_path != None :
            pth = torch.load(opt.load_model_path) 
            model.load_state_dict(pth) 

        auto_create_path(opt.save_model_path)
        train(opt,model)
    else:
        pth = torch.load(opt.load_model_path) 
        model.load_state_dict(pth) 

        auto_create_path(opt.save_results)
        test(opt,model)



    