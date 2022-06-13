import os
import argparse
from test import *
from train import *
from model.MARNs import *

## 設定parser
parser = argparse.ArgumentParser()
parser.add_argument('--mode',default='train' ,type=str, required=False)
parser.add_argument('--train_path',default='AOAF_results/train/input' ,type=str, required=False)
parser.add_argument('--train_gt', default='AOAF_results/train/gt' ,type=str,required=False)
parser.add_argument('--test_path', default='AOAF_results/test/input',type=str,required=False)
parser.add_argument('--max_epoch', default=80,type=int,required=False)
parser.add_argument('--batch_size', default=8,type=int,required=False)
parser.add_argument('--device',help='cpu or gpu')
parser.add_argument('--resume', default=False,type=bool,required=False)
parser.add_argument('--load_model_path', default="exp/MARNs/model/model_19",type=str,required=False)
parser.add_argument('--save_model_path', default="exp/MARNs/model",type=str,required=False)
parser.add_argument('--save_results', default="exp/MARNs/output",type=str,required=False)
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
        if opt.resume == True :
            pth = torch.load(opt.load_model_path) 
            model.load_state_dict(pth) 

        auto_create_path(opt.save_model_path)
        train(opt,model)
    else:
        pth = torch.load(opt.load_model_path) 
        model.load_state_dict(pth) 

        auto_create_path(opt.save_results)
        test(opt,model)



    