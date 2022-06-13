import torch
import torch.nn as nn
import argparse
from myutils.utils import*
from dataset.denoise_dataset import*
from tqdm import tqdm


def train(opt,model):

    ## 設定 model 
    model.train()
    model.to(opt.device)
    
    ##設定dataset和dataloader
    trainx = readfile(opt.train_path)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainy = readgt(opt.train_gt)
    train_set = imgdataset(trainx,trainy,train_transform)
    train_loader = DataLoader(train_set,batch_size = opt.batch_size, shuffle = True)

    ##實驗設定
    loss = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 
    num_epoch = opt.max_epoch

    pbar = tqdm(total= num_epoch, ncols=0, desc="Train", unit=" step")

    print(f"[Info]: Start training data!",flush = True)
    for epoch  in range(num_epoch) :
        tot_s_loss = []
        for i , (input , gt) in enumerate(train_loader): 
            optimizer.zero_grad()

            input_image = input.to(opt.device)
            gt_image = gt.to(opt.device)
            output_image = model(input_image)
            batch_loss = loss(output_image,gt_image)
            batch_loss.backward()
            optimizer.step()
            tot_s_loss.append(batch_loss.item())

        mean_s_loss = np.mean(tot_s_loss)
        pbar.update() 
        pbar.set_postfix({'training loss' : mean_s_loss  ,'epoch':epoch+1})

        if (epoch+1) % 1 == 0 or (epoch+1) == opt.max_epoch:
                torch.save(model.state_dict(), os.path.join(opt.save_model_path,rf'model_{epoch}'))
                print('save !')
    
    torch.save(model.state_dict(), os.path.join(opt.save_model_path,rf'model_last'))
    pbar.close()