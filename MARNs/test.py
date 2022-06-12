import torch
from myutils.utils import*
from dataset.denoise_dataset import*



def test(opt,model):
    model.eval()
    model.to(opt.device)
    testx = readfile(opt.test_path)
    test_set = imgdataset(testx ,None,train_transform)
    test_loader = DataLoader(test_set,batch_size =1,shuffle = False)

    print(f"[Info]: Start testing data!",flush = True)

    ##暫時放結果的array
    result_array = np.ones([1000,1,512,512] , dtype= 'uint8')
    with torch.no_grad() :
        for i , data in enumerate(test_loader) :
            test_pred = model(data.cuda())
            out_np = test_pred.detach().cpu().numpy()*255
            result_array[i] = out_np

    #####存圖
    result_array = np.transpose(result_array,(0,2,3,1)) 

    print("[Info] : save images")
    savefile(result_array,opt.opt.save_results,opt.test_path)