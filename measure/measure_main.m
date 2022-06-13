clear all;
clc ;


%路徑
title_Isrc = './MARNs/gt/';
title_Iout = './MARNs/output/';
addpath(title_Isrc); 
addpath(title_Iout);

%批量讀取數據(input)
dpath_input = [dir([title_Isrc '*' 'bmp']) ; dir([title_Isrc '*' 'jpg']) ; dir([title_Isrc '*' 'png']) ; ]; 
dpath_gt = [dir([title_Iout '*' 'bmp']) ; dir([title_Iout '*' 'jpg']) ; dir([title_Iout '*' 'png']) ; ]; 


%設定excel 格式
xlsFile = 'psnr_ssim_niqe.xls';
sheetName='PSNR & SSIM for score';
data={'PIC','PSNR','','SSIM','','NIQE'};



for i = 1 : 1 : length(dpath_input)
   
   img_input = sprintf('%s%s', title_Isrc, dpath_input(i).name); 
   img_gt = sprintf('%s%s', title_Iout, dpath_gt(i).name); 
   
   
   input_img=im2double(imread(img_input));   
   gt_img=im2double(imread(img_gt));  
   
   
   data{i+2,1}=i;
   
   psn=PSN_color(input_img,gt_img); 
      
   data{i+2,2}=psn;
   

   input_img=imread(img_input);   
   gt_img=imread(img_gt);  
       
   [ssimval, ssimmap] = ssim_color(input_img,gt_img); 
   data{i+2,4}=ssimval;
      

   ni = niqe(input_img);
    data{i+2,6}=ni;
   
 
	
end

[status, message] = xlswrite(xlsFile, data, sheetName);
dos(['start ' xlsFile]);