clc ; 
clear ; 
  
imgRoot = './input_noise_image/';
outRoot = './AOAF_result/';
addpath(imgRoot); 
addpath(outRoot);
dpath = [dir([imgRoot '*' 'bmp']) ; dir([imgRoot '*' 'jpg']) ; dir([imgRoot '*' 'png']) ; ]; %批量讀取數據(input)
dpath_2 = [dir([outRoot '*' 'bmp']) ; dir([outRoot '*' 'jpg']) ; dir([outRoot '*' 'png']) ; ]; %批量讀取數據(output)
for i = 1 : 1 : length( dpath )
    %tic;
    cd('./test') ;  % 進入放圖的地方
    input_img  = imread(dpath(i).name );  % 讀圖的name 
 
    cd('../') ;  % 往上一個資料夾
    TF = contains(dpath(i).name,'irz') ;
    if TF == 0
        
     %RGB分開來denoise
     r = AOAF( input_img( : , : , 1 )) ;  
     g = AOAF( input_img( : , : , 2 )) ; 
     b = AOAF( input_img( : , : , 3 )) ; 
     
     
     %concatenate起來(R,G,B)
     output_img(: , : , 1) = r ;  output_img(: , : , 2) = g ;  output_img(: , : , 3) = b ; 
     %存圖
     imwrite(output_img , sprintf('%s%s', outRoot, dpath(i).name ));  % sprintf()是用來放的位址。
    end
    
    %time=toc;
    %disp(time) ;
    
end



