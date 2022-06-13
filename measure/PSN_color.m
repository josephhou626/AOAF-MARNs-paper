function [out]=PSN_color(orgimg,mimg)
% PSN is PSNR of Image Peak signal to noise ratio.
% orgimg    = Orignal Image
% mimg      = Modified Image
% Size of two images must be same.

% Code Developed BY : Suraj Kamya
% kamyasuraj@yahoo.com
% Visit to: kamyasuraj.blogspot.com

orgimg =im2double(orgimg);
mimg   =im2double(mimg);

orgimg_red=orgimg(:,:,1);
orgimg_green=orgimg(:,:,2);
orgimg_blue=orgimg(:,:,3);

mimg_red=mimg(:,:,1);
mimg_green=mimg(:,:,2);
mimg_blue=mimg(:,:,3);


Mse_red=sum(sum((orgimg_red-mimg_red).^2))/(numel(orgimg_red));
Mse_green=sum(sum((orgimg_green-mimg_green).^2))/(numel(orgimg_green));
Mse_blue=sum(sum((orgimg_blue-mimg_blue).^2))/(numel(orgimg_blue));

Mse=(Mse_red+Mse_green+Mse_blue)/3;

%Mse=sum(sum((orgimg-mimg).^2))/(numel(orgimg)); %Mse = Mean square Error


out=10*log10(1/Mse);