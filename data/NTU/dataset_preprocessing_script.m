% NAME: MATLAB code for dataset rgb-depth calibration preprocessing (obsolete)
% AUTHOR: Zhou Ren, 06/2012
% FUNCTION: because this dataset was captured by a Kinect v1, back then the rgb color image and the corresponding depth map 
% were not perfectly calibrated (mismatches observed). This script is to calibrate the color image and depth pair, using 
% a simple cropping based preprocessing method.

% LOOP through each pair (here we only give the inner-loop processing code)
dir_If = [dir_Ik num2str(k) '.jpg']; % directory of a certain rgb image
dir_Df = [dir_Dk num2str(k) '.txt']; % the corresponding depth map

% Depth_Cur is the current frame depth data, which will be unaltered.
Depth_Cur = load(dir_Df); 

% we modify the color image to make it calibrate with the depth map.
Image_Cur=[]; 
R_Cur=[]; G_Cur=[]; B_Cur=[];
I_ori = imread(dir_If, 'jpg'); % I_ori is the original image information
RR=I_ori(:,:,1); GG=I_ori(:,:,2); BB=I_ori(:,:,3);
R_Cur(:,:) = RR(24:478,51:634);
G_Cur(:,:) = GG(24:478,51:634);
B_Cur(:,:) = BB(24:478,51:634); % heuristic cropping based calibration
Image_Cur(:,:,1)=R_Cur; Image_Cur(:,:,2)=G_Cur; Image_Cur(:,:,3)=B_Cur;
Image_Cur = imresize(Image_Cur,[480,640]); % after image resizing, Image_Cur is now calibrated with Depth_Cur.

% Now, Image_Cur and Depth_Cur are the calibrated image-depth pair.
% preprocessing done. :)