function [A] = get_atmospheric_light(image,patch_size)
if ~exist('patch_size', 'var')
    patch_size= 55; % larger patch size to filter out the local white regions
end
dark_channel=minfilt2(min(image,[],3),patch_size);
R = image(:,:,1);
G = image(:,:,2);
B = image(:,:,3);
[m,n]=size(dark_channel);
number = m * n;
dark_channel_sort=sort(reshape(dark_channel,1,[]),'descend');
threshold=dark_channel_sort(floor(number*0.001));
index=dark_channel>=threshold;
A=ones(1,3);
A(1) = median(R(index));
A(2) = median(G(index));
A(3) = median(B(index));
end

