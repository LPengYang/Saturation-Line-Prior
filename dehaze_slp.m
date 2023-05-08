function [dehaze] = dehaze_slp(I,patch_size,A_map)
if ~exist('patch_size', 'var')
    patch_size = 15;
end
II=double(I)./255;
[m,n,~]=size(II);
number_threshold = 10;
length_threshold = 0.1;
b0=20/255;
b1=300/255;

if ~exist('A_map', 'var')
    A=get_atmospheric_light(II);
    A_map=reshape(repmat(A,[m*n,1]),m,n,3);
end

t_boundary=max(max((II-A_map)./(b1-A_map+eps),(II-A_map)./(b0-A_map)),[],3);
t_boundary =maxfilt2(t_boundary,patch_size);

HSV=rgb2hsv(II./A_map);
S_Vr=HSV(:,:,2:3);
S_Vr(:,:,2)=1./(S_Vr(:,:,2)+eps);
fun = @(block_struct)get_transmission(block_struct.data,number_threshold,length_threshold);

t_crop_1=blockproc(S_Vr,[patch_size,patch_size],fun);
patch_mid=floor(patch_size/2);
t_crop_2=blockproc(S_Vr(patch_mid:end,patch_mid:end,:),[patch_size,patch_size],fun);
t_crop_2=[zeros(patch_mid-1,n);[zeros(m-patch_mid+1,patch_mid-1),t_crop_2]];

number=(t_crop_1>0)+(t_crop_2>0);
t_mean=(t_crop_1+t_crop_2)./number;

t_f = zeros(m,n);
index_if_success = number>0;
t_f(index_if_success) = t_mean(index_if_success);
t_f(~index_if_success) =t_boundary(~index_if_success);

t_sl =reshape(t_mean(index_if_success),1,[]);
t_l= prctile(t_sl,5);
t_min= mean(t_sl(t_sl<t_l));
t_r = min(max(t_f,t_min),1);

t_final = guidedfilter(min(II,[],3),t_r,2*patch_size,0.01);
t_final = repmat(t_final,1,1,3);
dehaze = (II-A_map)./t_final+A_map;
dehaze = min(max(dehaze,0),1);
end