
function [transmission] = get_transmission(S_Vr,number_threshold,length_threshold)
[m,n,~] = size(S_Vr);
transmission = zeros(m,n);

number_pixels = m*n;
y = reshape(S_Vr(:,:,1),1,number_pixels);
x = reshape(S_Vr(:,:,2),1,number_pixels);

pixel_feature =zeros(1,number_pixels);
for i = 1:size(pixel_feature,2)
    k_pixel = (y(i)-y)./(x(i)-x);
    pixel_feature(i) = sum((k_pixel>-1)&(k_pixel<0));
end

selection_feature = (pixel_feature>=0.5*number_pixels);

num_selected = sum(selection_feature);
x_selected = x(selection_feature);
y_selected = y(selection_feature);

if  num_selected>number_threshold
    x_mean = mean(x_selected);
    y_mean = mean(y_selected);
    k = (sum(x_selected.*y_selected)-num_selected*x_mean*y_mean)/(sum(x_selected.^2)-num_selected*x_mean.^2);
    b = y_mean-k*x_mean;
    t = 1+k/b;
    length = ((1+k*k).^0.5)*(max(x_selected)-min(x_selected));

    if k>-1&&k<0 && length>length_threshold
        transmission = (transmission+1)*t;
    end

end
end

