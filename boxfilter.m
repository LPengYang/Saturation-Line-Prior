%function imDst = boxfilter(imSrc, r)
function imDst = boxfilter(imSrc, r)

%   BOXFILTER   O(1) time box filtering using cumulative sum
%
%   - Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
%   - Running time independent of r; 
%   - Equivalent to the function: colfilt(imSrc, [2*r+1, 2*r+1], 'sliding', @sum);
%   - But much faster.

% [hei, wid] = size(imSrc);
% imDst = zeros(size(imSrc));
% 
% %cumulative sum over Y axis
% imCum = cumsum(imSrc, 1);
% %difference over Y axis
% imDst(1:r+1, :) = imCum(1+r:2*r+1, :);
% imDst(r+2:hei-r, :) = imCum(2*r+2:hei, :) - imCum(1:hei-2*r-1, :);
% imDst(hei-r+1:hei, :) = repmat(imCum(hei, :), [r, 1]) - imCum(hei-2*r:hei-r-1, :);
% 
% %cumulative sum over X axis
% imCum = cumsum(imDst, 2);
% %difference over Y axis
% imDst(:, 1:r+1) = imCum(:, 1+r:2*r+1);
% imDst(:, r+2:wid-r) = imCum(:, 2*r+2:wid) - imCum(:, 1:wid-2*r-1);
% imDst(:, wid-r+1:wid) = repmat(imCum(:, wid), [1, r]) - imCum(:, wid-2*r:wid-r-1);

% %% the following implementation is for C langugage
% [height, width] = size(imSrc);
% Sum = zeros(size(imSrc));
% imDst = zeros(size(imSrc));
% rho = r;
% %%%%Sum along Y direction
% 
% for j=1:width
%     Sum(1,j) = imSrc(1,j);
% end
% for i=2:height
%     for j=1:width
%         Sum(i,j) = Sum(i-1,j)+imSrc(i,j);
%     end
% end
% for i=1:rho+1
%     for j=1:width
%         imDst(i,j) = Sum(i+rho,j);
%     end
% end
% for i=rho+2:height-rho
%     for j=1:width
%         imDst(i,j) = Sum(i+rho,j)-Sum(i-rho-1,j);
%     end
% end
% for i=height-rho+1:height
%     for j=1:width
%         imDst(i,j) = Sum(height,j)-Sum(i-rho-1, j);
%     end
% end
% 
% 
% 
% %%% Sum along X direction
% 
% for i=1:height
%     Sum(i,1) = imDst(i,1);
% end
% 
% for i=1:height
%     for j=2:width
%         Sum(i,j) = Sum(i,j-1)+imDst(i,j);
%     end
% end
% 
% for i=1:height
%     for j=1:rho+1
%         imDst(i,j) = Sum(i,j+rho);
%     end
% end
% 
% for i=1:height
%     for j=rho+2:width-rho
%         imDst(i,j) = Sum(i,j+rho)-Sum(i,j-rho-1);
%     end
% end
% 
% for i=1:height
%     for j=width-rho+1:width
%         imDst(i,j) = Sum(i,width)-Sum(i, j-rho-1);
%     end
% end


[height, width] = size(imSrc);
Sum = zeros(size(imSrc));
imTmp = zeros(size(imSrc));
imDst = zeros(size(imSrc));
rho = r;
%%%%Sum along Y direction
for j=1:width
%%%initialization    
    Sum(1,j) = imSrc(1,j);
    for i=2:rho
        Sum(i,j) = Sum(i-1,j)+imSrc(i,j);
    end
    for i=rho+1:2*rho+1
        Sum(i,j) = Sum(i-1,j)+imSrc(i,j);
        imTmp(i-rho,j) = Sum(i,j);
    end
    for i=2*rho+2:height
        Sum(i,j) = Sum(i-1,j)+imSrc(i,j)-imSrc(i-2*rho-1,j);
        imTmp(i-rho,j) = Sum(i,j);
    end
    for i=height-rho+1:height
        imTmp(i,j) = imTmp(i-1,j)-imSrc(i-rho-1,j);
    end
end

%%% Sum along X direction

for i=1:height
%%%initialization     
    Sum(i,1) = imTmp(i,1);
    for j=2:rho
        Sum(i,j) = Sum(i,j-1)+imTmp(i,j);
    end
    for j=rho+1:2*rho+1
        Sum(i,j) = Sum(i,j-1)+imTmp(i,j);
        imDst(i,j-rho) = Sum(i,j); 
    end
    for j=2*rho+2:width
        Sum(i,j) = Sum(i,j-1)+imTmp(i,j)-imTmp(i,j-2*rho-1);
        imDst(i,j-rho) = Sum(i,j); 
    end
    for j=width-rho+1:width
        imDst(i,j) = imDst(i,j-1)-imTmp(i,j-rho-1); 
    end
end

clear Sum;
clear imTmp;

end

