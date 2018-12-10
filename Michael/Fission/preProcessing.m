load('fissiondata_labeled.mat');




imds2.ReadSize = numpartitions(imds2);
imds2.ReadFcn = @(loc)imresize(imread(loc),[176,176]);


%for m = 100:5000
%    imshow(readimage(imds2,m))
%end


numImages = size(imds2.Labels,1)
 xlen = zeros(1,numImages);
 ylen = zeros(1,numImages);
 for i = 1:numImages
     img = readimage(imds2,i);
     [xlen(i),ylen(i)] = size(img);
     %if xlen(i)>=176
         %imresize(imdsTrain.read(),[176 176])
     %end
 end
