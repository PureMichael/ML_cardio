function [imdsTest, trainedNet,state] = makeCNN()
%MAKECNN Summary of this function goes here



%clear
load('fissiondata_labeled.mat');
load('fissiondata_labeled_resized.mat')

%% Coded to set the yes and no to equal size. (2485 of each)

tbl = countEachLabel(imds_resized);
minSetCount = min(tbl{:,2});
imds = splitEachLabel(imds_resized,minSetCount,'randomize');
countEachLabel(imds);

%% Make sure labels are correct
%fyes = find(imds.Labels== 'yes',1);
%fno  = find(imds.Labels=='no',1);
%figure
%subplot(1,2,1);
%imshow(readimage(imds,fyes));
%subplot(1,2,2);
%imshow(readimage(imds,fno));

%% Get size of images   (2485*2 = 4970)
numImages = size(imds.Labels,1);
% xlen = zeros(1,numImages);
% ylen = zeros(1,numImages);
% for i = 1:numImages
%     img = readimage(imds,i);
%     [xlen(i),ylen(i)] = size(img); 
% end

%tic
%% Prepare Training and Testing data
numTrainFiles = round(numImages*0.5*0.75);
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainFiles,'randomize');

%% Setup Layers and Training Options
layers = [
    imageInputLayer([176 176 1])
    
    convolution2dLayer(3,8,'Padding','same')  %3 = filter size, 8 = no of Filters
    batchNormalizationLayer
    reluLayer                                 %Rectifying layer. Removes negatives.
    
    maxPooling2dLayer(2,'Stride',2)           %Downsampling to remove redundant data
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',3)  %changed from 2 by 2
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2) %Number of output categories
    softmaxLayer
    classificationLayer];
   
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',1, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','none');


  
%% Train the network
%resizeData = imresize(imdsTrain.read(),[176 176])
[trainedNet, trainedinfo] = trainNetwork(imds,layers,options);

%toc
%trainedinfo.ValidationAccuracy(152);


end

