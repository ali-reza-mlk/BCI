clc

a=34; %%%% cropping matrix
b=65;
c=1;
d=32;
xtrain = q(a:b,c:d,1,:);
xval = qv(a:b,c:d,1,:);

ytrain = categorical(cat(1,zeros(900,1),ones(900,1)));

yval = categorical(cat(1,zeros(150,1),ones(150,1)));

layers = [
    imageInputLayer([b-a+1  b-a+1 1],'Name','input')
        
    convolution2dLayer(4,10,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    maxPooling2dLayer(4,'stride',4,'Name','maxpool_1')
    
    convolution2dLayer(3,10,'Padding','same','Name','conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(4,'stride',4,'Name','maxpool_2')
    
    convolution2dLayer(3,5,'Padding','same','Name','conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','relu_3')
    
    globalAveragePooling2dLayer('Name','globpool_1')
    fullyConnectedLayer(2,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classOutput')]



options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.1, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.9, ...
    'LearnRateDropPeriod',5, ...
    'ValidationData',{xval,yval}, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',20, ...
    'Verbose',true, ...
    'Plots','training-progress');

net = trainNetwork(xtrain, ytrain,layers,options);
