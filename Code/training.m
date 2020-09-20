clc

xtrain = cat(3,real(K1),real(K2));
xtrain = reshape(abs(xtrain),[52 52 1 1800]);
ytrain = categorical(cat(1,zeros(900,1),ones(900,1)));
xvalid = cat(3,real(Kv1),real(Kv2));
xvalid = reshape(abs(xvalid),[52 52 1 300]);
yvalid = categorical(cat(1,zeros(150,1),ones(150,1)));

layers = [
    imageInputLayer([52 52 1])
    
    convolution2dLayer(5,5,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2)
    
    convolution2dLayer(5,5,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2)
    
    convolution2dLayer(5,5,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer]


options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.05, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{xvalid,yvalid}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');




net = trainNetwork(xtrain, ytrain,layers,options);
