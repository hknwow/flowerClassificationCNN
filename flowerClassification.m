% Getting images
flowerds             = imageDatastore('Flowers','IncludeSubfolders',true,'LabelSource','foldernames');

% Splitting data into test and train datasets
[trainImgs,testImgs] = splitEachLabel(flowerds,0.6);

% Num of species (Labels)
numClasses           = numel(categories(flowerds.Labels));

% Initializing pretrained CNN
net                  = alexnet;

% adding fully connected layer and classification layer
layers               = net.Layers;
layers(end-2)        = fullyConnectedLayer(numClasses);
layers(end)          = classificationLayer;

% setting training options
options              = trainingOptions('sgdm','InitialLearnRate', 0.001);

% training
[flowerNet,info]     = trainNetwork(trainImgs, layers, options);

% testing
testPreds            = classify(flowerNet,testImgs);

% evalution of results
nnz(testPreds == testImgs.Labels)/numel(testPreds)
confusionchart(testImgs.Labels,testPreds);
