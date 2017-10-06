%%% pedestrain detection system 

basePath = '/tmp/rsubrama/INRIAPerson/' ;

path = sprintf('%s/Train/pos/', basePath);

COLS = 100; 
%testSamples  = 50;

% Process all images in pos folder
filePattern = fullfile(path, '*.*g'); 
theFiles = dir(filePattern);

masterIMageDataSet = []; 
masterIMageDataSetPos = []; 
masterIMageDataSetNeg = []; 

trainSamples = size(theFiles);

fprintf('Training positive dataset (%d photos)\n', trainSamples(1,1));

for k=1: trainSamples(1,1)
    baseFileName = theFiles(k).name;
    file = fullfile(path, baseFileName);
    
    I = imread(file);
    J = imresize(I, [256,256]);

    [hog, pic] = extractHOGFeatures(J,'CellSize',[16, 16], 'BlockSize', [1,1]);

    masterIMageDataSetPos = [masterIMageDataSetPos; hog]; 
end
masterIMageDataSetPos = [masterIMageDataSetPos, ones(k,1)];


path = sprintf('%s/Train/neg/', basePath);

% Process all images in pos folder
filePattern = fullfile(path, '*.*g'); 
theFiles = dir(filePattern);
trainSamples = size(theFiles);

fprintf('Training Negative dataset (%d photos)\n', trainSamples(1,1));

for k=1: trainSamples(1,1)
    baseFileName = theFiles(k).name;
    file = fullfile(path, baseFileName);
    
    I = imread(file);
    J = imresize(I, [256,256]);

    [hog, pic] = extractHOGFeatures(J,'CellSize',[16, 16], 'BlockSize', [1,1]);
    
    masterIMageDataSetNeg = [masterIMageDataSetNeg; hog]; 
end
masterIMageDataSetNeg = [masterIMageDataSetNeg, zeros(k,1)];
masterIMageDataSet = [masterIMageDataSetPos; masterIMageDataSetNeg];


%%% Perform PCA 
m = mean(masterIMageDataSet(:,1:end-1));
x_size = size(masterIMageDataSet);
M = repmat(m, x_size(1,1),1);
M = [M, zeros(x_size(1,1),1)];
XM = masterIMageDataSet-M;

% Perform PCA
[eigvec_org eigval_org] = eig(cov(masterIMageDataSet(:,1:end-1)));
eigvec = fliplr(eigvec_org); % largest evector on 1st col
eigval = flipud(diag(eigval_org)); % largest evalue on top

esize = size(eigvec);
eigvec = [eigvec; zeros(1, esize(1,2))];
esize = size(eigvec);
newCol = zeros(esize(1,1), 1);
newCol(end,1) = 1;
eigvec = [eigvec(:,:), newCol];
PCA = XM*eigvec; 


%SVMStructNew = svmtrain(masterIMageDataSet(:,1:end-1),masterIMageDataSet(:,end)); 
SVMStructNew = svmtrain(PCA(:,1:COLS), PCA(:,end)); 


%%% Positive Test Data 
path = sprintf('%s/Test/pos/', basePath);

pos_result = 0;
total_pos = 0;

% Process all images in pos folder
filePattern = fullfile(path, '*.*g'); 
theFiles = dir(filePattern);
testSamples = size(theFiles);

Group = []; 
imageMatrix = []; 

for k=1: testSamples(1,1)  
    baseFileName = theFiles(k).name;
    file = fullfile(path, baseFileName);

    I = imread(file);
    J = imresize(I, [256,256]);    
    
    [hog, pic] = extractHOGFeatures(J,'CellSize',[16, 16], 'BlockSize', [1,1]);
    imageMatrix = [imageMatrix ; hog];
end

testImagePCA = imageMatrix * eigvec(1:end-1, 1:end-1);
Group = svmclassify(SVMStructNew,testImagePCA(:,1:COLS));
%Group = svmclassify(SVMStructNew,imageMatrix);
pos = sum(Group);
pos_result = sum(Group)/testSamples(1,1);
total_pos = testSamples(1,1);
fprintf('Classification accuracy for Positive Dataset: %f\n', pos_result);

%%%% Classification accuracy for Negative
%%% Negative Test Data 
path = sprintf('%s/Test/neg/', basePath);

neg_result = 0;

% Process all images in neg folder
filePattern = fullfile(path, '*.*g'); 
theFiles = dir(filePattern);
testSamples = size(theFiles);

total_neg = 0; 
imageMatrix = []; 
GroupNeg = []; 
testImagePCA = [];

for k=1: testSamples(1,1)
    baseFileName = theFiles(k).name;
    file = fullfile(path, baseFileName);

    I = imread(file);
    J = imresize(I, [256,256]);    
    
    [hog, pic] = extractHOGFeatures(J,'CellSize',[16, 16], 'BlockSize', [1,1]);
    imageMatrix = [imageMatrix ; hog];
end

testImagePCA = imageMatrix * eigvec(1:end-1, 1:end-1);
GroupNeg = svmclassify(SVMStructNew,testImagePCA(:,1:COLS));
%GroupNeg = svmclassify(SVMStructNew,imageMatrix);

neg = testSamples(1,1)-sum(GroupNeg); 
neg_result = (testSamples(1,1)-sum(GroupNeg))/testSamples(1,1);

fprintf('Classification accuracy for Negative Dataset: %f\n', neg_result);
total_neg = testSamples(1,1);
overall =  (neg+pos) / (total_pos + total_neg);

fprintf('Overall Accuracy : %f\n', overall);
