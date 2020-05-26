%___________________________________________________________________________________________________________________
% Automatic Detection of Microaneurysm from Color Fundus images
%                    Digital Image Processing
%                           - Ritu Lahoti
% International Insitute of Information Technology, Bangalore (IIITB)
%___________________________________________________________________________________________________________________
%%
clc
clear 
% close all

%%  Part II - Splitting the data into training and testing, building a classifer model using training data 
%   and identify class label for MA candidates in testing data

%% Reading the extracted features from .xls file obtained using MA_FeatureExtraction.m code

sheet = 1;
[data, label]= xlsread('features.xls', sheet,'A2:L5297' );

%% Splitting the data into training: 70% and testing: 30%

cv = cvpartition(size(data,1),'HoldOut',0.3);
idx = cv.test;

train = data(~idx,:);
test  = data(idx,:);

%% Build a Naive Bayes classification model based on kernel density estimate with varying width

% Obtain bandwidth value for each feature vector (/predictor variable) belonging to each classes i.e.
% a matrix of 2 rows of both classes and 12 columns of each feature

label1 = label(~idx(:));
data1 = data(~idx,:);
h=1;
k=1;
% FP, TP: indices of training data belonging to falsePos(Non-MA) and truePos(MA)
for ind = 1:1:3708
    if isequal(label1(ind), {'falsePos(Non-MA)'})==1
        FP(h,:) = ind;
        h=h+1;
    else
        TP(k,:) = ind;
        k=k+1;
    end
end

%% Estimating bandwidth value for each feature vector (/predictor variable)

for i=1:1:11
    [f,xi,bw1] = ksdensity(data1(FP,i)); 
    bandwidth1(:,i) = bw1;
 
    [f,xi,bw2] = ksdensity(data1(TP,i));
    bandwidth2(:,i) = bw2;
end

%% Build a Naive Bayes classification model with kernel distribution

bandwidth = vertcat(bandwidth1,bandwidth2);
distribution=['kernel','kernel'];

modelNB = fitcnb(data(~idx,:),label(~idx,:), 'DistributionNames', distribution,...
    'ClassNames',{'falsePos(Non-MA)','truePos(MA)'}, 'Width', bandwidth)

%% Predicting class labels for testing data and measuring its accuracy

predictions = modelNB.predict(data(idx,:));
cp = classperf(label);
metric = classperf(predictions,label(idx));
figure, ConfusionMat2 = confusionchart(label(idx),predictions);

%% Extract features from MA Candidates of a test data and predict its label 
%  and mark circle around both labels with different colors

srcFiles = dir('testData\DS00091B.jpg');
count=0;

for z = 1 : length(srcFiles)
    filename = strcat('testData\',srcFiles(z).name);
    I = imread(filename); 
    green_channel = I(:,:,2);
    grayImage = green_channel;
    figure, imshow(grayImage),title('Green Channel')

    mask=imbinarize(grayImage,0.05); % im2bw

    %%
    % backgroundFilterSize
    bgFilterSize = [78 78]; %68 68

    % Median filtering, to remove Salt& Pepper (S-P) noise
    intermediate.SPdenoised = medfilt2(grayImage, [3, 3]); %3 3

    % Histogram equalisation
    intermediate.histeq = adapthisteq(intermediate.SPdenoised);
    figure, imshow(intermediate.histeqImage),title('Contrast Enhanced')

    % Gaussian Smoothing to attenuate noise
    intermediate.gaussImage = imgaussfilt( double(intermediate.histeq), 2, 'FilterSize', [3 3]);

    % MedianImage for background estimation (very large median filter)
    intermediate.bgEstimateImage = medfilt2( uint8(intermediate.gaussImage), bgFilterSize);
    figure, imshow(intermediate.bgEstimateImage),title('Background Estimate')

    % Store this as the background image
    intermediate.shadecorrectedImage = intermediate.gaussImage ./ double(intermediate.bgEstimateImage+1);
    preprocessedImage = intermediate.shadecorrectedImage / (std2(intermediate.shadecorrectedImage)+1);

    v = preprocessedImage.*mask;
    figure, imshow(v),title('Shade Corrected')

    %%
    % Performs morphological closing operation using linear structuring element to extract blood vessels    
    % A degree range from 0 to 180 degrees (with increment factor '3')is considered, since the line strel is symmetrical.
    vessel = detectVessel(v,'tophatStrelSize', 11);
    tophatImage = vessel - v;
    figure, imshow(vessel), title('Extracted Blood Vessels')
    figure, imshow(tophatImage),title('Top-hat Filtered')

    % Creates a 2D-Gaussian lowpass filter and perform filtering on tophatImage
    gaussWindowSize = [15 15];
    gaussSigma = 1;
    h = fspecial('gaussian', gaussWindowSize, gaussSigma);
    gaussImage = imfilter(tophatImage, h, 'same');

    %%
    se = strel('disk',5);
    closeBW = imclose(gaussImage,se);

    bin = imbinarize(closeBW,0.1);
    bwCandidates=imfill(bin,'holes');
    figure, imshow(bwCandidates)

    grayCandidates=bwCandidates.*gaussImage;
    [grayLabels, numConn] = bwlabeln(grayCandidates,26); 
    
    stats1 = regionprops(bwCandidates,'Centroid','Area','Perimeter','Eccentricity','Extent','MajoraxisLength','MinoraxisLength','Orientation');
    stats2 = regionprops(grayLabels,grayCandidates,'all');
    
    numberOfBlobs = size(stats2, 1);
    for k = 1 : numberOfBlobs          
        pos = stats2(k).BoundingBox; 
        x = pos(1);
        y = pos(2);
        w = pos(3);
        h = pos(4);
        candi = grayImage(y:y+w,x:x+h);
        G = graycomatrix(candi,'NumLevels',256);
        glcm(k,:) = graycoprops(G);
    end
    
    % Predict label for each MA candidates and mark on the green channel of
    % original input image
    
    figure, imshow(I)
%     figure, imshow(grayImage)
    hold on;
    for s = 1:1:numConn
        c = s+count;
        aspect_ratio = stats1(s).MajorAxisLength./ stats1(s).MinorAxisLength;
        feat(c,:) = [stats1(s).Area, stats1(s).Perimeter, stats1(s).Eccentricity,...
            stats1(s).Extent, aspect_ratio, stats1(s).Orientation, stats2(s).MaxIntensity,...
            stats2(s).MeanIntensity, stats2(s).MinIntensity, glcm(s).Energy, glcm(s).Homogeneity];
        
        % Predict the class of test data
        results = modelNB.predict(feat(c,:))
        t=uint8(2);
        r = stats1(s).Perimeter;
        if isequal(results, {'falsePos(Non-MA)'})==1
            viscircles(stats1(s).Centroid, r,'Color','r','LineWidth',t);
        else
            viscircles(stats1(s).Centroid, r,'Color','b','LineWidth',t);
        end
    end
    count=numConn + count;
    
end

%% Defining linear morphological operation along multiple directions

function [vessel] = detectVessel(img, varargin)

    p = inputParser();

    addParameter(p, 'degreeRange', 0:3:180);
    addParameter(p, 'tophatStrelSize', 11);
    parse(p, varargin{:});
    degrees = p.Results.degreeRange;
    strel_size = p.Results.tophatStrelSize;

    vessel= ones( size(img)) * 9999;
    for deg=degrees
        str_el = strel('line', strel_size, deg);
        c = imclose(img, str_el);
        vessel = min(c, vessel);
    end
end
