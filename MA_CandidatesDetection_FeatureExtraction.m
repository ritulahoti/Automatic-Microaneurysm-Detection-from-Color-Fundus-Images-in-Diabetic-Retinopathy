%___________________________________________________________________________________________________________________
% Automatic Detection of Microaneurysm from Color Fundus images
%                    Digital Image Processing
%                           - Ritu Lahoti
% International Insitute of Information Technology, Bangalore (IIITB)
%___________________________________________________________________________________________________________________
%%
clc
clear

%%    Part I - Extracting features from MA Candidates and storing them in .xls file

%% False Positive MA Candidates
%%
srcFiles = dir('falseData\*.jpg');
count=0;

for z = 1 : length(srcFiles)
    filename = strcat('falseData\',srcFiles(z).name);
    [folder, baseFileNameNoExt, extension] = fileparts(filename);

    I = imread(filename); 
    green_channel = I(:,:,2);
    grayImage = green_channel;

    mask=imbinarize(grayImage,0.05); %im2bw

    %%
    % BackgroundFilterSize
    bgFilterSize = [78 78]; 

    % Median filtering, to remove noise
    intermediate.SPdenoised = medfilt2(grayImage, [3, 3]); 
    
    % Histogram equalisation
    intermediate.histeq = adapthisteq(intermediate.SPdenoised);

    % Gaussian smoothing
    intermediate.gaussImage = imgaussfilt( double(intermediate.histeq), 2, 'FilterSize', [3 3]);

    % MedianImage for background estimation (very large median filter)
    intermediate.bgEstimateImage = medfilt2( uint8(intermediate.gaussImage), bgFilterSize);

    % Store this as the background image
    intermediate.shadecorrectedImage = intermediate.gaussImage ./ double(intermediate.bgEstimateImage+1);
    preprocessedImage = intermediate.shadecorrectedImage / (std2(intermediate.shadecorrectedImage)+1);

    % Multiple the preprocessed image with binary mask of input image
    v = preprocessedImage.*mask;

    %%
    % Performs morphological closing operation using linear structuring element to extract blood vessels    
    % A degree range from 0 to 180 degrees (with increment factor '3')is considered, since the line strel is symmetrical.
    vessel = detectVessel(v,'tophatStrelSize', 11);
    tophatImage = vessel - v;

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
    
    grayCandidates=bwCandidates.*gaussImage;
    [grayLabels, numConn] = bwlabeln(grayCandidates,26); % impixelinfo
    
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

    for s = 1:1:numConn
        c = s+count;
        aspect_ratio = stats1(s).MajorAxisLength./ stats1(s).MinorAxisLength;
        feat1(c,:) = [stats1(s).Area, stats1(s).Perimeter, stats1(s).Eccentricity,...
            stats1(s).Extent, aspect_ratio, stats1(s).Orientation, stats2(s).MaxIntensity,...
            stats2(s).MeanIntensity, stats2(s).MinIntensity, glcm(s).Energy, glcm(s).Homogeneity];
    end
    count=numConn + count;
    
end

sheet = 1;
tag={'Label'};
c={'Area' 'Perimeter' 'Eccentricity' 'Extent' 'AspectRatio' 'Orientation' ' MaxIntensity' 'MeanIntensity' 'MinIntensity' 'Energy' 'Homogeneity'};
parameters(1,:)=c;
xlswrite('features.xls',tag,sheet,'A1')
xlswrite('features.xls',parameters,sheet,'B1')
xlswrite('features.xls',feat1,sheet,'B2')

str1 = "A";
str2 = num2str(count+1);
str = append(str1,'2',':',str1,str2);
tag={'falsePos(Non-MA)'};
predict(:,1)=tag;

xlswrite('features.xls',predict,sheet,str)
rowNo = count+2;

%--------------------------------------------------------------------------
%%  True Positive MA Candidates
%%
srcFiles = dir('OriginalImage_trueData\*.jpg');
count=0;

for z = 1 : length(srcFiles)
    filename1 = strcat('OriginalImage_trueData\',srcFiles(z).name);
    I1 = imread(filename1); 
    grayImage = I1(:,:,2);
    [folder, baseFileNameNoExt, extension] = fileparts(filename1);
   
    sameName = append(baseFileNameNoExt,'.png');

    filename2 = strcat('trueData\',sameName);
    bwCandidates = imread(filename2);
    
    %% To generate and store filtered grayscale image (gaussImage)
    
    mask=imbinarize(grayImage,0.05); % im2bw
    
    bgFilterSize = [78 78]; 

    intermediate.SPdenoised = medfilt2(grayImage, [3, 3]); %3 3

    intermediate.histeq = adapthisteq(intermediate.SPdenoised);

    intermediate.gaussImage = imgaussfilt( double(intermediate.histeq), 2, 'FilterSize', [3 3]);

    intermediate.bgEstimateImage = medfilt2( uint8(intermediate.gaussImage), bgFilterSize);

    intermediate.shadecorrectedImage = intermediate.gaussImage ./ double(intermediate.bgEstimateImage+1);
    preprocessedImage = intermediate.shadecorrectedImage / (std2(intermediate.shadecorrectedImage)+1);

    v=preprocessedImage.*mask;

    vessel = detectVessel(v,'tophatStrelSize', 11);
    tophatImage = vessel - v;

    gaussWindowSize = [15 15];
    gaussSigma = 1;%1
    h = fspecial('gaussian', gaussWindowSize, gaussSigma);
    gaussImage = imfilter(tophatImage, h, 'same');
    
    %% Saving gaussImage in a new folder and can be used for further processing directly 
    
%     [folder, baseFileNameNoExt, extension] = fileparts(filename1);
%     newName = baseFileNameNoExt;
%     newFolder = 'gauss_MA';
%     fullFileName = fullfile(newFolder, newName);
%     fileFolder = append(fullFileName,'.jpg');
%     imwrite(gaussImage, fileFolder);

    %%
    bwCandidates = imbinarize(bwCandidates,0.1);

    grayCandidates = bwCandidates.*gaussImage;
    [grayLabels, numConn] = bwlabel(grayCandidates,8); % impixelinfo

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

    for s = 1:1:numConn
        c = s+count;
        aspect_ratio = stats1(s).MajorAxisLength./ stats1(s).MinorAxisLength;
        feat2(c,:) = [stats1(s).Area, stats1(s).Perimeter, stats1(s).Eccentricity,...
            stats1(s).Extent, aspect_ratio, stats1(s).Orientation, stats2(s).MaxIntensity,...
            stats2(s).MeanIntensity, stats2(s).MinIntensity, glcm(s).Energy, glcm(s).Homogeneity];
    end
    count=numConn + count;

end

str3 = "B";
str4 = num2str(rowNo);
str5 = append(str3,str4);

sheet = 1;
xlswrite('features.xls',feat2,sheet,str5)

str1 = "A";
str2 = num2str(rowNo);
str3 = num2str(count+rowNo);
str = append(str1,str2,':',str1,str3);
tag={'truePos(MA)'};
predict(:,1)=tag;
xlswrite('features.xls',predict,sheet,str)

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

