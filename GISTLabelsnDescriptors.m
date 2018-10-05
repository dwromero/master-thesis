clear param
param.imageSize = [256 256]; % set a normalized image size
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Pre-allocate gist:
directory = 'C:/Users/David Romero/Documents/MasterThesis/Object Detection/Dataset/voc2006_test/VOCdevkit/VOC2006/PNGImages/';
files = dir(directory);
files = files(3:end);
Nimages = size(files,1);

Nfeatures = sum(param.orientationsPerScale)*param.numberBlocks^2;
gist = zeros([Nimages Nfeatures]);
labels = strings([Nimages 1]);

%

% Load first image and compute gist:
path = strcat(directory,files(1).name)
%str = string(files(1).name)
%int2str(label(path))
labels(1) = strcat(files(1).name,';',int2str(label(path)));
img = imread(path);
[gist(1, :), param] = LMgist(img, '', param); % first call

% Loop:
for i = 2:Nimages
   path = strcat(directory,files(i).name)
   labels(i) = strcat(files(i).name,';',int2str(label(path)));
   img = imread(path);
   gist(i, :) = LMgist(img, '', param); % the next calls will be faster
end


dlmwrite("OD_GistDescriptors_test.txt",gist,';');

fid = fopen('OD_GistLabels_test.txt','wt');
fprintf(fid, '%s\n', labels);


function y = label(string)
if contains(string,'coast') y = 1; return; end
if contains(string,'forest') y = 2; return; end
if contains(string,'highway') y = 3; return; end
if contains(string,'insidecity') y = 4; return; end
if contains(string,'mountain') y = 5; return; end
if contains(string,'opencountry') y = 6; return; end
if contains(string,'street') y = 7; return; end
if contains(string,'tallbuilding') y = 8; return; end
y = 0;
end