ANNOTATIONS_FOLDER = 'C:\Users\David Romero\Documents\MasterThesis\Object Detection\Dataset\voc2006_test\VOCdevkit\VOC2006\Annotations\';

annotation_list = dir(ANNOTATIONS_FOLDER)
annotations = strings(1);

for i = 1 : size(annotation_list)
    if ~contains(annotation_list(i).name,'.txt') continue; end
    path = strcat(ANNOTATIONS_FOLDER , annotation_list(i).name)
    p_ann = PASreadrecord(path);
    for obj = 1 : numel(p_ann.objects)
        object = p_ann.objects(obj);
        label_obj = label(object.label);
        if label_obj == 999 continue; end
        
        objbox = p_ann.objects(obj).bbox;
        string = strcat(p_ann.imgname,';',int2str(objbox(1)),';',int2str(objbox(2)),';',int2str(objbox(3)),';',int2str(objbox(4)),';',int2str(label_obj))
        annotations(end + 1) = string;
    end
end

numel(annotations)
fid = fopen('PASCAL_test_annotations.txt','wt');
fprintf(fid, '%s\n', annotations);


function y = label(string)
if contains(string,'car') y = 1;
elseif contains(string,'person') y = 2; 
elseif contains(string,'horse') y = 3; 
elseif contains(string,'cow') y = 4; 
else y = 999;
end
end