function [X,Y] = load_data(filepath,show_flag)
%     filepath = 'PreTreatment\Train'; % ing data path

if nargin<2
    show_flag=false;
end

    imds = imageDatastore(filepath, ... 
        'IncludeSubfolders',true, ... 
        'LabelSource','foldernames'); 
    filepath = imds.Files;
    tmp = imread(filepath{1});
    [w,h,c] = size(tmp);
    num = length(filepath);
    X = zeros(w,h,c,num);
    for i = 1:num
        if rem(i,500)==0
            string_disp="loading data: " + num2str(i)+"/"+num2str(num);
            disp(string_disp)
        else if i==num
            string_disp="loading data: " + num2str(i)+"/"+num2str(num);
            disp(string_disp)
            end
        end
         pic = imread(filepath{i});
    %     imshow(pic)
        X(:,:,:,i) = pic;
    end
    Y  = imds.Labels;
    
    if show_flag
        figure;
        idx = randperm(numel(Y),9);
        for i = 1:numel(idx)
            subplot(3,3,i);
            imshow(X(:,:,:,idx(i)));
        end
    end

end