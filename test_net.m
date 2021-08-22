
function [confusion_mat,evaluate,test_accuracy] = test_net(trainedNet_path,filepath,net_name)
    % filepath = "PreTreatment\Test";
    % trainedNet_path = "BestNet\GoogLeNet_0.11127_.mat";
    %% load data
    imdsTest= imageDatastore(filepath, ... 
            'IncludeSubfolders',true, ... 
            'LabelSource','foldernames');

    %% load net
    trainedNet = load(trainedNet_path);
    trainedNet = trainedNet.trainedNet;
    %% test
    inputSize = trainedNet.Layers(1, 1).InputSize;
    augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

    %‘§≤‚∑÷¿‡ 
    [YPred, scores] = classify(trainedNet,augimdsTest);
    YTest = imdsTest.Labels;
    test_accuracy =  mean(YPred == YTest);
    

    TP=[];
    TN=[];
    FP=[];
    FN=[];
    for i = 1:length(YTest)
        if YPred(i)==categorical(1)
            if YPred(i)==YTest(i)
                TP(end+1)=1;
            else
                FP(end+1)=1;
            end
        else
            if YPred(i)==YTest(i)
                TN(end+1)=1;
            else
                FN(end+1)=1;
            end
        end
    end
    TN = length(TN);
    FN = length(FN);
    FP = length(FP);
    TP = length(TP);
    confusion_mat = [TN,FN;
                      FP,TP];
    confusion_mat
    
    Recall = TP/(TP+FN);
    Specificity = TN/(TN+FP);
    Precision = TP/(TP+FP);
    F1 = 2*Precision*Recall/(Precision+Recall);
    Recall,Specificity,Precision,F1
    
    evaluate=[Recall,Specificity,Precision,F1];
    YPred = double(string(YPred));
    YTest = double(string(YTest));
    savepath = "TestResult\"+net_name+"_pre&real.mat";
    save(savepath,"YTest","YPred");
end





