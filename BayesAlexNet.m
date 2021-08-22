clc;
clear all;
close all;
%% 加载网络
load AlexNet.mat;
net_name = "AlexNet";
%% load dataset

[XTrain,YTrain]=load_data('PreTreatment\Train');
[XVal,YVal]=load_data('PreTreatment\Val',true);

%% 需要优化的参数
optimVars = [
    optimizableVariable('InitialLearnRate',[3e-4 1e-2],'Transform','log')
    optimizableVariable('Momentum',[0.8 0.98])
    optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];

ObjFcn = makeGObjFcn(XTrain,YTrain,XVal,YVal,net,net_name);

BayesObject = bayesopt(ObjFcn,optimVars, ...
    'MaxTime',14*60*60, ...
    'IsObjectiveDeterministic',false, ...
    'UseParallel',false);