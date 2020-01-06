% import HW5_Bow.*;
% a = HW5_BoW();
% a.main();
load trD.mat;
load tstD.mat;
load trLbs.mat;
[trainK, testK] = cmpExpX2Kernel(trD',tstD',2);
s1 = '-t 4 -c';
s2 = num2str(32);
s3 = '-g';
s4 = num2str(2);
s = [s1 ' ' s2 ' ' s3 ' ' s4];
trainK = [(1:1777)', trainK];
testK = [(1:1600)', testK];
[model]  = svmtrain(trLbs,trainK,s);
test_labels = ones(1600,1);
[preds] = svmpredict(test_labels,testK, model);
tstIds = ml_load('test.mat', 'imIds');
writetable(table(tstIds',preds), 'predTestLabels.csv');
function [trainK, testK] = cmpExpX2Kernel(trainD, testD, gamma)
[m,~] = size(trainD);
[a,~] = size(testD);
trainK = zeros(m,m);
testK = zeros(a,m);
epsilon = 0.000001;
for i = 1:m
    num = trainD - trainD(i,:);
    den = trainD + trainD(i,:);
    trainK(:,i) = sum(num.^2 ./ (den + epsilon), 2);
end
train_gamma = mean(trainK, 'all');
trainK = exp(trainK * (-1/gamma));
for i = 1:m
    num = testD - trainD(i,:);
    den = testD + trainD(i,:);
    testK(:,i) = sum(num.^2 ./ (den + epsilon),2);
end
test_gamma = mean(testK, 'all');
testK = exp(testK * (-1/gamma));
end