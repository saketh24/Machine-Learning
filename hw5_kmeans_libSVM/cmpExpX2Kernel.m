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