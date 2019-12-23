tr_data = csvread('trainData.csv',0,1);
trLb = csvread('trainLabels.csv',0,1);
val_data = csvread('valData.csv', 0, 1);
val_labels = csvread('valLabels.csv', 0, 1);
test_data = csvread('testData_new.csv', 0, 1);
% tr_data(:,1) = [];
% val_data(:,1) = [];
% test_data(:,1) = [];
tr_data = transpose(tr_data);
val_data = transpose(val_data);
%tr_data = [tr_data,val_data];
%trLb = [trLb;val_labels];
test_data = transpose(test_data);
[o,p] = size(test_data);
[f,g] = size(val_data);
results = zeros(p,1);
[w,b,obj, cvErrs] = ridgeReg(tr_data, trLb, 1);
val_error = 0;
for i = 1:g
    val = transpose(w) * val_data(:, i) + b - val_labels(i);
    val_error = val_error + (val)^2;
end
disp("Val_error:" + sqrt(val_error/g));
for i = 1:p
    results(i) = transpose(w) * test_data(:, i) + b;
end

function [w,b,obj, cvErrs] = ridgeReg(X,y,lambda)
[k, n] = size(X);
One_vec = ones(1,n);
Zero_vec = zeros(k,1);
X_bar = [X;One_vec];
I_bar = [eye(k),Zero_vec; transpose(Zero_vec),0];
C = X_bar * transpose(X_bar) + lambda * I_bar;
d = X_bar * y;
W_bar = C\d;
w = W_bar(1:k);
b = W_bar(k+1);
sum = 0;
tr_error = 0;
cv_error = 0;
parfor i = 1:n
    error = (transpose(w) * X(:, i) + b - y(i))^2;
    sum = sum + error;
end
tr_error = sqrt(sum/n);
disp(sum);
obj = lambda * (norm(w))^2 + sum;
cvErrs = zeros(n,1);
cv_sum = 0;
parfor i = 1:n
    numerator = transpose(W_bar) * X_bar(:,i) - y(i,:);
    denominator = 1 - transpose(X_bar(:,i))/C * X_bar(:,i);
    coov = (numerator/denominator)^2;
    cvErrs(i) = coov;
    cv_sum = cv_sum + coov;
    disp(i);
    disp(coov);
end
cv_error = sqrt(cv_sum/n);
disp("Cv_error:" + cv_error);
disp("Training error:" + tr_error);
end


