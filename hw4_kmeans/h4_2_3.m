test_data = csvread('test_data.csv',0,0);
test_data = normalize(test_data);
[r_test,c_test] = size(test_data);
% name = 'Test_Features.csv';
% fid = fopen(name,'r');
% test_data = [];
% test_labels = [];
% while ~feof(fid)
% tline = fgetl(fid);
% temp = strsplit(tline);
% test_labels = [test_labels;temp(1)];
% temp(1) = [];
% test_data = [test_data;temp];
% end
% fclose(fid);
preds = [];
for i =1:r_test
    max = -100000;
    max_v = 0;
    for n = 1:4
        y_pred = test_data(i,:)*w_vecs(n,:)' + b_vals(n);
        if(y_pred > max)
            max = y_pred;
            max_v = n;
        end
    end
    preds = [preds;max_v];
end