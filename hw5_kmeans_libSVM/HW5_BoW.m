classdef HW5_BoW    
% Practical for Visual Bag-of-Words representation    
% Use SVM, instead of Least-Squares SVM as for MPP_BoW
% By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
% Created: 18-Dec-2015
% Last modified: 16-Oct-2018    
    
    methods (Static)
        function main()
            scales = [8, 16, 32, 64];
            normH = 16;
            normW = 16;
            %bowCs = HW5_BoW.learnDictionary(scales, normH, normW);
            load centroids.mat;
            disp("hello");
            [trIds, trLbs] = ml_load('train.mat',  'imIds', 'lbs');             
            tstIds = ml_load('test.mat', 'imIds'); 
%             trD  = HW5_BoW.cmpFeatVecs(trIds, scales, normH, normW, bowCs);
%             tstD = HW5_BoW.cmpFeatVecs(tstIds, scales, normH, normW, bowCs);
%             save('trD.mat', 'trD');
%             save('tstD.mat', 'tstD');
%             save('trLbs.mat', 'trLbs');
            load trD.mat;
            load tstD.mat;
            load trLBs.mat;
            %model = svmtrain(trLbs,trD', '-v 5');
            %disp(model);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Write code for training svm and prediction here            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             best_acc = 0;
%             best_c  = 0;
%             best_g = 0;
%             for c = -5:10
%                 for g = -10:5
%                     s1 = '-v 5 -c';
%                     s2 = num2str(2^c);
%                     s3 = '-g';
%                     s4 = num2str(2^g);
%                     s = [s1 ' ' s2 ' ' s3 ' ' s4];
%                     acc = svmtrain(trLbs, trD', s);
%                     if acc > best_acc
%                         best_acc = acc;
%                         best_c = 2^c;
%                         best_g = 2^g;
%                     end
%                 end
%             end
%             disp('Best Acc:' + best_acc);
%             disp('Best C:' + best_c);
%             disp('Best G:' + best_g);


            best_acc = 0;
            best_c = 0;
            best_g = 0;
            for c = -5:10
                for g = -1:6
                    [trainK, ~] = HW5_BoW.cmpExpX2Kernel(trD',tstD',2^g);
                    trainK = [(1:1777)', trainK];
                    s1 = '-t 4 -v 5 -c';
                    s2 = num2str(2^c);
                    s3 = '-g';
                    s4 = num2str(2^g);
                    s = [s1 ' ' s2 ' ' s3 ' ' s4];
                    model = svmtrain(trLbs, trainK, s);
                    if model > best_acc
                        best_acc = model;
                        best_c = 2^c;
                        best_g = 2^g;
                    end
                end
                fprintf("current C: %d", c);
                fprintf("best so far: %f", best_acc);
            end
            fprintf("Best Acc: %f \n", best_acc);
            fprintf("Best C: %f \n", best_c);
            fprintf("Best Gamma: %f \n", best_g);
        end
                
        function bowCs = learnDictionary(scales, normH, normW)
            % Number of random patches to build a visual dictionary
            % Should be around 1 million for a robust result
            % We set to a small number her to speed up the process. 
            nPatch2Sample = 100000;
            
            % load train ids
            trIds = ml_load('train.mat', 'imIds'); 
            nPatchPerImScale = ceil(nPatch2Sample/length(trIds)/length(scales));
                        
            randWins = cell(length(scales), length(trIds)); % to store random patches
            for i=1:length(trIds);
                ml_progressBar(i, length(trIds), 'Randomly sample image patches');
                im = imread(sprintf('bigbangtheory_v3/%06d.jpg', trIds(i)));
                im = double(rgb2gray(im));  
                for j=1:length(scales)
                    scale = scales(j);
                    winSz = [scale, scale];
                    stepSz = winSz/2; % stepSz is set to half the window size here. 
                    
                    % ML_SlideWin is a class for efficient sliding window 
                    swObj = ML_SlideWin(im, winSz, stepSz);
                    
                    % Randomly sample some patches
                    randWins_ji = swObj.getRandomSamples(nPatchPerImScale);
                    
                    % resize all the patches to have a standard size
                    randWins_ji = reshape(randWins_ji, [scale, scale, size(randWins_ji,2)]);                    
                    randWins{j,i} = imresize(randWins_ji, [normH, normW]);
                end
            end
            randWins = cat(3, randWins{:});
            randWins = reshape(randWins, [normH*normW, size(randWins,3)]);
                                    
            fprintf('Learn a visual dictionary using k-means\n');
            K=1000;
            [~,~,~, bowCs] = HW5_BoW.kmeans(randWins',1, K);
            bowCs = bowCs';
            save('centroids.mat', 'bowCs');
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Use your K-means implementation here                       %
            % to learn visual vocabulary                                 %
            % Input: randWinds contains your data points                 %
            % Output: bowCs: centroids from k-means, one column for each centroid  
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
                
        function D = cmpFeatVecs(imIds, scales, normH, normW, bowCs)
            n = length(imIds);
            D = cell(1, n);
            startT = tic;
            for i=1:n
                ml_progressBar(i, n, 'Computing feature vectors', startT);
                im = imread(sprintf('bigbangtheory_v3/%06d.jpg', imIds(i)));                                
                bowIds = HW5_BoW.cmpBowIds(im, scales, normH, normW, bowCs);                
                feat = hist(bowIds, 1:size(bowCs,2));
                D{i} = feat(:);
            end
            D = cat(2, D{:});
            D = double(D);
            D = D./repmat(sum(D,1), size(D,1),1);
        end        
        
        % bowCs: d*k matrix, with d = normH*normW, k: number of clusters
        % scales: sizes to densely extract the patches. 
        % normH, normW: normalized height and width oMf patches
        function bowIds = cmpBowIds(im, scales, normH, normW, bowCs)
            im = double(rgb2gray(im));
            bowIds = cell(length(scales),1);
            for j=1:length(scales)
                scale = scales(j);
                winSz = [scale, scale];
                stepSz = winSz/2; % stepSz is set to half the window size here.
                
                % ML_SlideWin is a class for efficient sliding window
                swObj = ML_SlideWin(im, winSz, stepSz);
                nBatch = swObj.getNBatch();
                
                for u=1:nBatch
                    wins = swObj.getBatch(u);
                    
                    % resize all the patches to have a standard size
                    wins = reshape(wins, [scale, scale, size(wins,2)]);                    
                    wins = imresize(wins, [normH, normW]);
                    wins = reshape(wins, [normH*normW, size(wins,3)]);
                    
                    % Get squared distance between windows and centroids
                    dist2 = ml_sqrDist(bowCs, wins); % dist2: k*n matrix, 
                    
                    % bowId: is the index of cluster closest to a patch
                    [~, bowIds{j,u}] = min(dist2, [], 1);                     
                end                
            end
            bowIds = cat(2, bowIds{:});
        end 
        
        function [clusters,iter_count,square_sum, centroids] = kmeans(X,israndom,num_clusters)   
[m,n] = size(X);
if israndom == 0
    centroids = X(1:num_clusters,:);
else
    rand = randperm(m);
    centroids = X(rand(1:num_clusters),:);
end
cluster_check = zeros(m,1);
for iter_count = 1:100

square_sum = 0;
clusters = zeros(m,1);
for i = 1:m
    for j = 1:num_clusters
        dist = norm(X(i,:) - centroids(j,:))^2;
        if j ==1
            nearest = dist;
            cluster_i = j;
            continue;
        else
            if dist < nearest
                nearest = dist;
                cluster_i = j;
            end
        end
    end
    clusters(i) = cluster_i;
    square_sum = square_sum + nearest;
end
if clusters == cluster_check
    break;
end
for i = 1:num_clusters
centroids(i,:) = mean(X(clusters == i,:));
end
cluster_check = clusters;
end
return
        end
        

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
    end    
end

