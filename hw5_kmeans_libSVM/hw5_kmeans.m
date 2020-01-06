X = load("digit.txt");
Y = load("labels.txt");
[clusters,iters,square_sum,p1,p2,p3,centroids] = kmeans(X,Y,0,6);
sums = [];
p1_vals = [];
p2_vals = [];
p3_vals = [];
k_vals = [1 2 3 4 5 6 7 8 9 10];
for k = 1:10
    [clusters,iters,square_sum,p1,p2,p3,centroids] = kmeans(X,Y,1,k);
    sums = [sums,square_sum];
    p1_vals = [p1_vals,p1];
    p2_vals = [p2_vals,p2];
    p3_vals = [p3_vals,p3];
end
plot(k_vals,p1_vals,k_vals,p2_vals,k_vals,p3_vals);
title('P values vs K');
xlabel('K value');
ylabel('P values');
legend('p1', 'p2', 'p3');


function [clusters,iter_count,square_sum,p1,p2,p3, centroids] = kmeans(X,Y,israndom,num_clusters)   
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
[a,b] = size(Y);
same_class = 0;
same_cluster = 0;
diff_class = 0;
diff_cluster = 0;
for i = 1:a
    for j = i+1:a
        if Y(i,:) == Y(j,:)
            same_class = same_class +1;
            if clusters(i,:) == clusters(j,:)
                same_cluster = same_cluster + 1;
            end
        else
            diff_class = diff_class + 1;
            if clusters(i,:) ~= clusters(j,:)
                diff_cluster = diff_cluster +1;
            end
        end       
    end
end
p1 = (same_cluster/same_class) * 100;
p2  = (diff_cluster/diff_class) * 100;
p3 = (p1+p2)/2;
return
end


