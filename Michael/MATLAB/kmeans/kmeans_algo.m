% res = kmeans(num(:,23),3);
% 
% 
% %disp('kmeans')
% 
% score = 0;
% for m = 1:x
%    if res(m) == num(m,23)
%       score = score+1; 
%    end
% end
% 
% acc = score/x
tic
clear
load('heartdata75.mat');
[x,y] = size(data_train);

%% Get cluster locations and SDevs for the 3 clusters
cluster_mean = zeros(3,y);
cluster_sd = zeros(3,y);

for m = 1:y
    cluster_mean(1,m) = mean(num(i1train,m));
    cluster_mean(2,m) = mean(num(i2train,m));
    cluster_mean(3,m) = mean(num(i3train,m));
    
    cluster_sd(1,m) = std(num(i1train,m));
    cluster_sd(2,m) = std(num(i2train,m));
    cluster_sd(3,m) = std(num(i3train,m));

end

%% Get distance metric for each point
[xt,yt] = size(data_test);
dist_metric = zeros(xt,4);

for n = 1:xt
    
    diff1 = data_test(n,1:21) - cluster_mean(1,:);
    z1 = diff1./cluster_sd(1,:);
    dist_metric(n,1) = norm(z1);
    
    diff2 = data_test(n,1:21) - cluster_mean(2,:);
    z2 = diff2./cluster_sd(2,:);
    z2(6) = 0.0;
    dist_metric(n,2) = norm(z2);
    
    diff3 = data_test(n,1:21) - cluster_mean(3,:);
    z3 = diff3./cluster_sd(3,:);
    dist_metric(n,3) = norm(z3);
    
    minidx = find(dist_metric(n,1:3)== min(dist_metric(n,1:3)));
    dist_metric(n,4) = minidx;
end

%% Plot cloud cluster

scatter3(dist_metric(:,1),dist_metric(:,2),dist_metric(:,3),4,dist_metric(:,4));
xlabel('Distance from Category1');
ylabel('Distance from Category2');
zlabel('Distance from Category3');


%% Measure accuracy

correct_idx = find(dist_metric(:,4) == data_test(:,23));
size(correct_idx)


