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
y= y-2;
%% Get cluster locations and SDevs for the 3 clusters
cluster_mean = zeros(3,y);
cluster_sd = zeros(3,y);

cluster_mean(1,:) = mean(num(i1train,1:21));
cluster_mean(2,:) = mean(num(i2train,1:21));
cluster_mean(3,:) = mean(num(i3train,1:21));
cluster_sd(1,:) = std(num(i1train,1:21));
cluster_sd(2,:) = std(num(i2train,1:21));
cluster_sd(3,:) = std(num(i3train,1:21));

%% Get distance metric for each training point
dist_metrict = zeros(x,4);
z1t = zeros(x,21);
z2t = zeros(x,21);
z3t = zeros(x,21);
for n = 1:x  
    diff1 = data_train(n,1:21) - cluster_mean(1,:);
    z1t(n,:) = diff1./cluster_sd(1,:);   
    diff2 = data_train(n,1:21) - cluster_mean(2,:);
    z2t(n,:) = diff2./cluster_sd(2,:);
    z2t(n,6) = 0.0;  
    diff3 = data_train(n,1:21) - cluster_mean(3,:);
    z3t(n,:) = diff3./cluster_sd(3,:);
end


res_perct = zeros(1,y);
for p = 1:y
for t = 1:x
    dist_metrict(t,1) = norm(z1t(t,p));
    dist_metrict(t,2) = norm(z2t(t,p));
    dist_metrict(t,3) = norm(z3t(t,p));
    minidx = find(dist_metrict(t,1:3)== min(dist_metrict(t,1:3)));
    dist_metrict(t,4) = minidx;
    
end


    %% Measure accuracy
    results = zeros(3,3);

    [results(1,1),y11] = size(find(dist_metrict(:,4) == 1 & data_train(:,23)==1));
    [results(2,2),y22] = size(find(dist_metrict(:,4) == 2 & data_train(:,23)==2));
    [results(3,3),y33] = size(find(dist_metrict(:,4) == 3 & data_train(:,23)==3));

    [results(1,2),y12] = size(find(dist_metrict(:,4) == 1 & data_train(:,23)==2));
    [results(1,3),y13] = size(find(dist_metrict(:,4) == 1 & data_train(:,23)==3));
    [results(2,1),y21] = size(find(dist_metrict(:,4) == 2 & data_train(:,23)==1));
    [results(2,3),y23] = size(find(dist_metrict(:,4) == 2 & data_train(:,23)==3));
    [results(3,1),y31] = size(find(dist_metrict(:,4) == 3 & data_train(:,23)==1));
    [results(3,2),y32] = size(find(dist_metrict(:,4) == 2 & data_train(:,23)==2));

    results;
    res_perct(p) = (results(1,1)+results(2,2)+results(3,3))/x;

end
res_perct = zeros(1,21);
res_perct(7) = 1;
res_perct(10)=0.00;


%% Get distance metric for each point
[xt,yt] = size(data_test);
dist_metric = zeros(xt,4);
z1 = zeros(xt,21);
z2 = zeros(xt,21);
z3 = zeros(xt,21);

for n = 1:xt 
    diff1 = data_test(n,1:21) - cluster_mean(1,:);
    z1(n,:) = diff1./cluster_sd(1,:);
    
    diff2 = data_test(n,1:21) - cluster_mean(2,:);
    z2(n,:) = diff2./cluster_sd(2,:);
    z2(n,6) = 0.0;
    
    diff3 = data_test(n,1:21) - cluster_mean(3,:);
    z3(n,:) = diff3./cluster_sd(3,:);
end

%% Categorize by shortest distance
res_perc = zeros(1,y);
for p = 1:1
for t = 1:xt
    dist_metric(t,1) = norm(z1(t,:).*res_perct);
    dist_metric(t,2) = norm(z2(t,:).*res_perct);
    dist_metric(t,3) = norm(z3(t,:).*res_perct);
    minidx = find(dist_metric(t,1:3)== min(dist_metric(t,1:3)));
    dist_metric(t,4) = minidx;
    
end


%% Measure accuracy
results = zeros(3,3);

[results(1,1),y11] = size(find(dist_metric(:,4) == 1 & data_test(:,23)==1));
[results(2,2),y22] = size(find(dist_metric(:,4) == 2 & data_test(:,23)==2));
[results(3,3),y33] = size(find(dist_metric(:,4) == 3 & data_test(:,23)==3));

[results(1,2),y12] = size(find(dist_metric(:,4) == 1 & data_test(:,23)==2));
[results(1,3),y13] = size(find(dist_metric(:,4) == 1 & data_test(:,23)==3));
[results(2,1),y21] = size(find(dist_metric(:,4) == 2 & data_test(:,23)==1));
[results(2,3),y23] = size(find(dist_metric(:,4) == 2 & data_test(:,23)==3));
[results(3,1),y31] = size(find(dist_metric(:,4) == 3 & data_test(:,23)==1));
[results(3,2),y32] = size(find(dist_metric(:,4) == 2 & data_test(:,23)==2));

results
res_perc(p) = (results(1,1)+results(2,2)+results(3,3))/xt;
end
res_perc(1)



%% Plot cloud cluster

%scatter3(dist_metric(:,1),dist_metric(:,2),dist_metric(:,3),4,dist_metric(:,4));
%xlabel('Distance from Category1');
%ylabel('Distance from Category2');
%zlabel('Distance from Category3');

toc