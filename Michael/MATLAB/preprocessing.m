%num = xlsread('CTG.xls','MLData');
load('MLdata.mat');

tic
[x y] = size(num);

len_train = round(0.9*x);
len_test = x - len_train;

data_train = num(1:len_train,1:21);
data_test = num(len_train+1:x,1:21);

labels = ["baseline value",'accelerations','foetal movement','uterine contractions','light decelerations',...
    'severe decelerations','prolongued decelerations','% time with abnormal shortterm var',...
    'mean time shortterm var',"% time with abnormal longterm var",'mean time longterm var',...
    'histogram width','low freq of hist','high freq of hist','num of peaks','num of zeros',...
    'hist mode','hist mean','hist median','hist variance','hist tendency',"class","Category: NSP"];
    

%plot(data_train)
i1all = find(num(:,23)==1);
i2all = find(num(:,23)==2);
i3all = find(num(:,23)==3);
s1all = size(i1all);
s2all = size(i2all);
s3all = size(i3all);
p1all = s1all(1)/(s1all(1)+s2all(1)+s3all(1));
p2all = s2all(1)/(s1all(1)+s2all(1)+s3all(1));
p3all = s3all(1)/(s1all(1)+s2all(1)+s3all(1));


close all
yt = 23;
for m = 1:yt
    %figure(m)
    subplot(5,5,m);
    hold on
    title(labels(m));
    histogram(num(i1all,m));
    histogram(num(i2all,m));
    histogram(num(i3all,m));
    hold off
    
end


% i1train = find(data_train(:,23)==1);
% i2train = find(data_train(:,23)==2);
% i3train = find(data_train(:,23)==3);
% s1train = size(i1train);
% s2train = size(i2train);
% s3train = size(i3train);
% p1train = s1train(1)/(s1train(1)+s2train(1)+s3train(1));
% p2train = s2train(1)/(s1train(1)+s2train(1)+s3train(1));
% p3train = s3train(1)/(s1train(1)+s2train(1)+s3train(1));
% 
% 
% s1test = size(find(data_test(:,23)==1));
% s2test = size(find(data_test(:,23)==2));
% s3test = size(find(data_test(:,23)==3));

%p = data_train(:,23);

toc