close all
clear all
clc
writeit=1;
t=0:(2*pi)/175:2*pi;
M=[];
figure()
hold on
ii=0;
% for phi=0:pi/4:7*pi/4
for phi=0:(pi/4)/7:pi/4
    y = sin(t+phi)+rand(randi([500,6000],1),length(t))/10;
    y = [ii*ones(length(y(:,1)),1) ii*ones(length(y(:,1)),1) y];
    plot(t,y(1,3:end))
    M=[M; y];
    ii=ii+1;
end
figure();hist(M(:,1))
axis tight
if writeit==1
    csvwrite('MatLabData.csv',M)
end