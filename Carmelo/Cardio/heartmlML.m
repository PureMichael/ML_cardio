clear all
close all
clc
% 
m=xlsread('CTG_ML.xlsx');
% plotmatrix(m(:,1:end))
c3=find(m(:,end)==3);
c2=find(m(:,end)==2);
tot=1:length(m(:,end));
tot=tot';
tot(c2)=0;
tot(c3)=0;
tot(tot==0)=[];
c1=randperm(length(tot),length(c3))';
c1=tot(c1);
ev=[c1; c2(randperm(length(c2),length(c3))); c3];
n=m(ev,:);

[mm,nm]=size(n);
for ii=1:nm-1
    n(:,ii)=(n(:,ii)-min(n(:,ii)));
    n(:,ii)=(1/max(n(:,ii)))*n(:,ii);
end
figure()
ax1=plot(nan(2,length(n(:,1))),'k.','markersize',20);
for kk=1:length(ev)
    if round(n(kk,end),1) == 1
        color=[1 0 0];
    elseif  round(n(kk,end),1) == 2
        color = [0 1 0];
    else
        color= [0 0 1];
    end
    set(ax1(kk),'XData',(1:23),'YData',n(kk,1:end),'color',color)
end
% xlswrite('EvenDistribution.xls', n)
figure()
plotmatrix(n)
% axis tight
% a=sum(n(1:176,1:end-2))/176;
% b=sum(n(177:176*2+2,1:end-2))/176;
% c=sum(n(176*2+2:end,1:end-2))/176;
% figure()
% hold on
% plot(a,'r.','markersize',20)
% plot(b,'g.','markersize',20)
% plot(c,'b.','markersize',20)
% smalln=n(:,[1 2 8 10 13 22 23]);
smalln=n(:,[1 4 8 11 12 14 17 18 19 22 23]);
% xlswrite('EvenDistribution_LessVars.xls', smalln)
figure()
plotmatrix(smalln(:,1:end-2))




