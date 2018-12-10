close all
clear all
clc
norm_them=0;
wavelength=csvread('GT_Test.csv',0,2,[0 2 0 171]);
M=csvread('GT_Test.csv',1,1);
M(:,1)=M(:,1)+1;
cats=unique(M(:,1));

if norm_them ==1
%     M(:,2:end)=0.1+ M(:,2:end)-min(M(:,2:end),[],2);
    M(:,2:end)=M(:,2:end)./max(M(:,2:end),[],2);
end 


%% Plot the average of all categories 
figure();hold on
for ii=1:length(cats)
    d(cats(ii)).data=M(M(:,1)==cats(ii),2:end);
    plot(mean(d(cats(ii)).data),'linewidth',2)
    legend_entries{ii}=strcat('Category ',num2str(ii-1));
end 
legend(legend_entries)
axis tight
box on
%% Plot n random spectra from each category on its own figure
n=20;
for ii=1:length(cats)
    ii    
    figure();
    plot_these = randi(length(d(ii).data(:,1)),n,1);
    line = plot(0,nan(length(plot_these),1));
    xd=(1:length(d(ii).data(1,:)))-1;
    for jj=1:length(plot_these)
        if mod(jj,10)==0
            clc
            round(100*jj/length(line))
        end 
        set(line(jj),'XData',xd,'YData',d(ii).data(plot_these(jj),:))
    end 
    title(legend_entries{ii})
    axis tight
end 

%%
figure()
pt = randi(length(M(:,1)),5,1);
hold on
for ii=1:length(pt)
    plot(M(pt(ii),2:end),'linewidth',2)
    le{ii}=num2str(M(pt(ii),1)-1);
end 
legend(le);axis tight