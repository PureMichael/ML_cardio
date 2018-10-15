function [accuracy,results] = testNet(imdsTest,trainednet,state)
%tic

results = zeros(2);
%idx_yes =zeros(1,621);
%idx_no = zeros(1,621);
%yescount = 1;
%nocount = 1;
[testx,~] = size(imdsTest.Labels);
for m = 1:testx
   label = classify(trainednet,readimage(imdsTest,m));
   correctLabel = imdsTest.Labels(m);
   
   if imdsTest.Labels(m) == state(1)
    if label == correctLabel
       results(1,1) = results(1,1)+1;
       %idx_yes(yescount) = m;
       %yescount = yescount +1;
    else
       results(1,2) = results(1,2)+1;
    end
   else
    if label == correctLabel
       results(2,2) = results(2,2)+1;
       %idx_no(nocount)= m;
       %nocount = nocount + 1;
    else
       results(2,1) = results(2,1)+1;
    end   
   end  
end

%toc
results;
accuracy = (results(1,1)+results(2,2))/sum(sum(results));
end

