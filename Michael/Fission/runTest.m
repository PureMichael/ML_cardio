%% Run net on test data
tic

results = zeros(2);
[testx,testy] = size(imdsTest.Labels);
for m = 1:testx
   label = classify(trainednet,readimage(imdsTest,m));
   correctLabel = imdsTest.Labels(m);
   
   if imdsTest.Labels(m) == state(1)
    if label == correctLabel
       results(1,1) = results(1,1)+1;
    else
       results(1,2) = results(1,2)+1;
    end
   else
    if label == correctLabel
       results(2,2) = results(2,2)+1;
    else
       results(2,1) = results(2,1)+1;
    end   
   end  
end

toc
results
accuracy = (results(1,1)+results(2,2))/sum(sum(results))