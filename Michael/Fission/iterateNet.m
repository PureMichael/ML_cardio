iterations = 1000;
accuracy = zeros(1,iterations);
results = zeros(2,2,iterations);


for m = 1:iterations
    [imdsTest, trainedNet,state] = makeCNN();
    [accuracy(m),results(:,:,m)]= testNet(imdsTest,trainedNet,state);
    save('iterations.mat','accuracy','results')
    m
end

accuracy(1);
accuracy(2);
results(:,:,1);
results(:,:,2);