

res = kmeans(num(:,23),3);


%disp('kmeans')

score = 0;
for m = 1:x
   if res(m) == num(m,23)
      score = score+1; 
   end
end

acc = score/x