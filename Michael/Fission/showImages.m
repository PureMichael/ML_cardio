for m = 1:525
  idx = idx_yes(m);
  subplot(25,25,m);
  imshow(readimage(imdsTest,idx));
end