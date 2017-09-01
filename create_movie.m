function create_movie(image_names, movie_name)
% create video for predictions
writerObj = VideoWriter(movie_name);
writerObj.FrameRate=2;

open(writerObj);
for k= 1:length(image_names)
  filename = sprintf(image_names{k});
  thisimage = imread(filename);
  img_size = size(thisimage);
  if img_size(1)==435 && img_size(2)==441 && img_size(3)==3
    writeVideo(writerObj, thisimage);
  end
end
close(writerObj);