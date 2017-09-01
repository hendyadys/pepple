% make movies
% % normal pred over epochs
% num_epochs = 200;
% image_names = cell(num_epochs);
% for i=1:num_epochs
%     image_names{i}= sprintf('./runVertical2/figs/avg_pred_i100_e%d.png', i);
% end
% create_movie(image_names, 'pred_movie.avi')

% % rotation movies
% num_epochs = 379;
% for deg=5:5:20
%     image_names = cell(num_epochs,1);
%     for i=1:num_epochs
% %         image_names{i} = sprintf('./runs/runVertical/figs/combined/recombined_pred_i100_e%d_rotation_m%d.png', i, deg);
%         image_names{i} = sprintf('./runs/runVertical2/figs/combinedOld/recombined_pred_i100_e%d_rotation_m%d.png', i, deg);
%     end
%     create_movie(image_names, sprintf('rotation_deg%d_lowLR.avi', deg))
% end
% 
% % translation movies
% num_epochs = 100;
% for deg=[20,50,100,200,300]
%     image_names = cell(num_epochs,1);
%     for i=1:num_epochs
%         image_names{i} = sprintf('./runs/runVertical/figs/combined/recombined_pred_i100_e%d_translation_m%d.png', i, deg);
% %         image_names{i} = sprintf('./runs/runVertical/figs/combined/recombined_pred_i100_e%d_rotation_m%d.png', i, deg);
%     end
%     create_movie(image_names, sprintf('translation_pixels_%d.avi', deg))
% end

% qlearning movies
num_preds = 1000;
image_names = cell(num_preds,1);
for i=1:num_preds
    image_names{i} = sprintf('./qLearning/DeRuyter-Inflamed_20170710mouse8_Day1_Right_807.json_i%d.png', i);
end
create_movie(image_names, 'qlearning_static.avi')

num_preds = 525;
image_names2 = cell(num_preds-10,1);
for i=1:num_preds
    image_names2{i} = sprintf('./qLearning/DeRuyter-Inflamed_20170710mouse8_Day1_Right_807.json_dyn_i%d.png', i+10);
end
create_movie(image_names2, 'qlearning_dynamic.avi')