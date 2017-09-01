% read table and visualize
combined_dice = readtable('C:\\Users\\yue\\Documents\\pepple\\combined_dice.txt');
combined_dice = table2array(combined_dice);
data_size = size(combined_dice);

period = 20;
rolling_mean = movmean(combined_dice, period);
size(rolling_mean)

figure(1)
plot(1:data_size(1), rolling_mean(:,2:end)); grid on;
title('dice over iterations')
xlabel('iter')
ylabel('loss')
col_names = {'strip dice', 'whole image dice', 'whole image dice no padding'};
legend(col_names);

