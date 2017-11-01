clear
% plot epoch
% read table and visualize
train_losses = readtable('C:\\Users\\yue\\Documents\\keras-frcnn-master\\accell_weights\\weights_32_32_noisy\\epoch_log.txt');
train_losses = table2array(train_losses);
data_size = size(train_losses);
train_losses_better = readtable('C:\\Users\\yue\\Documents\\keras-frcnn-master\\accell_weights\\weights_32_32_noisy\\epoch_log_noisy.txt');
train_losses_better = table2array(train_losses_better);
data_size_better = size(train_losses_better);

figure(1)
% plot(1:data_size(1), train_losses(:,2:end-1)); grid on;   % includes accuracy
plot(1:data_size(1), train_losses(:,2:end-2)); grid on;
title('losses over iterations')
xlabel('iter')
ylabel('loss')
col_names = {'rpn-cls', 'rpn-regr', 'detector-cls', 'detector-regr', 'detector-acc'};
legend(col_names);
ylim([0,.2])

figure(2)
plot(1:data_size(1), train_losses(:,end)); grid on;
title('total loss')
xlabel('iter')
ylabel('loss')
ylim([0,.2])

figure(3)
plot(1:data_size(1), train_losses(:,end-1)); grid on;
title('prediction accuracy over iterations')
xlabel('iter')
ylabel('accuracy')
legend(col_names(end));

% compare losses for different anchor_box_scales (orig vs better iou)
figure(4)
num_epochs = min(data_size(1), data_size_better(1));
plot(1:num_epochs, [train_losses(1:num_epochs,end), train_losses_better(1:num_epochs,end)]); grid on;
title('total loss')
xlabel('iter')
ylabel('loss')
legend('Original scale', 'Better Scale');

figure(5)
plot(1:num_epochs, [train_losses(1:num_epochs,end-1), train_losses_better(1:num_epochs,end-1)]); grid on;
title('prediction accuracy over iterations')
xlabel('iter')
ylabel('accuracy')
legend('Original scale', 'Better Scale');

% plot iteration smoothed
% train_losses = readtable('C:\\Users\\yue\\Documents\\pepple\\1-process-data\\iter_log.txt');
% train_losses = readtable('C:\\Users\\yue\\Documents\\keras-frcnn-master\\iter_log.txt');
period = 20;
rolling_mean = movmean(train_losses, period);
size(rolling_mean)

figure(1)
plot(1:data_size(1), rolling_mean(:,2:end-1)); grid on;
title('losses over iterations')
xlabel('iter')
ylabel('loss')
col_names = {'rpn-cls', 'rpn-regr', 'detector-cls', 'detector-regr', 'detector-acc'};
legend(col_names);
ylim([0,.5])

figure(2)
plot(1:data_size(1), sum(rolling_mean(:,2:end-1),2)); grid on;
title('total loss')
xlabel('iter')
ylabel('loss')
ylim([0,1])

figure(3)
plot(1:data_size(1), rolling_mean(:,end)); grid on;
title('prediction accuracy over iterations')
xlabel('iter')
ylabel('accuracy')
legend(col_names(end));

% % epoch data
% figure(3)
% % epoch_losses = readtable('C:\\Users\\yue\\Documents\\overview\\1-process-data\\epoch_log.txt');
% epoch_losses = textscan('C:\\Users\\yue\\Documents\\overview\\1-process-data\\epoch_log.txt', '%f');
% epoch_losses = table2array(train_losses);