
data = complete;

% data = missing;

% data = reconstructed;

% data = difference;


line_idx = 6;
dt=0.004;  % sampling interval  Kerry: 0.004
dx=12.5; % trace interval   Kerry: 12.5


[S,f,k] = fk(data(:, :, line_idx), dt, dx, 9); 

figure('Position', [100, 100, 700, 600]);
imagesc(k,f,log(S));
% imagesc(k,f,S);
set(gca, 'FontSize', 28);
xlabel('Wavenumber (c/m)', 'FontName', 'Arial', 'FontWeight','bold');
ylabel('Frequency (Hz)', 'FontName', 'Arial', 'FontWeight','bold');



% 设置色图范围
min_value = 5.6; % 替换为你希望的最小值  kerry
max_value = 12;  % 替换为你希望的最大值 
clim([min_value, max_value]);
colormap('turbo');
% colorbar;

ylim([0, 125]);
xlim([-0.04, 0.04]); % kerry