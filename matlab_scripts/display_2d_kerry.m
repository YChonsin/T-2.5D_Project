
data = complete;

% data = missing;

% data = reconstructed; 

% data = difference;

% 记录最大最小值
data = data / max(abs(complete(:)));

%% 2. 设置坐标轴
line = 6;   % the line number you want to display Kerry: 6 
dt = 0.004;  % Kerry: 0.004

data = data(:, :, line);
[nt, nx] = size(data);

t = (0:nt-1)*dt;               % 时间轴
x = 1:nx;                      % 道号

%% 3. 绘制二维地震剖面
figure('Color','w','Position',[100 100 640 640]);

imagesc(x,t,data);
colormap(cseis);        % 灰度显示
% colorbar;

%% 5. 美化
set(gca,'FontSize',28,'LineWidth',1.2);

set(gca,'YDir','reverse');   % 时间向下（地震常用显示方式）
set(gca,'XAxisLocation','top')  % x轴到上面
% set(gca,'TickDir','out')
% set(gca,'TickDir','out','Box','off')
xlabel('Trace','FontSize', 28, 'FontWeight', 'bold');
ylabel('Time (s)','FontSize', 28, 'FontWeight', 'bold');
yticks([0, 0.30, 0.60, 0.89]);   % kerry
yticklabels({'0','0.30','0.60','0.89'})

xticks([1, 60, 120, 180, 224]);
xticklabels({'1','60','120','180', '224'})
% ytickformat('%.2f');
set(get(gca,'Children'),'Interpolation','bilinear'); % 平滑
clim([-0.5, 0.5]) 