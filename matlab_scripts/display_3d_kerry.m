
data_v = complete;

% data_v = missing;

% data_v = reconstructed;

% data_v = difference;

dt = 0.004;  % Kerry: 0.004

figure('Position',[400 400 755 700], 'Color','w');


data_v_max = max(abs(complete(:)));
data_v = data_v / data_v_max;

[tline, xxline, inline] = size(data_v);

drr_plot3d(data_v, dt, [1, xxline - 1, 1]); % kerry
set(gca,'Linewidth',2,'Fontsize',44, 'FontName', 'Arial');
xlabel('','Fontsize',36, 'FontName', 'Arial', 'FontWeight','bold');
ylabel('','Fontsize',36, 'FontName', 'Arial', 'FontWeight','bold');
zlabel('','Fontsize',36, 'FontName', 'Arial', 'FontWeight','bold');

pbaspect([2 1.5 2]);
colormap(cseis);
% colorbar;

clim([-0.75,0.75])
yticks([0, 10, 20])
shading interp; 
% axis off;