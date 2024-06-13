clc;clear;

%% attitude
data = csvread('att_cost_surface.csv', 1, 0);

A = data(:, 1) * 180 / pi;
T = data(:, 2);
r1 = data(:, 3);
r2 = data(:, 4);

figure()
set(gca, 'LooseInset', [0.01, 0.01, 0.01, 0.01]);
[x, y] = meshgrid(linspace(min(A), max(A), 50), linspace(min(T), max(T), 50));

% plot smc
z1 = griddata(A, T, r1, x, y);
mesh(x, y, z1, 'facecolor', 'b', 'EdgeColor', 'none'); hold on;

% plot rl
z2 = griddata(fx, fy, fz, x, y);
mesh(x, y, z2, 'facecolor', 'r', 'EdgeColor', 'none'); hold on;
% legend('smc', 'rl');
% title('attitude');
grid on;
