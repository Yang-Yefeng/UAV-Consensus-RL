clc;clear;

%% attitude
data = csvread('att_cost_surface.csv', 1, 0);

A = data(:, 1) * 180 / pi;
T = data(:, 2);

r_rl_no_obs = data(:, 3);
r_fntsmc_no_obs = data(:, 4);
r_rl_obs = data(:, 5);
r_fntsmc_obs = data(:, 6);

figure()
set(gca, 'LooseInset', [0.01, 0.01, 0.01, 0.01]);
[x, y] = meshgrid(linspace(min(A), max(A), 50), linspace(min(T), max(T), 50));

%% plot RL-NO-OBS
z_rl_no_obs = griddata(A, T, r_rl_no_obs, x, y);
mesh(x, y, z_rl_no_obs, 'facecolor', 'cyan', 'EdgeColor', 'none'); hold on;

%% plot FNTSMC-NO-OBS
z_fntsmc_no_obs = griddata(A, T, r_fntsmc_no_obs, x, y);
mesh(x, y, z_fntsmc_no_obs, 'facecolor', 'green', 'EdgeColor', 'none'); hold on;

%% plot RL-OBS
z_rl_obs = griddata(A, T, r_rl_obs, x, y);
mesh(x, y, r_rl_obs, 'facecolor', 'red', 'EdgeColor', 'none'); hold on;

%% plot FNTSMC-OBS
z_fntsmc_obs = griddata(A, T, r_fntsmc_obs, x, y);
mesh(x, y, z_fntsmc_obs, 'facecolor', 'blue', 'EdgeColor', 'none'); hold on;


% legend('smc', 'rl');
% title('attitude');
grid on;
