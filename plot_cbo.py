import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
import seaborn as sns


# =============== Plotting Functions for Modeling Error ================= #


def cnt_plt(a1, a2, value, n, title):
    """ Contour plots for 2D input space """
    fig, ax = plt.subplots()
    h = ax.contourf(a1, a2, value.reshape(len(a1), len(a2)), n)
    cbar = plt.colorbar(h)
    cbar.ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_ylabel('x2')
    ax.set_xlabel('x1')
    plt.show()


def surf_plt(a1, a2, value, title):
    """ Surface plots for 2D input space """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(a1, a2, value.reshape(len(a1), len(a2)), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Evaluations')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def plot_err_bars(mse_df, n_classes, n_partitions, hetero_ratio, file_name):
    fig, ax = plt.subplots(sharex='all')
    ax.set_title(str(n_classes) + " Classes, " + str(n_partitions)
                 + " partitions, hetero_ratio = " + str(hetero_ratio))
    ax.set_xlabel('Methods', fontsize=20, fontweight='bold')
    ax.set_ylabel('Modeling MSE', fontsize=20, fontweight='bold')
    sns.boxplot(data=mse_df, ax=ax)
    fig.tight_layout()
    img_file_name = file_name + ".png"
    plt.savefig(img_file_name)
    plt.show()


def plot_rewards(rewards_desc1, rewards_desc2, iterations, ymax, label1, label2, file_name):
    fig, ax = plt.subplots()

    # Reward 1
    ax.plot(range(iterations), rewards_desc1.loc['mean'], '-', color='tab:orange', label=label1)
    lower1 = rewards_desc1.loc['mean'] - rewards_desc1.loc['std']
    upper1 = rewards_desc1.loc['mean'] + rewards_desc1.loc['std']

    # Reward 2
    ax.plot(range(iterations), rewards_desc2.loc['mean'], '-', color='tab:blue', label=label2)
    lower2 = rewards_desc2.loc['mean'] - rewards_desc2.loc['std']
    upper2 = rewards_desc2.loc['mean'] + rewards_desc2.loc['std']

    # Plotting Confidence Band
    ax.fill_between(range(iterations), lower1, upper1, alpha=0.2, color='tab:orange')
    ax.fill_between(range(iterations), lower2, upper2, alpha=0.2, color='tab:blue')

    # Title, Labels and Legend of the plot
    ax.set_title('Max Reward Observed vs Iterations,  Objective max = %f' % ymax)
    ax.set_ylabel('Max Function Evaluation')
    ax.set_xlabel('Iterations')
    plt.legend()
    fig.tight_layout()
    img_file_name = file_name + ".png"
    plt.savefig(img_file_name)
    plt.show()

    print("Reward 1 maximum mean = ", max(rewards_desc1.loc['mean']),
          " and Reward 2 maximum mean = ", max(rewards_desc2.loc['mean']))


def plot3d_mean_std(x1, z1, mean, std, z_label1, z_label2, n):
    # set up a figure twice as tall as it is wide
    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(2, 2, 1)
    cp = ax.contourf(x1, z1, mean.reshape(n, n))
    cbar = fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    cbar.ax.set_ylabel(z_label1)

    ax1 = fig.add_subplot(2, 2, 2, projection='3d')
    surf = ax1.plot_surface(x1, z1, mean.reshape(n, n), cmap=cm.coolwarm, linewidth=0)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel(z_label1)

    ax2 = fig.add_subplot(2, 2, 3)
    cp1 = ax2.contourf(x1, z1, std.reshape(n, n))
    cbar1 = fig.colorbar(cp1)  # Add a colorbar to a plot
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    cbar1.ax.set_ylabel(z_label2)

    ax3 = fig.add_subplot(2, 2, 4, projection='3d')
    surf1 = ax3.plot_surface(x1, z1, std.reshape(n, n), cmap=cm.coolwarm, linewidth=0)
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_zlabel(z_label2)

    plt.tight_layout()
    plt.show()