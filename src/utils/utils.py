"""这里是prediction/utils, 用于绘图"""
import logging

import numpy as np
import pandas as pd
import time

from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec


def plot_enhanced_histogram_v2(ax, data, bins, plot_range, vertical=True, theme_color='red'):
    """Enhanced histogram plotting with layered colors

    Args:
        ax: matplotlib axis
        data: data to plot
        bins: histogram bins
        plot_range: plot range [min, max]
        vertical: if True, plot vertical histogram, else horizontal
        theme_color: theme color for the plot
    """
    # 计算kde
    kde_points = np.linspace(plot_range[0], plot_range[1], 200)
    kde = gaussian_kde(data)
    kde_values = kde(kde_points)

    # 计算直方图
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 获取暗色版本作为边框色
    dark_color = plt.get_cmap('Dark2')(plt.matplotlib.colors.to_rgba(theme_color)[0])

    if vertical:
        ax.fill_between(kde_points, kde_values, alpha=0.35, color=theme_color, zorder=1)
        ax.bar(bin_centers, hist, width=np.diff(bin_edges), alpha=0.35,
               color=theme_color, edgecolor=dark_color, linewidth=0.5, zorder=2)
        ax.plot(kde_points, kde_values, alpha=0.35, color=dark_color, linewidth=0.5, zorder=3)
    else:
        ax.fill_betweenx(kde_points, kde_values, alpha=0.35, color=theme_color, zorder=1)
        ax.barh(bin_centers, hist, height=np.diff(bin_edges), alpha=0.35,
                color=theme_color, edgecolor=dark_color, linewidth=0.5, zorder=2)
        ax.plot(kde_values, kde_points, alpha=0.35, color=dark_color, linewidth=0.5, zorder=3)


def plot_results_enhanced_v2(true_ddg, predictions, save_path, metrics=None, dataset=None, title="", theme_color='red'):
    """Using direct axes management approach with fixed layout issues

    Args:
        ...
        theme_color: color theme for the plot. Can be 'red', 'blue', 'purple', 'orange' or any valid color
    """
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': ['DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 12,
        'axes.linewidth': 2,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'font.weight': 'bold',
    })

    # 创建图形和坐标轴
    fig = plt.figure(figsize=(9, 9))

    main_ax = fig.add_axes((0.15, 0.15, 0.7, 0.7))
    top_ax = fig.add_axes((0.15, 0.85, 0.7, 0.15))
    right_ax = fig.add_axes((0.85, 0.15, 0.15, 0.7))

    plot_range = [-10, 10]

    # 计算统计指标
    if metrics:
        pcc = metrics.get('PCC', 0)
        rmse = metrics.get('RMSE', 0)
    else:
        pcc = np.corrcoef(true_ddg, predictions)[0, 1]
        rmse = np.sqrt(np.mean((np.array(true_ddg) - np.array(predictions)) ** 2))

    # 计算拟合直线
    slope, intercept = np.polyfit(true_ddg, predictions, 1)
    data_range_min = np.percentile(true_ddg, 0.2)
    data_range_max = np.percentile(true_ddg, 99.8)
    fit_x = np.array([data_range_min, data_range_max])
    fit_line = slope * fit_x + intercept

    # 设置显示范围和刻度
    main_ax.set_xlim(plot_range)
    main_ax.set_ylim(plot_range)
    top_ax.set_xlim(plot_range)
    right_ax.set_ylim(plot_range)

    major_ticks = np.arange(-10, 12, 2)
    minor_ticks = np.arange(-10, 11, 1)

    main_ax.set_xticks(major_ticks)
    main_ax.set_yticks(major_ticks)
    main_ax.set_xticks(minor_ticks, minor=True)
    main_ax.set_yticks(minor_ticks, minor=True)

    main_ax.tick_params(axis='both', which='major', length=6, width=2, direction='out')
    main_ax.tick_params(axis='both', which='minor', length=4, width=1.5, direction='out')

    main_ax.grid(True, which='major', linestyle='--', alpha=0.5, color='gray', zorder=1)
    main_ax.grid(True, which='minor', linestyle='--', alpha=0.2, color='gray', zorder=1)

    # 绘制散点和直线
    main_ax.scatter(true_ddg, predictions, alpha=0.6, color=theme_color, s=50, zorder=3)
    # main_ax.plot(plot_range, plot_range, '--', color='black', alpha=0.8, linewidth=1.5, zorder=2)
    main_ax.plot(fit_x, fit_line, '-', color=theme_color, alpha=0.8, linewidth=2.0, zorder=2)

    # 直方图
    bins = np.arange(-10, 11, 1)
    plot_enhanced_histogram_v2(top_ax, true_ddg, bins, plot_range, vertical=True, theme_color=theme_color)
    plot_enhanced_histogram_v2(right_ax, predictions, bins, plot_range, vertical=False, theme_color=theme_color)

    # 隐藏直方图刻度
    top_ax.set_xticks([])
    top_ax.set_yticks([])
    right_ax.set_xticks([])
    right_ax.set_yticks([])

    # 控制边框显示
    main_ax.spines['top'].set_visible(False)
    main_ax.spines['right'].set_visible(False)
    top_ax.spines['top'].set_visible(False)
    top_ax.spines['right'].set_visible(False)
    top_ax.spines['left'].set_visible(False)
    right_ax.spines['top'].set_visible(False)
    right_ax.spines['right'].set_visible(False)
    right_ax.spines['bottom'].set_visible(False)

    # 准备文本
    dataset_name = f"{dataset.upper()}" if dataset else ""
    dataset_text = f"{dataset_name}\n"
    metrics_text = f"RMSE: {rmse:.3f}\nPCC   : {pcc:.3f}"
    # 处理截距的符号
    intercept_str = f"- {abs(intercept):.3f}" if intercept < 0 else f"+ {intercept:.3f}"
    equation_text = f"y = {slope:.3f}x {intercept_str}"

    # 添加文本标注
    main_ax.text(0.02, 0.98, dataset_text,
                 transform=main_ax.transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 fontsize=30,
                 fontweight='bold',
                 fontstyle='italic',
                 family='DejaVu Serif',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                           alpha=0,
                           edgecolor='none'),
                 zorder=5)

    main_ax.text(0.02, 0.91, metrics_text,
                 transform=main_ax.transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 fontsize=18,
                 fontweight='bold',
                 fontstyle='italic',
                 family='DejaVu Serif',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                           alpha=0,
                           edgecolor='none'),
                 zorder=5)

    main_ax.text(0.02, 0.81, equation_text,
                 transform=main_ax.transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 fontsize=18,
                 fontweight='bold',
                 fontstyle='italic',
                 family='DejaVu Serif',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                           alpha=0,
                           edgecolor='none'),
                 zorder=5)

    # 添加标签
    main_ax.set_xlabel('True', fontsize=20, fontweight='bold')
    main_ax.set_ylabel('Prediction', fontsize=20, fontweight='bold')

    # 保存图形
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def plot_enhanced_histogram_v1(ax, data, bins, plot_range, vertical=True):
    """Enhanced histogram plotting with layered colors

    Args:
        ax: matplotlib axis
        data: data to plot
        bins: histogram bins
        plot_range: plot range [min, max]
        vertical: if True, plot vertical histogram, else horizontal
    """
    # 计算kde
    kde_points = np.linspace(plot_range[0], plot_range[1], 200)
    kde = gaussian_kde(data)
    kde_values = kde(kde_points)

    # 计算直方图
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if vertical:
        # 绘制kde曲线填充
        ax.fill_between(kde_points, kde_values, alpha=0.35, color='red', zorder=1)

        # 绘制直方图
        ax.bar(bin_centers, hist, width=np.diff(bin_edges), alpha=0.35,
               color='red', edgecolor='darkred', linewidth=0.5, zorder=2)

        # 绘制kde曲线
        # ax.plot(kde_points, kde_values, color='darkred', linewidth=2, zorder=3)
        ax.plot(kde_points, kde_values, alpha=0.35, color='darkred', linewidth=0.5, zorder=3)
    else:
        # 水平方向的直方图和kde
        ax.fill_betweenx(kde_points, kde_values, alpha=0.35, color='red', zorder=1)
        # ax.fill_betweenx(kde_points, kde_values, alpha=0.3, color='salmon', zorder=1)

        ax.barh(bin_centers, hist, height=np.diff(bin_edges), alpha=0.35,
                color='red', edgecolor='darkred', linewidth=0.5, zorder=2)

        ax.plot(kde_values, kde_points, alpha=0.35, color='darkred', linewidth=0.5, zorder=3)

def plot_results_enhanced_v1(true_ddg, predictions, save_path, metrics=None, dataset=None, title=""):
    """Using direct axes management approach with fixed layout issues"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': ['DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 12,
        'axes.linewidth': 2,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'font.weight': 'bold',  # 设置全局字体为粗体
    })

    # 创建图形和坐标轴
    fig = plt.figure(figsize=(9, 9))

    main_ax = fig.add_axes((0.15, 0.15, 0.7, 0.7))
    top_ax = fig.add_axes((0.15, 0.85, 0.7, 0.15))
    right_ax = fig.add_axes((0.85, 0.15, 0.15, 0.7))

    # 数据范围计算:刻度自适应
    # all_values = np.concatenate([true_ddg, predictions])
    # min_val = np.floor(min(all_values))
    # max_val = np.ceil(max(all_values))
    # margin = (max_val - min_val) * 0.1
    # plot_range = [min_val - margin, max_val + margin]
    # 或使用固定的绘图范围:固定刻度
    plot_range = [-10, 10]

    # 计算统计指标
    if metrics:
        pcc = metrics.get('PCC', 0)
        rmse = metrics.get('RMSE', 0)
    else:
        pcc = np.corrcoef(true_ddg, predictions)[0, 1]
        rmse = np.sqrt(np.mean((np.array(true_ddg) - np.array(predictions)) ** 2))

    # 计算拟合直线
    slope, intercept = np.polyfit(true_ddg, predictions, 1)

    # 为拟合直线计算一个更合适的范围，基于数据的实际分布
    data_range_min = np.percentile(true_ddg, 0.2)  # 使用1%分位数
    data_range_max = np.percentile(true_ddg, 99.8)  # 使用99%分位数
    fit_x = np.array([data_range_min, data_range_max])
    fit_line = slope * fit_x + intercept

    # 设置显示范围
    main_ax.set_xlim(plot_range)
    main_ax.set_ylim(plot_range)
    top_ax.set_xlim(plot_range)
    right_ax.set_ylim(plot_range)

    # 创建刻度:自适应
    # major_ticks = np.arange(np.floor(plot_range[0]), np.ceil(plot_range[1]) + 1, 2)
    # minor_ticks = np.arange(np.floor(plot_range[0]), np.ceil(plot_range[1]) + 1, 1)
    # 创建固定的刻度
    major_ticks = np.arange(-10, 12, 2)  # 偶数刻度，从-10到10
    minor_ticks = np.arange(-10, 11, 1)  # 整数刻度，作为次刻度

    main_ax.set_xticks(major_ticks)
    main_ax.set_yticks(major_ticks)
    main_ax.set_xticks(minor_ticks, minor=True)
    main_ax.set_yticks(minor_ticks, minor=True)

    main_ax.tick_params(axis='both', which='major', length=6, width=2, direction='out')
    main_ax.tick_params(axis='both', which='minor', length=4, width=1.5, direction='out')

    main_ax.grid(True, which='major', linestyle='--', alpha=0.5, color='gray', zorder=1)
    main_ax.grid(True, which='minor', linestyle='--', alpha=0.2, color='gray', zorder=1)

    # 绘制散点和直线
    main_ax.scatter(true_ddg, predictions, alpha=0.6, color='red', s=50, zorder=3)
    main_ax.plot(plot_range, plot_range, '--', color='black', alpha=0.8, linewidth=1.5, zorder=2)
    # 增加拟合直线的宽度，使用主题色
    main_ax.plot(fit_x, fit_line, '-', color='red', alpha=0.8, linewidth=2.0, zorder=2)

    # 直方图
    # bins = np.linspace(plot_range[0], plot_range[1],
    #                    int(np.ceil(plot_range[1]) - np.floor(plot_range[0])) + 1)
    # 使用固定范围的直方图bins
    bins = np.arange(-10, 11, 1)  # 从-10到10，间隔为1
    plot_enhanced_histogram_v1(top_ax, true_ddg, bins, plot_range, vertical=True)
    plot_enhanced_histogram_v1(right_ax, predictions, bins, plot_range, vertical=False)

    # 隐藏直方图刻度
    top_ax.set_xticks([])
    top_ax.set_yticks([])
    right_ax.set_xticks([])
    right_ax.set_yticks([])

    # 控制边框显示
    main_ax.spines['top'].set_visible(False)
    main_ax.spines['right'].set_visible(False)
    top_ax.spines['top'].set_visible(False)
    top_ax.spines['right'].set_visible(False)
    top_ax.spines['left'].set_visible(False)
    right_ax.spines['top'].set_visible(False)
    right_ax.spines['right'].set_visible(False)
    right_ax.spines['bottom'].set_visible(False)

    # 准备文本
    dataset_name = f"{dataset.upper()}" if dataset else ""
    dataset_text = f"{dataset_name}\n"

    metrics_text = (
        f"RMSE: {rmse:.3f}\n"
        f"PCC   : {pcc:.3f}"
    )

    # 拟合直线方程（使用黑色）
    equation_text = f"y = {slope:.3f}x + {intercept:.3f}"

    # 添加文本标注
    main_ax.text(0.02, 0.98, dataset_text,
                 transform=main_ax.transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 fontsize=30,
                 fontweight='bold',
                 fontstyle='italic',
                 family='DejaVu Serif',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                           alpha=0,
                           edgecolor='none'),
                 zorder=5)

    main_ax.text(0.02, 0.91, metrics_text,
                 transform=main_ax.transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 fontsize=18,
                 fontweight='bold',
                 fontstyle='italic',
                 family='DejaVu Serif',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                           alpha=0,
                           edgecolor='none'),
                 zorder=5)

    main_ax.text(0.02, 0.81, equation_text,
                 transform=main_ax.transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 fontsize=18,
                 fontweight='bold',
                 fontstyle='italic',
                 family='DejaVu Serif',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                           alpha=0,
                           edgecolor='none'),
                 zorder=5)

    # 添加标签
    main_ax.set_xlabel('True', fontsize=20, fontweight='bold')
    main_ax.set_ylabel('Prediction', fontsize=20, fontweight='bold')

    # 保存图形
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def plot_results_plus(true_ddg, predictions, save_path, metrics=None):
    """
    Enhanced plotting function combining the best features of both versions
    """

    # Set clean style with white background and grid
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.grid': True,
        'grid.color': 'gray',
        'grid.linestyle': '-',
        'grid.alpha': 0.2,
        'font.size': 10
    })

    # Create figure and layout
    fig = plt.figure(figsize=(10, 10))
    gs = plt.GridSpec(3, 3, width_ratios=[4, 4, 1], height_ratios=[1, 4, 4],
                      hspace=0.05, wspace=0.05)

    # Create axes
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

    # Calculate data range
    all_values = np.concatenate([true_ddg, predictions])
    min_val = np.floor(min(all_values))
    max_val = np.ceil(max(all_values))
    margin = 0.5
    plot_range = [min_val - margin, max_val + margin]

    # Plot scatter points with improved style
    ax_main.scatter(true_ddg, predictions, alpha=0.5, color='salmon', s=40, zorder=3)

    # Plot diagonal line with refined style
    ax_main.plot(plot_range, plot_range, '--', color='red', alpha=0.6, linewidth=1.5, zorder=2)

    # Create histogram bins with gaps
    bins = np.linspace(plot_range[0], plot_range[1], 25)

    # Plot histograms with gaps (using rwidth)
    ax_top.hist(true_ddg, bins=bins, density=True, alpha=0.4, color='salmon',
                rwidth=0.8, zorder=2, edgecolor='none')
    ax_right.hist(predictions, bins=bins, density=True, orientation='horizontal',
                  alpha=0.4, color='salmon', rwidth=0.8, zorder=2, edgecolor='none')

    # Add density curves with refined style
    kde_points = np.linspace(plot_range[0], plot_range[1], 200)

    # Top density curve
    kde_x = gaussian_kde(true_ddg)
    ax_top.plot(kde_points, kde_x(kde_points), color='salmon', linewidth=1.5, zorder=3)

    # Right density curve
    kde_y = gaussian_kde(predictions)
    ax_right.plot(kde_y(kde_points), kde_points, color='salmon', linewidth=1.5, zorder=3)

    # Set axis limits
    ax_main.set_xlim(plot_range)
    ax_main.set_ylim(plot_range)

    # Configure spines
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(False)

    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)

    # Remove specific ticks and labels
    ax_top.set_yticks([])
    ax_right.set_xticks([])
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # Enable grid for main plot
    ax_main.grid(True, linestyle='-', alpha=0.2)
    ax_main.set_axisbelow(True)  # Ensure grid is below data points

    # Add metrics text with refined position and style
    if metrics:
        metrics_text = f"PCC={metrics['PCC']:.2f}, RMSE={metrics['RMSE']:.2f}"
    else:
        pcc = np.corrcoef(true_ddg, predictions)[0, 1]
        rmse = np.sqrt(np.mean((np.array(true_ddg) - np.array(predictions)) ** 2))
        metrics_text = f"PCC={pcc:.2f}, RMSE={rmse:.2f}"

    ax_top.text(0.02, 0.85, metrics_text,
                transform=ax_top.transAxes,
                verticalalignment='top',
                fontsize=12,
                color='black')

    # Add labels with refined style
    ax_main.set_xlabel('Experimental ΔΔG (kcal/mol)', fontsize=11)
    ax_main.set_ylabel('Predicted ΔΔG (kcal/mol)', fontsize=11)

    # Fine-tune tick parameters
    ax_main.tick_params(direction='out', length=4, width=2)
    ax_main.tick_params(which='both', bottom=True, top=False, left=True, right=False)

    # Ensure main plot's x-axis labels are visible
    ax_main.xaxis.set_tick_params(labelbottom=True)

    # Save with high quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_results(true_ddg, predictions, save_path):
    """绘制预测结果对比图"""
    plt.figure(figsize=(8, 8))
    plt.scatter(true_ddg, predictions, alpha=0.5)
    plt.plot([min(true_ddg), max(true_ddg)], [min(true_ddg), max(true_ddg)], 'r--')
    plt.xlabel('Experimental ΔΔG (kcal/mol)')
    plt.ylabel('Predicted ΔΔG (kcal/mol)')
    plt.title('Prediction vs Experiment')
    plt.savefig(save_path)
    plt.close()


def calculate_metrics(true_ddg, predictions):
    """计算各种评估指标"""
    return {
        'PCC': pearsonr(true_ddg, predictions)[0],
        'RMSE': np.sqrt(mean_squared_error(true_ddg, predictions)),
        'MAE': mean_absolute_error(true_ddg, predictions),
        'R2': np.corrcoef(true_ddg, predictions)[0, 1] ** 2,
        'Mean Error': np.mean(predictions - true_ddg),
        'Std Error': np.std(predictions - true_ddg)
    }


def print_header(text, width=50):
    """打印美观的标题"""
    print("\n" + "=" * width)
    print(f"{text:^{width}}")
    print("=" * width + "\n")


def print_section(text, width=50):
    """打印小节标题"""
    print(f"\n{'-' * 5} {text} {'-' * (width - len(text) - 7)}")


def format_percentage(value, total):
    """格式化百分比"""
    return f"{value} ({value / total * 100:.1f}%)"


def print_table(headers, rows, column_widths=None):
    """简单的表格打印函数，替代tabulate"""
    if column_widths is None:
        # 计算每列的最大宽度
        widths = []
        for i in range(len(headers)):
            col_items = [str(row[i]) for row in rows] + [str(headers[i])]
            widths.append(max(len(item) for item in col_items) + 2)
    else:
        widths = column_widths

    # 打印表头
    header = " ".join(f"{str(h):<{w}}" for h, w in zip(headers, widths))
    print(header)
    print("-" * sum(widths))

    # 打印数据行
    for row in rows:
        print(" ".join(f"{str(item):<{w}}" for item, w in zip(row, widths)))


def log_results(metrics, predictions, true_ddg, mutant_names, model_path, test_data_path):
    """使用print重构的结果输出函数"""
    print(f"\nPrediction completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    print_header("PREDICTION RESULTS")

    # 1. 模型和数据信息
    print_section("Model Information")
    info_rows = [
        ["Model Path", model_path],
        ["Test Dataset", test_data_path],
        ["Number of Samples", len(predictions)]
    ]
    print_table(["Item", "Value"], info_rows)

    # 2. 性能指标
    print_section("Performance Metrics")
    metrics_rows = [[k, f"{v:.4f}"] for k, v in metrics.items()]
    print_table(["Metric", "Value"], metrics_rows)

    # 3. 创建结果DataFrame
    results_df = pd.DataFrame({
        'Mutant': mutant_names,
        'True_DDG': true_ddg,
        'Predicted_DDG': predictions,
        'Absolute_Error': np.abs(predictions - true_ddg)
    })

    # 4. 预测统计
    print_section("Prediction Statistics")
    total = len(predictions)
    within_1 = (results_df['Absolute_Error'] <= 1.0).sum()
    within_2 = (results_df['Absolute_Error'] <= 2.0).sum()

    stats_rows = [
        ["Total Mutations", total],
        ["Within 1 kcal/mol", format_percentage(within_1, total)],
        ["Within 2 kcal/mol", format_percentage(within_2, total)]
    ]
    print_table(["Statistic", "Value"], stats_rows)

    # 5. 错误分布分析
    print_section("Error Distribution")
    error_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, float('inf'))]
    error_rows = []
    for low, high in error_ranges:
        mask = (results_df['Absolute_Error'] > low) & (results_df['Absolute_Error'] <= high)
        count = mask.sum()
        error_rows.append([
            f"{low}-{high if high != float('inf') else '∞'} kcal/mol",
            format_percentage(count, total)
        ])
    print_table(["Range", "Count"], error_rows)

    # 6. DDG范围分析
    print_section("DDG Range Analysis")
    ddg_ranges = [(-float('inf'), -3), (-3, -1), (-1, 1), (1, 3), (3, float('inf'))]
    ddg_rows = []
    for low, high in ddg_ranges:
        mask = (results_df['True_DDG'] > low) & (results_df['True_DDG'] <= high)
        if mask.any():
            subset = results_df[mask]
            mae = subset['Absolute_Error'].mean()
            count = len(subset)
            range_str = f"{low:>3.0f} to {high:<3.0f}" if low != float('-inf') else f"< {high:<3.0f}"
            ddg_rows.append([range_str, format_percentage(count, total), f"{mae:.2f}"])
    print_table(["DDG Range", "Count", "MAE (kcal/mol)"], ddg_rows)

    # 7. 最大/最小错误案例
    print_section("Largest Prediction Errors")
    largest_errors = results_df.nlargest(5, 'Absolute_Error')[
        ['Mutant', 'True_DDG', 'Predicted_DDG', 'Absolute_Error']
    ]
    print(largest_errors.to_string(index=False))

    print_section("Smallest Prediction Errors")
    smallest_errors = results_df.nsmallest(5, 'Absolute_Error')[
        ['Mutant', 'True_DDG', 'Predicted_DDG', 'Absolute_Error']
    ]
    print(smallest_errors.to_string(index=False))

    print("\nResults have been saved to:")
    print("- enhanced_predictions_detailed.csv")
    print("- prediction_enhanced_analysis.png")

    return results_df


# 保持其他函数不变
def save_results(predictions, true_ddg, metrics, filename='enhanced_predictions.csv'):
    """保存预测结果和指标"""
    results = pd.DataFrame({
        'True_DDG': true_ddg,
        'Predicted_DDG': predictions,
        'Absolute_Error': np.abs(predictions - true_ddg)
    })

    for metric, value in metrics.items():
        results.attrs[metric] = value

    results.to_csv(filename, index=False)
    logging.info(f"Results saved to {filename}")


def save_detailed_results(predictions, true_ddg, mutant_names, metrics, filename):
    """保存详细的预测结果并输出完整统计信息"""
    results = pd.DataFrame({
        'Mutant': mutant_names,
        'True_DDG': true_ddg,
        'Predicted_DDG': predictions,
        'Absolute_Error': np.abs(predictions - true_ddg)
    })

    for metric, value in metrics.items():
        results.attrs[metric] = value

    results = results.sort_values('Absolute_Error', ascending=False)
    results.to_csv(filename, index=False)

    logging.info(f"Detailed results saved to {filename}")
    return results