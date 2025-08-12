import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
from numpy.typing import NDArray

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_time_series(data_series, title="Time Series Plot", 
                     xlabel="Time", ylabel="Traffic Flow", figsize=(6, 4), show=True, savepath=None):
    """
    绘制多组时间序列数据的折线图
    
    参数:
    data_series: 包含多个字典的列表，每个字典代表一组数据，包含:
        'timestamps': 时间点列表 (datetime对象或时间字符串)
        'values': 对应时间点的数值列表
        'name': 数据序列名称 (用于图例)
        'color': 折线颜色 (可选)
        'linestyle': 折线样式 (可选)
    title: 图表标题
    xlabel: X轴标签
    ylabel: Y轴标签
    figsize: 图表尺寸
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    for series in data_series:
        # 转换时间戳为datetime对象
        times = [t if isinstance(t, datetime) else datetime.strptime(t, "%Y-%m-%d %H:%M:%S") 
                for t in series['timestamps']]
        
        # 获取样式参数 (提供默认值)
        color = series.get('color', None)
        linestyle = series.get('linestyle', '-')
        
        # 绘制折线
        ax.plot(
            times,
            series['values'],
            label=series['name'],
            color=color,
            linestyle=linestyle,
            marker='o' if len(times) < 10 else None,  # 数据点少时显示标记
            markersize=6,
            linewidth=2
        )
    
    # 设置时间轴格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签
    
    # 添加标签和标题
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # 添加图例和网格
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()
    
def ndarray2plot(arr:NDArray[np.float32], startTime:datetime, name:str, color:str, linestype='-', granularity=timedelta(minutes=5)):
    timestamps = [startTime + i * granularity for i in range(arr.size)]
    return {'name':name, 'timestamps':timestamps, 'values':arr.tolist(), 'color':color, 'linestyle':linestype}

# 测试代码
if __name__ == "__main__":
    # 创建测试数据 (包含datetime对象和字符串两种格式)
    data = [
        {
            'name': "温度(℃)",
            'timestamps': [
                "2023-06-01 08:00:00",
                "2023-06-01 12:00:00",
                "2023-06-01 16:00:00",
                "2023-06-01 20:00:00"
            ],
            'values': [22.1, 26.5, 28.3, 24.8],
            'color': 'crimson',
            'linestyle': '-'
        },
        {
            'name': "湿度(%)",
            'timestamps': [
                datetime(2023, 6, 1, 9, 30),
                datetime(2023, 6, 1, 12, 45),
                datetime(2023, 6, 1, 15, 20),
                datetime(2023, 6, 1, 18, 10),
                datetime(2023, 6, 1, 21, 0)
            ],
            'values': [65, 58, 52, 68, 72],
            'color': 'royalblue',
            'linestyle': '--'
        },
        {
            'name': "气压(hPa)",
            'timestamps': [
                "2023-06-01 06:00:00",
                "2023-06-01 12:00:00",
                "2023-06-01 18:00:00",
                "2023-06-02 00:00:00"
            ],
            'values': [1012, 1010, 1011, 1013],
            'color': 'forestgreen',
            'linestyle': '-.'
        }
    ]

    # 绘制图表
    plot_time_series(
        data,
        title="环境传感器数据 (2023-06-01)",
        ylabel="测量值",
        figsize=(14, 7)
    )