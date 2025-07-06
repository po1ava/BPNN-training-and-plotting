import pandas as pd
from pyecharts.charts import Line
from pyecharts.options import AxisLineOpts
from pyecharts.options import AxisOpts
from pyecharts.options import AxisTickOpts
from pyecharts.options import InitOpts
from pyecharts.options import ItemStyleOpts
from pyecharts.options import LabelOpts
from pyecharts.options import LegendOpts
from pyecharts.options import LineStyleOpts
from pyecharts.options import SplitLineOpts
from pyecharts.options import TextStyleOpts
from pyecharts.options import TitleOpts
from pyecharts.options import ToolboxOpts
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot

log_path = f'selu_l3_lr0.005.xlsx'
log_row = range(0, 25, 5)
chart_name = log_path.replace('.xlsx', '')
title = chart_name + ' training result'
chart_width = '1200px'
chart_height = '675px'
xaxis_min = 0
xaxis_max = 300
yaxis_min = 0.3
yaxis_max = 0.7
yaxis_interval = 0.05
color = ['rgb(255,90,73)', 'rgb(243,173,114)', 'rgb(115,176,94)', 'rgb(58,106,126)', 'rgb(112,59,130)']

file = pd.read_excel(log_path)
max_num = int(file.columns[-1].replace('val', ''))
length = max_num - int(file.columns[-2].replace('val', ''))
x_list = range(length, max_num + 1, length)

data = []
for i in log_row:
    data_train = file.iloc[i, :int(max_num / length) + 3]
    data_test = file.iloc[i, int(max_num / length) + 3:]
    data.append([data_train, data_test])

line = Line(init_opts=InitOpts(width=chart_width, height=chart_height))
line.add_xaxis(x_list)

for i in range(len(data)):
    line.add_yaxis(series_name=f'{file.iloc[log_row[i], 0].split("_")[0]} train    ', y_axis=data[i][0],
                   label_opts=LabelOpts(is_show=False), is_smooth=True, is_symbol_show=False,
                   linestyle_opts=LineStyleOpts(width=2.5, color=color[i]),
                   itemstyle_opts=ItemStyleOpts(border_width=0))

for i in range(len(data)):
    line.add_yaxis(series_name=f'{file.iloc[log_row[i], 0].split("_")[0]} val    \u2009\u2009\u2009', y_axis=data[i][1],
                   label_opts=LabelOpts(is_show=False), is_smooth=True, is_symbol_show=False,
                   linestyle_opts=LineStyleOpts(width=2.5, type_='dotted', color=color[i]),
                   itemstyle_opts=ItemStyleOpts(border_width=0))

line.set_global_opts(
    title_opts=TitleOpts(title=title,
                         pos_left="center", pos_bottom="92%",
                         title_textstyle_opts=TextStyleOpts(font_family='Times New Roman', font_size=25,
                                                            color='black')),

    legend_opts=LegendOpts(is_show=True, item_height=0, item_width=30, pos_bottom='12%', pos_left='35%',
                           border_color='black', border_width=0,
                           textstyle_opts=TextStyleOpts(font_family='Times New Roman', font_size=15,
                                                        color='black')),

    toolbox_opts=ToolboxOpts(is_show=False),

    xaxis_opts=AxisOpts(min_=xaxis_min, max_=xaxis_max, name='epochs',
                        axisline_opts=AxisLineOpts(linestyle_opts=LineStyleOpts(width=2, color='black')),
                        axistick_opts=AxisTickOpts(length=6, linestyle_opts=LineStyleOpts(width=2, color='black')),
                        splitline_opts=SplitLineOpts(is_show=False),
                        axislabel_opts=LabelOpts(font_family='Times New Roman', font_size=15, interval=24, ),
                        name_textstyle_opts=TextStyleOpts(font_family='Times New Roman', font_size=18)),

    yaxis_opts=AxisOpts(min_=yaxis_min, max_=yaxis_max, interval=yaxis_interval, name='R²',
                        axisline_opts=AxisLineOpts(linestyle_opts=LineStyleOpts(width=2, color='black')),
                        axistick_opts=AxisTickOpts(length=6, linestyle_opts=LineStyleOpts(width=2, color='black')),
                        splitline_opts=SplitLineOpts(is_show=False),
                        axislabel_opts=LabelOpts(font_family='Times New Roman', font_size=15, interval=24, ),
                        name_textstyle_opts=TextStyleOpts(font_family='Times New Roman', font_size=18))
)

make_snapshot(snapshot, line.render(f"{chart_name}_line.html"), f'{chart_name}_line.png')
