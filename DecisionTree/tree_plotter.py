# encoding: utf-8

# from pylab import *
import matplotlib.pyplot as plt


# 指定默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 定义文本框和箭头格式
decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plot_node(node_txt, center_pt, parent_pt, node_type):
    """
    绘制带箭头的注解
    :param node_txt:
    :param center_pt:
    :param parent_pt:
    :param node_type:
    :return:
    """
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                             textcoords='axes fraction', va='center',
                             ha='center', bbox=node_type, arrowprops=arrow_args)


def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node(u'决策节点', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node(u'叶节点', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


if __name__ == '__main__':
    create_plot()




