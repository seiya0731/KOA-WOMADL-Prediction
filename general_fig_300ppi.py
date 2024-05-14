
#旷圣遇#
#ROC曲线
#DCA曲线
#delongheatmap

from sklearn.metrics import roc_curve
import matplotlib
from sklearn.metrics import confusion_matrix
import scipy.stats as st
from pandas import Series, DataFrame
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
# 一种可视化库
import seaborn as sns
# 一种色卡库
import palettable

#delong检验
class DelongTest():
    def __init__(self, preds1, preds2, label, threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1 = preds1
        self._preds2 = preds2
        self._label = label
        self.threshold = threshold
        self._show_result()

    def _auc(self, X, Y) -> float:
        return 1 / (len(X) * len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self, X, Y) -> float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y == X else int(Y < X)

    def _structural_components(self, X, Y) -> list:
        V10 = [1 / len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1 / len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self, V_A, V_B, auc_A, auc_B) -> float:
        return 1 / (len(V_A) - 1) * sum([(a - auc_A) * (b - auc_B) for a, b in zip(V_A, V_B)])

    def _z_score(self, var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB) ** (.5) + 1e-8)

    def _group_preds_by_label(self, preds, actual) -> list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)

        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_A01, auc_A,
                                                                                                    auc_A) * 1 / len(
            V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1 / len(V_B10) + self._get_S_entry(V_B01, V_B01, auc_B,
                                                                                                    auc_B) * 1 / len(
            V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_B01,
                                                                                                       auc_A,
                                                                                                       auc_B) * 1 / len(
            V_A01))

        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p =st.norm.sf(abs(z)) * 2

        return z, p

    def _show_result(self):
        z, p = self._compute_z_p()
        # print(f"z score = {z:.5f};\np value = {p:.5f};")
        # if p < self.threshold:
        #     print("There is a significant difference")
        # else:
        #     print("There is NO significant difference")
#画图
class general_fig:
    def __init__(self, filename):
        # 导入数据
        self.filename = filename
        self.__data = read_csv(filename)
        self.__y_pred = self.__data.iloc[:, 1:]  # 各模型预测值
        self.__y_pred_scores = np.array(self.__y_pred)
        self.__y_class = self.__data.iloc[:, 0:1]  # 类
        self.__y_label = np.ravel(self.__y_class)
        self.__model_names = self.__y_pred.columns
        self.__thresh_group = np.arange(0, 1, 0.01)
        # 自定义颜色方案（一共10个）
        self.__colors = ['#0780cf', '#75623c', '#fa6d1d', '#57648a', '#b6b51f', '#da8784', '#6f5b6d', '#fcaef2',
                         '#009db2',
                         '#024b51']
    # ROC曲线

    def multi_models_roc(self):
        """
        将多个机器模型的roc图输出到一张图上
        Args:
            filename:含有不同模型预测概率的excel表格名称，string型
        """

        #下面matplotlib.rc设置的是图画框的一些数据,例如size=30，影响的是图表刻度以及legend里面（如果它不声明字体大小）的参数
        matplotlib.rc('font', family='Microsoft YaHei', size=30, weight='bold')
        matplotlib.rc('lines', lw=5)
        plt.rcParams['axes.facecolor'] = '#ffffff'
        plt.figure(figsize=(20, 20))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1])
        plt.xlabel('1-Specificity', fontsize=40)
        plt.ylabel('Sensitivity', fontsize=40)
        ax = plt.gca()  # 获取边框
        ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
        ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
        # ax.spines['left'].set_color('none')  # 设置上‘脊梁’为无色

        # plt.title('ROC Curve', fontsize=50)
        # 设置输出格式为tif
        plt.rcParams['savefig.format'] = 'tif'
        plt.rcParams['figure.autolayout'] = True
        colors = ['#0780cf', '#75623c', '#fa6d1d', '#57648a', '#b6b51f', '#da8784', '#6f5b6d', '#fcaef2', '#009db2',
                  '#024b51']
        for (model_name, color) in zip(self.__model_names, colors[0:len(self.__model_names)]):
            #每个模型的一维预测概率
            np.ravel(self.__y_pred[model_name])
            fpr, tpr, thresholds = roc_curve(self.__y_label, np.ravel(self.__y_pred[model_name]), pos_label=1)
            # AUC = auc(fpr, tpr)
            # plt.plot(fpr, tpr, lw=5, label='{} (AUC={:.3f})'.format(model_name, AUC), color=color)
            plt.plot(fpr, tpr, lw=5, label='{}'.format(model_name), color=color)
            plt.plot([0, 1], [0, 1], '--', lw=5, color='grey')
            # 两种显示图例位置的参数：borderaxespad=3 指图例距离相近的边或角的距离，具体百度
            plt.legend(loc='best', fontsize=25, borderaxespad=2.5)
            # 在图的外面不合适
            # plt.legend(loc='lower right', fontsize=30, bbox_to_anchor=(0.1,0.1))
        # bbox_inches='tight' 这个参数有时会导致 两组数据画出来的图大小不一致（重要）
        # plt.savefig(str(filename).replace(".csv", '', -1) + str('ROC') + str('.tif'), dpi=600, bbox_inches='tight')
        # plt.tight_layout()  bbox_inches='tight'都有让图像完全显示的意思，效果有待验证，必要时一起用（重要）
        plt.tight_layout()
        plt.savefig(str(filename).replace(".csv", '', -1) + str('ROC') + str('.tif'), dpi=300)
        # plt.show()

    #DCA曲线
    def calculate_net_benefit_model(self):
        self.net_benefit_model = np.array([])
        for thresh in self.__thresh_group:
            self.__y_pred_label = self.__y_pred_score > thresh
            tn, fp, fn, tp = confusion_matrix(self.__y_label, self.__y_pred_label).ravel()
            n = len(self.__y_label)
            net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            self.net_benefit_model = np.append(self.net_benefit_model, net_benefit)
        return self.net_benefit_model

    def calculate_net_benefit_all(self):
        self.net_benefit_all = np.array([])
        tn, fp, fn, tp = confusion_matrix(self.__y_label, self.__y_label).ravel()
        total = tp + tn
        for thresh in self.__thresh_group:
            net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
            self.net_benefit_all = np.append(self.net_benefit_all, net_benefit)
        return self.net_benefit_all

    def plot_DCA(self, ax):
        ax.set_xlim(-0.1, 1)
        ax.set_ylim(-0.2, 0.8) # adjustify the y axis limitation
        ax.set_xlabel(
            xlabel='Threshold Probability',
            fontsize=30
            # fontdict={'family': 'Times New Roman', 'fontsize': 15}
            )
        ax.set_ylabel(
            ylabel='Net Benefit',
            fontsize=30
            # fontdict={'family': 'Times New Roman', 'fontsize': 15}
            )
        #设置网格线
        ax.grid('major', color='#ffffff')
        #设置轴
        ax.spines['right'].set_color((0.8, 0.8, 0.8))
        ax.spines['top'].set_color((0.8, 0.8, 0.8))
        for (model_name, color) in zip(self.__model_names, self.__colors[0:len(self.__model_names)]):
            self.__y_pred_score = np.ravel(self.__y_pred[model_name])
            self.net_benefit_model = self.calculate_net_benefit_model()
            ax.plot(self.__thresh_group, self.net_benefit_model, color=color, label='{} '.format(model_name))
            ax.legend(loc='upper right', fontsize=20, framealpha=0.6, ncol=3)
        return ax

    def DCA(self):
        plt.figure(figsize=(15, 11))
        plt.rcParams['axes.facecolor'] = '#e5e5e5'
        # 设置输出格式为tif
        plt.rcParams['savefig.format'] = 'tif'
        plt.rcParams['figure.autolayout'] = True

        # 位置要放对
        fig, ax = plt.subplots(figsize=(15, 11))
        # 设置显示刻度
        plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=20)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8], size=20)
        self.net_benefit_all = self.calculate_net_benefit_all()
        plt.plot(self.__thresh_group, self.net_benefit_all, color='black', label='Treat all')
        plt.plot((0, 1), (0, 0), color='black', linestyle='--', label='Treat none')
        self.plot_DCA(ax)
        plt.tight_layout()
        plt.savefig(str(filename).replace(".csv", '', -1) + str('DCA') + str('.tif'), dpi=300)

    #delongheatmap图
    def Delongheatmap(self):
        p_matrix = np.ones(shape=(self.__y_pred.shape[1], self.__y_pred.shape[1]))
        # print(p_matrix)
        for col in self.__y_pred.columns:
            for col2 in self.__y_pred.drop(col, axis=1).columns: #axis=1 表示列
                preds_A = np.array(self.__y_pred[col])
                preds_B = np.array(self.__y_pred[col2])
                actual = self.__y_label
                _, p_value = DelongTest(preds_A, preds_B, actual)._compute_z_p()
                p_matrix[self.__y_pred.columns.to_list().index(col), self.__y_pred.columns.tolist().index(col2)] = p_value
               # mask1 = np.tril(p_values < 0.05) #tril函数的使用 => 取矩阵的左下三角部分
        # mask2 = np.invert(np.tril(p_values < 0.05)) #tril函数的使用 => 取矩阵的左下三角部分 #invert对于二进制数据的作用相当于取反

        X1 = DataFrame(p_matrix, index=self.__model_names, columns=self.__model_names)
        print(X1)
        # # p值图显
        matplotlib.rc('font', family='Microsoft YaHei', size=20, weight='normal')
        plt.rcParams['axes.facecolor'] = '#ffffff'
        # 设置输出格式为tif
        plt.rcParams['savefig.format'] = 'tif'
        plt.rcParams['figure.autolayout'] = True
        b = plt.figure(figsize=(11, 9))
        a = sns.heatmap(data=X1,
                vmax=1,
                vmin=0,
                cmap=palettable.cmocean.sequential.Amp_8.mpl_colors, #palettable:一种色库
                annot=True, #是否在heatmap中每个方格写入数据
                fmt=".2f", #每个小方格保留小数点后几位
                annot_kws={'size': 25, 'weight': 'bold', 'color': '#f88f08'}, #每个小方格的颜色，是否加粗，尺寸等信息
                # mask=np.invert(np.tril(p_values < 0.05)), #tril函数的使用 => 取矩阵的左下三角部分 #invert对于二进制数据的作用相当于取反
                mask=np.triu(np.ones_like(p_matrix, dtype=np.bool_)),
                square=True, linewidths=.5,#每个方格外框显示，外框宽度设置
                cbar_kws={"shrink": .5}, #热力图旁边小条条设置，其中.5就是0.5,shrink:收缩。"shrink": 1代表不收缩
                ax=None, #默认值
                xticklabels=True, #热力图x轴标签，注意不是figure标签设置,x轴为正
                # xticklabels=lablex,
                yticklabels=True, #热力图y轴标签，注意不是figure标签设置，y轴是倒过来
                # yticklabels=labley
                # 隐藏热力图小条子
                # cbar=False # 隐藏小条子
               )
        # # ------------设置颜色条刻度字体的大小-------------------------#
        # cb = a.figure.colorbar(a.collections[0])  # 显示colorbar
        # cb.ax.tick_params(labelsize=14)  # 设置colorbar刻度字体大小。

        xtick_label = self.__model_names
        a.set_yticklabels(xtick_label, rotation=0, fontsize=20)
        a.set_xticklabels(xtick_label, rotation=90, fontsize=20)
        #下面这行代码可以让图完全显示(重要)
        plt.tight_layout()
        plt.show()
        # 导出图
        heatmap = a.get_figure()
        heatmap.savefig(str(filename).replace(".csv", '', -1) + str('Delong') + str('heatmap') + str('.tif'), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    filenames = ['score_train.csv', 'score_test.csv']

    for filename in filenames:
        p = general_fig(filename)
        p.multi_models_roc()
        p.DCA()
        p.Delongheatmap()
        plt.show()



