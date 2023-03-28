import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

# 二値分類
class BinaryClass:
    competitionID = 0
    home_path = os.environ['HOME']
    signate_path = os.path.join(home_path, 'data/signate')
    data_path = ""
    train_csv = None
    test_csv = None
    train_data = None
    train_labels = None
    test_data = None
    best_max_depth = None
    best_min_child_weight = None
    best_label_threshold = .5
    model = None
    pred = None
    pred_label = None
    f1_score = None

    def __init__(self) -> None:
        pd.set_option('display.max_columns', 100)   # DataFrame全表示

    # competitionIDの取得
    def get_competitionID(self, competitionID):
        self.competitionID = competitionID
        self.data_path = os.path.join(self.signate_path, str(competitionID))

    # 使用するデータの読み込み
    def load(self, na_values):
        train_path = os.path.join(self.data_path, 'train.csv')
        test_path = os.path.join(self.data_path, 'test.csv')
        self.train_csv = pd.read_csv(train_path, na_values=na_values)
        self.test_csv = pd.read_csv(test_path, na_values=na_values)

    # 欠損値の個数確認
    def show_na_val(self):
        print("========== train ==========")
        print(self.train_csv.count())
        print("========== test ==========")
        print(self.test_csv.count())

    # 欠損値を特定の値で埋める
    def fill_na_value(self, columns, na_val, val=0):
        for column in columns:
            self.train_data[column] = self.train_data[column].replace(na_val, val)
            self.test_data[column] = self.test_data[column].replace(na_val, val)

    # データの基本統計量の表示
    def describe(self):
        print("========== train ==========")
        print(self.train_csv.describe())
        print("========== test ==========")
        print(self.test_csv.describe())

    # 学習データをデータとラベルに分割する
    def split_labels(self, label_name):
        self.train_data = self.train_csv.drop([label_name], axis=1)
        self.train_labels = self.train_csv[label_name]

    # 不要なコラムの削除
    def drop_columns(self, columns):
        self.train_data = self.train_data.drop(columns, axis=1)
        self.test_data = self.test_csv.drop(columns, axis=1)

    # データの値を標準化
    def standardization(self):
        scaler = StandardScaler()
        scaler.fit(self.train_data)
        self.train_data = scaler.transform(self.train_data)
        self.test_data = scaler.transform(self.test_data)
        print(type(self.train_data))

    # グリッドサーチ
    def tune_hyper(self):
        param_space = {
            'max_depth': [3, 5, 7],
            'min_child_widht': [1.8, 2.0, 4.8],
            'label_threshold': [i * 0.1 for i in range(10)]
        }
        param_combinations = itertools.product(param_space['max_depth'], param_space['min_child_widht'], param_space['label_threshold'])
        params = []
        scores = []
        for max_depth, min_child_weight, label_threshold in param_combinations:
            print("max_depth:", max_depth, "min_child_weight:", min_child_weight, "label_threshold:", label_threshold)
            logloss, accuracy = self.cross_val(max_depth, min_child_weight, label_threshold)
            params.append((max_depth, min_child_weight, label_threshold))
            scores.append(accuracy)
        best_idx = np.argmax(scores)
        self.best_max_depth = params[best_idx][0]
        self.best_min_child_weight = params[best_idx][1]
        self.best_label_threshold = params[best_idx][2]

    # 交差検証
    def cross_val(self, max_depth, min_child_weight, label_threshold=.5):
        scores_accuracy = []
        scores_logloss = []
        kf = KFold(n_splits=4, shuffle=True, random_state=71)
        for tr_idx, va_idx in kf.split(self.train_data):
            tr_x, va_x = self.train_data[tr_idx], self.train_data[va_idx]
            tr_y, va_y = self.train_labels[tr_idx], self.train_labels[va_idx]

            model = XGBClassifier(
                n_estimators=20, random_state=71,
                max_depth=max_depth, min_child_weight=min_child_weight
            )
            model.fit(tr_x, tr_y)

            va_pred = model.predict_proba(va_x)[:, 1]
            logloss = log_loss(va_y, va_pred)
            accuracy = accuracy_score(va_y, va_pred > label_threshold)

            scores_logloss.append(logloss)
            scores_accuracy.append(accuracy)
        logloss = np.mean(scores_logloss)
        accuracy = np.mean(scores_accuracy)
        print("CROSS VALIDATION")
        print("LOGLOSS:", logloss, "ACCURACY: ", accuracy)
        return logloss, accuracy

    # 混同行列の出力
    def draw_confusion_matrix(self, test_labels, prediction_labels):
        cm = confusion_matrix(test_labels, prediction_labels)
        regularized_cm = cm.astype('float')
        size = len(set(test_labels))
        for i in range(size): regularized_cm[i] /= np.sum(cm[i])
        fig, ax = plt.subplots()
        heatmap = ax.pcolor(regularized_cm, cmap=plt.cm.Blues)
        ys, xs = np.meshgrid(range(cm.shape[0]), range(cm.shape[1]), indexing='ij')
        for (x,y,val) in zip(xs.flatten(), ys.flatten(), cm.flatten()):
            color = 'white' if regularized_cm[y][x] > 0.7 else 'black'
            plt.text(x+0.5, y+0.5, int(val), color=color, horizontalalignment='center',verticalalignment='center',)
        ax.set_xticks(np.arange(size) + 0.5, minor=False)
        ax.set_yticks(np.arange(size) + 0.5, minor=False)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        #ax.xaxis.tick_top()
        index = [i for i in range(size)]
        ax.set_xticklabels(index, minor=False)
        ax.set_yticklabels(index, minor=False)
        plt.xlabel("Pre", fontsize=13)
        plt.yticks(rotation=0)
        plt.ylabel("GT", fontsize=13)
        plt.tight_layout()
        #plt.show()

    # 訓練データの学習
    def train(self):
        self.model = XGBClassifier(
            n_estimators=20, random_state=71,
            max_depth=self.best_max_depth, min_child_weight=self.best_min_child_weight
        )
        self.model.fit(self.train_data, self.train_labels)
        self.pred = self.model.predict_proba(self.test_data)[:, 1]
        self.pred_label = np.where(self.pred > self.best_label_threshold, 1, 0)

    # 提出ファイルの作成
    def submit(self, index, label_name):
        submission_file = os.path.join('submission', str(self.competitionID) + '.csv')
        os.system('touch ' + submission_file)
        submission = pd.DataFrame({index: self.test_csv[index], label_name: self.pred_label})
        print(submission)
        submission.to_csv(submission_file, index=False, header=False)

if __name__ == '__main__':
    binaryclass = BinaryClass()
    binaryclass.get_competitionID(748)
    binaryclass.load()
    binaryclass.describe()
    binaryclass.split_labels('Outcome')
