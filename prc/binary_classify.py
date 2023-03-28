import os, sys
sys.path.append(os.pardir)
from common.tools import BinaryClass

if __name__ == '__main__':
    binaryclass = BinaryClass()
    binaryclass.get_competitionID()  # competitionIDの指定
    na_values = ['', 'NA', -1, 9999]    # 欠損値一覧
    binaryclass.load(na_values)         # 欠損値を指定してデータ読込
    # binaryclass.describe()            # 統計値の出力
    binaryclass.split_labels('Outcome') # データとラベルに分ける
    columns = ['index']                 # 削除したいコラム
    binaryclass.drop_columns(columns)   # 不要なコラム削除
    binaryclass.standardization()       # 使用データの標準化
    binaryclass.tune_hyper()            # グリッドサーチ
    binaryclass.cross_val(
        binaryclass.best_max_depth,
        binaryclass.best_min_child_weight
    )                                   # 交差検証
    binaryclass.train()                 # 学習
    # binaryclass.draw_heatmap()          # 混同行列の出力
    binaryclass.submit(
        index='index',
        label_name='Outcome'
    )                                   # csv出力
