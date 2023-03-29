import os, sys
sys.path.append(os.pardir)
from common.tools import BinaryClass

if __name__ == '__main__':
    # データ読み込み
    binaryclass = BinaryClass()
    binaryclass.get_competitionID()     # competitionIDの指定
    na_values = ['', 'NA', -1, 9999]    # 欠損値一覧
    binaryclass.load(na_values)         # 欠損値を指定してデータ読込
    binaryclass.show_na_val()           # 欠損値の個数確認
    # binaryclass.fill_na_value(columns, na_val, val) #欠損値の値を指定して埋める
    binaryclass.describe()              # 統計値の出力
    binaryclass.draw_hist()             # ヒストグラム出力

    # 前処理
    objective = ''
    binaryclass.split_labels(objective)         # データとラベルに分ける
    drop_columns = ['index']                    # 削除したいコラム
    binaryclass.drop_columns(drop_columns)      # 不要なコラム削除
    # std_columns = []
    # binaryclass.standardization(std_columns)       # 使用データの標準化
    # minmax_columns = []
    # binaryclass.minmax_scaling(minmax_columns)        # Min−Maxスケーリング
    # boxcox_columns = []
    # binaryclass.box_cox(boxcox_columns)               # Box-Cox変換
    # yeo_columns = []
    # binaryclass.yeo_johnson(yeo_columns)

    # ハイパーパラメータ調整
    binaryclass.tune_hyper()            # グリッドサーチ
    binaryclass.cross_val(
        binaryclass.best_max_depth,
        binaryclass.best_min_child_weight
    )                                   # 交差検証
    binaryclass.train()                 # 学習
    # binaryclass.draw_heatmap()          # 混同行列の出力

    # 予測値をファイル出力
    binaryclass.submit(
        index='index',
        label_name='Outcome'
    )
