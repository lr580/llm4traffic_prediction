import numpy as np
import sys
import os, tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.baselines import Results, Result  # noqa: E402
from utils.datasets import LargeSTDatasetHandler, EvaluateResult  # noqa: E402


def evaluate_dataset(name: str, model: str = 'HA') -> Result:
    handler = LargeSTDatasetHandler(name, stat=True, loadData=True)
    dataset = handler.dataset
    start = dataset.val_end
    end = dataset.t
    y_true = dataset.get_test()[:, :, 0]
    y_pred = np.stack(
        [handler.stat.getRange(start, end, j, dataset.data) for j in range(dataset.n)],
        axis=1,
    ).astype(np.float32)
    evaluate_result = EvaluateResult.from_predictions(y_pred, y_true, verbose=False)
    return Result(model=model, dataset=name, evaluateResult=evaluate_result, tags='HA')


if __name__ == '__main__':
    BASELINE_DATASETS = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08', 'SD', 'GBA', 'GLA', 'CA']
    collected = Results()
    for name in tqdm.tqdm(BASELINE_DATASETS):
        collected.append(evaluate_dataset(name))
    df = collected.to_dataframe()[['dataset', 'mae', 'mape', 'rmse']].set_index('dataset')
    print(df)

'''
PEMS03 MAE:26.1007, MAPE:0.2687, RMSE:47.4744
PEMS04 MAE:26.4224, MAPE:0.1678, RMSE:43.4247
PEMS07 MAE:30.3553, MAPE:0.1280, RMSE:56.7535
PEMS08 MAE:23.2495, MAPE:0.1450, RMSE:40.5865
SD     MAE:34.5474, MAPE:0.1958, RMSE:72.9972 13+3+s
GBA    MAE:32.3985, MAPE:0.2477, RMSE:56.2634 45+10+s
GLA    MAE:34.5100, MAPE:0.2289, RMSE:63.0197 76+20+s
CA     MAE:31.8463, MAPE:0.2366, RMSE:59.0193 310+47+s
'''
