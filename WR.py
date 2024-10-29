from utils import ExtendedMQCNNEstimator
import mxnet as mx
import json
# Training and forecasting based on the public dataset
import pandas as pd
import numpy as np
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions

# Load data
data = pd.read_csv('./sfs_holtwinter_data_0801.csv', encoding='gbk')
forecast_date = '2014-06-01'
data['length'] = data['target'].apply(lambda x: len(eval(x)))

# Get the minimum start date and the maximum length of the series
start_date = data['start'].min()
length = data['length'].max()

# Filter data based on start date and length
data_filtered = data[(data['start'] == start_date) & (data['length'] == length)]

# Define metadata for the dataset
custom_ds_metadata = {
    'num_series': data_filtered.shape[0],
    'num_steps': 24,
    'prediction_length': 30,
    'freq': 'D',
    'start': [pd.Timestamp(start_date, freq='D') for _ in range(data_filtered.shape[0])]
}

# Calculate train and validation end indices
train_end_index = int((length - 30) * 0.8)
valid_end_index = int(length - 30)


def get_input_data(input_data, index_start=6):
    target = np.array(input_data['target'].apply(eval))
    target = np.array([np.array(i) for i in target.tolist()])
    feat_dynamic_real = np.array(input_data['feat_dynamic_real'].apply(eval))
    feat_dynamic_real = np.array([np.array(i) for i in feat_dynamic_real.tolist()])

    # Transpose the feat_dynamic_real array to match expected dimensions
    feat_dynamic_real_new = np.zeros(
        [feat_dynamic_real.shape[0], feat_dynamic_real.shape[2], feat_dynamic_real.shape[1]])
    for i in range(feat_dynamic_real.shape[0]):
        for j in range(feat_dynamic_real.shape[2]):
            feat_dynamic_real_new[i, j, :] = feat_dynamic_real[i, :, j]
    feat_dynamic_real = feat_dynamic_real_new[:, index_start:, :]

    feat_static_cat = np.array(input_data['feat_static_cat'].apply(eval))

    return target, feat_dynamic_real, feat_static_cat


# Get input data
target, feat_dynamic_real, feat_static_cat = get_input_data(data_filtered, index_start=0)

# Create datasets
train_ds = ListDataset([{
    FieldName.TARGET: target_item,
    FieldName.START: start,
    FieldName.FEAT_DYNAMIC_REAL: fdr,
    FieldName.FEAT_STATIC_CAT: fsc
} for target_item, start, fdr, fsc in
    zip(target[:, :train_end_index], custom_ds_metadata['start'], feat_dynamic_real[:, :, :train_end_index],
        feat_static_cat)], freq=custom_ds_metadata['freq'])

valid_ds = ListDataset([{
    FieldName.TARGET: target_item,
    FieldName.START: start,
    FieldName.FEAT_DYNAMIC_REAL: fdr,
    FieldName.FEAT_STATIC_CAT: fsc
} for target_item, start, fdr, fsc in
    zip(target[:, :valid_end_index], custom_ds_metadata['start'], feat_dynamic_real[:, :, :valid_end_index],
        feat_static_cat)], freq=custom_ds_metadata['freq'])

test_ds = ListDataset([{
    FieldName.TARGET: target_item,
    FieldName.START: start,
    FieldName.FEAT_DYNAMIC_REAL: fdr,
    FieldName.FEAT_STATIC_CAT: fsc
} for target_item, start, fdr, fsc in zip(target, custom_ds_metadata['start'], feat_dynamic_real, feat_static_cat)],
    freq=custom_ds_metadata['freq'])

# Model parameters
prediction_length = custom_ds_metadata['prediction_length']
context_length = 24

# Get training parameters from json file
TRAINING_ARGS = json.load(open('training_args.json', 'r'))
trainer = Trainer(ctx=mx.cpu(), **TRAINING_ARGS)
# Define the WREstimator
estimator = ExtendedMQCNNEstimator(freq=custom_ds_metadata['freq'], prediction_length=prediction_length,
                                   trainer=trainer,
                                   context_length=context_length, use_feat_dynamic_real=True, use_feat_static_cat=False,
                                   add_time_feature=False, add_age_feature=False, seed=100, quantiles=[0.5], alpha=1,
                                   N=3)

# Train the model
predictor = estimator.train(train_ds, valid_ds)

# Make evaluation predictions
forecast_it, ts_it = make_evaluation_predictions(dataset=test_ds, predictor=predictor, num_samples=100)
forecasts = list(forecast_it)
tss = list(ts_it)

# Output predictions
fore_result = []
data_4 = data_filtered.reset_index()
for index, row in data_4.iterrows():
    print(row['sku_id'])
    fore_result_df = pd.DataFrame()
    fore_result_df['forecast_result_date'] = pd.date_range(forecast_date, periods=30)
    median = np.median([i for i in forecasts[index].forecast_array[0, :] if i >= 0])
    fore_result_df['median_point'] = [i if i >= 0 else median for i in forecasts[index].forecast_array[0, :]]
    fore_result_df['mean_point'] = forecasts[index].forecast_array[0, :]
    fore_result_df['sku_id'] = str(row['sku_id'])
    fore_result_df['dc_id'] = str(row['dc_id'])
    fore_result_df['forecast_date'] = forecast_date
    fore_result.append(fore_result_df)

fore_result_final = pd.concat(fore_result)
