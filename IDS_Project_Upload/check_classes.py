import numpy as np
import joblib

le = joblib.load('models/label_encoder.pkl')
classes = list(le.classes_)
y_test = np.load('data/processed/y_test.npy')
y_train = np.load('data/processed/y_train.npy')

print('类别列表:')
for i, c in enumerate(classes):
    cnt_test = int((y_test == i).sum())
    cnt_train = int((y_train == i).sum())
    print(f'  {i}: {c:20s}  train={cnt_train:,}  test={cnt_test:,}')

print()
print('测试集总样本:', len(y_test))
print('训练集总样本:', len(y_train))
