import numpy as np

model_name1 = ""
name1 = ""
model1 = np.load(f'../metrices/{model_name1}/{name1}.npy', allow_pickle=True)

model_name2 = ""
name2 = ""
model2 = np.load(f'../metrices/{model_name2}/{name2}.npy', allow_pickle=True)

for i in range(4):
    print(model1[i] - model2[i])

for i in range(4, len(model2)):
    MAE = np.mean(np.abs(model1[i] - model2[i]))
    PCC = np.corrcoef(model1[i], model2[i])[0, 1]
    print(MAE, PCC)
