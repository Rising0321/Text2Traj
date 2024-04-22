import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model.bigModelQueen import bigModelQueen

from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss
from sklearn.metrics import mean_absolute_error, roc_auc_score, f1_score, average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def calc(task_type, phase, reshaped_prediction, reshaped_label):
    reshaped_prediction = np.concatenate(reshaped_prediction).reshape(-1)
    reshaped_label = np.concatenate(reshaped_label).reshape(-1)
    if task_type == "regression":
        mse = mean_squared_error(reshaped_prediction, reshaped_label)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(reshaped_prediction, reshaped_label)
        pcc = np.corrcoef(reshaped_prediction, reshaped_label)[0][1]
        print(phase, mae, mse, rmse, pcc)
        return (mae, mse, rmse, pcc)
    elif task_type == "biclassification":
        accuracy = accuracy_score(reshaped_prediction, reshaped_label)
        f1_micro = f1_score(reshaped_prediction, reshaped_label,average="micro")
        f1_macro = f1_score(reshaped_prediction, reshaped_label,average="macro")
        print(phase, f1_micro, f1_micro, f1_macro)
        return (f1_micro, f1_micro, f1_macro)
    else:
        accuracy = accuracy_score(reshaped_prediction, reshaped_label)
        auc = roc_auc_score(reshaped_prediction, reshaped_label)
        ap = average_precision_score(reshaped_prediction, reshaped_label)
        f1_micro = f1_score(reshaped_prediction, reshaped_label,average="micro")
        f1_macro = f1_score(reshaped_prediction, reshaped_label,average="macro")

        print(phase, accuracy, auc, ap, f1_micro, f1_macro)

        return (accuracy, auc, ap, f1_micro, f1_macro)
    

class QueenDataset(Dataset):
    def __init__(self, a, b, task):
        self.a = a
        self.b = b
        self.task = task

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        a_tensor = torch.tensor(self.a[idx][0])
        if self.task == "regression":
            b_tensor = torch.tensor(self.b[idx])
        else:
            b_tensor = torch.tensor(self.b[idx])
        return (a_tensor, b_tensor)

def get_human_embedding(path):
    
    uid_list = []
    emb_list = []
    
    for i in range(6):
        uid = np.load("./embeddings/"+path+"_uids_%d.npy"%i)
        emb = np.load("./embeddings/"+path+"_embeddings_%d.npy"%i)
        uid_list.extend(uid)
        emb_list.extend(emb)
    
    now = pd.DataFrame({"uid":uid_list,"embedding":emb_list})
    
    return now
    
selector = [["commuter", "commuter"], ["driver_bin","label"], ["subway","tripCountOf10Days"], ["car","is_has_car"]]

def get_sum_df(args, target):
    target = selector[target]
    human_embedding = get_human_embedding(args.model_name)
    task = pd.read_csv("./files/downstream/user_data/%s.csv"%target[0])
    task = task.rename(columns={task.columns[0]:"uid"})
    
    sum_df = pd.merge(task, human_embedding, on=["uid"])
    sum_df = sum_df[["embedding", target[1]]]
    # print(human_embedding)
    # exit(0)
    dim = 768
    return sum_df, dim    


def load_data(args):
    
    sum_df, dim = get_sum_df(args, args.target)
    
    shuffled_index = np.random.permutation(sum_df.index)
    sum_df = sum_df.iloc[shuffled_index]
    print(len(sum_df))
    target = selector[args.target]
    
    x = sum_df[["embedding"]].values
    y_ = sum_df[target[1]].values

    y_mean = np.mean(y_)
    max_feat = np.max(y_) + 1


    if args.task != "regression":
        y = torch.nn.functional.one_hot(torch.Tensor(y_).long(), num_classes=max_feat).numpy()
        # print(y)
    else:
        y = y_

    x_train, x_val, x_test = x[:int(0.7 * len(x))], x[int(0.7 * len(x)):int(0.8 * len(x))], x[int(0.8 * len(x)):]
    y_train, y_val, y_test = y[:int(0.7 * len(x))], y[int(0.7 * len(x)):int(0.8 * len(x))], y[int(0.8 * len(x)):]

    train_loader = DataLoader(QueenDataset(x_train, y_train, args.task), batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(QueenDataset(x_val, y_val, args.task), batch_size=args.batchsize, shuffle=False)
    test_loader = DataLoader(QueenDataset(x_test, y_test, args.task), batch_size=args.batchsize, shuffle=False)

    return train_loader, val_loader, test_loader, y_mean, max_feat, dim


if __name__ == '__main__':
    # python3 evaluation_transfer.py --gpu cuda:0 --task biclassification --target 0
    # python3 evaluation_transfer.py --gpu cuda:1 --task biclassification --target 1
    # python3 evaluation_transfer.py --gpu cuda:2 --task regression --target 2
    # python3 evaluation_transfer.py --gpu cuda:3 --task biclassification --target 3
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--batchsize', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--gpu', type=str, default="cuda:0")
    
    parser.add_argument('--model_name', type=str, default="Finalur")
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--target', type=int, default=0, help="")
    parser.add_argument('--task', type=str, default="classification", help="classification, regression")
    
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_args()

    train_loader, val_loader, test_loader, y_mean, max_feat, dim = load_data(args)

    model = bigModelQueen(dim, args.gpu, args.dropout, args.task, max_feat).to(args.gpu)

    if args.task == "regression":
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 9999
    ans_set = None
    best_count = 0

    for epoch in range(10000):
        update_flag = 0
        outputs = []
        anss = []
        print("------------------------%d------------------------" % epoch)
        for i, (a_batch, b_batch) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            
            output = model(a_batch.float().to(args.gpu))

            if args.task == "regression":
                loss = criterion(output, b_batch.float().unsqueeze(1).to(args.gpu))
                loss.backward()
                optimizer.step()
            else:
                loss = criterion(output, b_batch.float().to(args.gpu))
                loss.backward()
                optimizer.step()
                output = torch.argmax(output, dim=1)
                b_batch = torch.argmax(b_batch, dim=1)
            outputs.append(output.cpu().detach())

            anss.append(b_batch.float().unsqueeze(1).cpu())
        calc(args.task, "train", outputs, anss)

        with torch.no_grad():
            model.eval()
            outputs = []
            anss = []
            for i, (a_batch, b_batch) in enumerate(val_loader):
                output = model(a_batch.float().to(args.gpu))

                if args.task != "regression":
                    output = torch.argmax(output, dim=1)
                    b_batch = torch.argmax(b_batch, dim=1)

                outputs.append(output.cpu().detach())

                anss.append(b_batch.float().unsqueeze(1).cpu())
            now_val = calc(args.task, "val", outputs, anss)[2]
            if now_val > best_val:
                update_flag = 1
                best_val = now_val
                best_count = 0

        with torch.no_grad():
            outputs = []
            anss = []
            for i, (a_batch, b_batch) in enumerate(test_loader):
                output = model(a_batch.float().to(args.gpu))

                if args.task != "regression":
                    output = torch.argmax(output, dim=1)
                    b_batch = torch.argmax(b_batch, dim=1)

                outputs.append(output.cpu().detach())

                anss.append(b_batch.float().unsqueeze(1).cpu())

            now_res = calc(args.task, "test", outputs, anss)

            if update_flag == 1:
                ans_set = now_res

        best_count += 1

        if best_count > 100:
            break
    print(ans_set)
