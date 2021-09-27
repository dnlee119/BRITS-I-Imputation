import os, sys, time
import brits_I as B
import torch
import torch.optim as optim
import ujson as json
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import DataPreprocessing

if __name__ == "__main__":
    data_list = os.listdir("put_csv_in_here")
    if len(data_list) == 0:
        raise ValueError("No one data in 'put_csv_in_here' folder")
    elif len(data_list) > 1:
        raise ValueError("Must one data in 'put_csv_in_here' folder")
    else:
        DataPreprocessing.setting(data_list[0])

    # 하이퍼 파라미터
    epoch = 100
    learning_rate = 0.01

    # 기본 설정
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.random.manual_seed(0)
    np.random.seed(0)

    # 데이터 불러오기
    path = "./data.csv"
    B.makedata(path)
    df = pd.read_csv(path)
    length = len(df)

    # 모델 학습
    model = B.Brits_i(108, 1, 0, length, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    data_iter = B.get_loader('./dataset.json', batch_size=64)
    model.train()
    progress = tqdm(range(epoch))
    loss_graphic = []
    for i in progress:
        total_loss = 0.0
        for idx, data in enumerate(data_iter):
            data = B.to_var(data, device)
            ret = model.run_on_batch(data, optimizer, i)
            total_loss += ret["loss"]
        loss_graphic.append(total_loss.tolist())
        progress.set_description("loss: {:0.4f}".format(total_loss / len(data_iter)))

    # 결과
    result = B.predict_result(model, data_iter, device, df)
    to_csv_data = result.tolist()
    none_data = df["value"].isnull()
    for i in range(len(none_data)):
        if none_data.iloc[i] == True:
            df["value"].iloc[i] = to_csv_data[i]
    df.to_csv("./imputated_data.csv", index=False)
    with open("./predict_graph.json", "w") as fs:
        to_json = json.dumps(result.tolist())
        fs.write(to_json)
    with open("./loss_graph.json", "w") as fs:
        to_json = json.dumps(loss_graphic)
        fs.write(to_json)

    # test용 데이터 비교 그래프
    # real = pd.read_csv("서인천IC-부평IC 평균속도.csv",encoding='CP949')
    # plt.figure(figsize=(15, 5))
    # plt.plot(real["평균속도"], label="real")
    # lb = "predict"
    # switch = 0
    # tuple_ = tuple
    # missing_range = []
    # for i in range(len(none_data)):
    #     if (none_data.iloc[i] == True) and (switch == 0):
    #         tuple_ = (i,)
    #         switch = 1
    #     if (none_data.iloc[i] == False) and (switch == 1):
    #         tuple_ = tuple_ + (i,)
    #         switch = 0
    #         missing_range.append(tuple_)
    #         tuple_ = tuple
    # for start, end in missing_range:
    #     plt.plot(range(start - 1, end + 1), result[start - 1:end + 1], label=lb, color="orange")
    #     lb = None
    # plt.legend()
    # plt.show()

    # 예측 그래프
    plt.figure(figsize=(15, 5))
    plt.plot(df["value"], label="real", zorder=10)
    plt.plot(result, label="predict", color="orange")
    plt.legend()
    plt.show()

    # loss 그래프
    plt.figure(figsize=(5, 5))
    plt.plot(loss_graphic)
    plt.show()
