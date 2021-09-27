import brits_I as B
import torch
import torch.optim as optim
import ujson as json
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    # 하이퍼 파라미터
    epoch = 100
    learning_rate = 0.01

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.random.manual_seed(0)
    np.random.seed(0)

    path = "./data.csv"
    B.makedata(path)
    df = pd.read_csv(path)
    length = len(df)
    model = B.Brits_i(108, 1, 0, length, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    data_iter = B.get_loader('./traffic.json', batch_size=64)
    model.train()
    progress = tqdm(range(epoch))

    loss_graphic = []
    for i in progress:
        tl=0.0
        for idx, data in enumerate(data_iter):
            data = B.to_var(data, device)
            ret = model.run_on_batch(data, optimizer, i)
            tl += ret["loss"]
        loss_graphic.append(tl.tolist())
        progress.set_description("loss: {:0.3f}".format(tl/len(data_iter)))

    result = B.predict_result(model, data_iter, device, df)
    real = pd.read_csv("서인천IC-부평IC 평균속도.csv",encoding='CP949')
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

    plt.figure(figsize=(15, 5))
    plt.plot(real["평균속도"], label="real")
    lb = "predict"
    missing_range = [(100, 105), (200, 210), (300, 320), (400, 430), (550, 600)]
    for start, end in missing_range:
        plt.plot(range(start-1,end+1), result[start-1:end+1], label=lb, color="orange")
        lb=None
    plt.legend()
    plt.show()

    plt.figure(figsize=(15,5))
    plt.plot(df["value"], label="real", zorder=10)
    plt.plot(result, label="predict", color="orange")
    plt.legend()
    plt.show()

    plt.figure(figsize=(5,5))
    plt.plot(loss_graphic)
    plt.show()