import pandas as pd
import time


def setting(path, encoding="utf-8"):
    if encoding is True:
        df = pd.read_csv(path, encodinig=encoding)
    else:
        df = pd.read_csv(path)
    if len(df.columns) != 2:
        raise ValueError("Data must be columns size 2")
    df2 = pd.DataFrame()
    column = df.columns
    try:
        df2["time"] = pd.to_datetime(column[0], format="%Y%m%d%H%M")
    except:
        df2["time"] = df[column[0]]
    df2["value"] = df[column[1]]
    df2.to_csv("./data.csv", index=False)

if __name__ == "__main__":
    while True:
        name = input("csv파일 위치 입력(확장자까지 포함) : ")
        try:
            setting(name)
            break
        except:
            input("존재하지 않는 파일입니다. 종료를 원할시 'exit'를 입력해주세요")
    print("데이터 준비 완료 다음을 진행해주세요. 3초 후 자동 종료됩니다.")
    time.sleep(3)