import torch
import math
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import ujson as json


# 주어진 값에 대하여 연속성의 강도를 저장하기 위하여 다음을 작성(변화율)
def parse_delta(masks, dir_):
    # 만약 dir_가 backward로 주어졌을 때
    if dir_ == 'backward':
        # masks 값을 뒤집는다.
        masks = masks[::-1]
    # deltas를 저장할 리스트를 선언한다.
    deltas = []
    # masks의 길이만큼을 실행한다.
    for i in range(len(masks)):
        # 초기값 선언
        if i == 0:
            deltas.append(1)
        # 만약 연속되는 값이 존재할 경우 기존 값에서 1을 더해준다.
        # 만약 연속되는 값이 끊길 경우에는 모든 값을 버리고 1로 초기화한다.
        else:
            deltas.append(1 + (1 - masks[i]) * deltas[-1])
    # 이때의 값을 Numpy값으로 저장한다.
    return np.array(deltas)

# 데이터의 정보를 저장하는 함수
def parse_rec(values, masks, evals, eval_masks, dir_):
    # 데이터의 변화율(기울기)를 저장하기 위하여 다음 함수를 불러준다.
    deltas = parse_delta(masks, dir_)
    # values의 비어있는 값을 근접값으로 채운다. 앞 값으로 채우고 뒷값으로 채우며 만든다. 이후 valuses 값만 저장한다.
    # 2번 채우는 이유? : 결측값이 첫값이나 끝 값일 경우 안채워지는 것을 막기 위하여.
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(method="bfill").values
    # 값을 저장할 Dictionary를 선언
    rec = {}
    # 실제값을 저장
    # values의 nan값을 0num으로 대체하여 넣고 리스트로 저장한다.(Numpy배열은 json화 되지 않음)
    rec['values'] = np.nan_to_num(values).tolist()
    # 이때의 비어있는 값과 있는 값을 int값으로 치환(0, 1)한 후 리스트로 저장한다.
    rec['masks'] = masks.astype('int32').tolist()
    # 이때의 연속성 강도를 저장한다.
    rec['deltas'] = deltas.tolist()
    # 원하는 수치 저장(Ground-Truth)
    # evals의 값을 리스트로 저장
    rec['evals'] = np.nan_to_num(evals).tolist()
    # 이때의 비어있는 값과 있는 값을 int값으로 치환(0, 1)한 후 리스트로 저장한다.
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    # values에서 간단한 식으로 값을 채운 값을 리스트로 저장한다.
    rec['forwards'] = forwards.tolist()
    # 이 모든 값을 가진 Dictionary를 반환한다.
    return rec

# csv를 논문의 brits모델에 사용하기 위하여 json화 시켜준다.
def makedata(datapath):
    # 인자로 받은 위치의 파일을 Pandas의 데이터프레임으로 불러온다.
    df = pd.read_csv(datapath)
    # 현재 데이터가 가진 평균값을 저장한다.
    mean = df["value"].mean()
    # 현재 데이터가 가진 표준편차를 저장한다.
    std = df["value"].std()
    # 정규화 식을 위한 리스트를 선언(Z-transform)
    evals = []
    # df의 길이만큼
    for i in range(len(df)):
        # evals 리스트에 df의 value값을 저장한다.
        evals.append(df["value"].iloc[i])
    # 이때 모든 값에 대하여 정규화를 실시한다.
    # 이를 위하여 evals 리스트를 Numpy해준다.(해줌으로서 단순 계산식으로 리스트 전체가 적용됨)
    evals = (np.array(evals) - mean) / std
    # Numpy 배열을 복사본을 준비해준다.
    values = evals.copy()
    # 복사한 Numpy 배열에서 nan값이 아닌 것을 True로 반환한다. 그리고 전체를 반전시킨다.(~ : NOT 연산자)
    # 전체적인 형태는 값이 있는 것들이 True로 반환.
    masks = ~np.isnan(values)
    # evals Numpy 배열을 전체 True로 반환한 상태에서 XOR(^ : XOR 연산자)연산을 이용하여 전체가 False인 Numpy 배열을 만든다.
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))
    # 값을 정리할 Dictionary를 임의로 만든다.
    rec = {'label': 0}
    # 정방향과 순방향에 대한 Dictionary Key값에 다음을 value값으로 한다.
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')
    # 이때의 rec값을 json으로 저장한다.
    rec = json.dumps(rec)
    with open("./dataset.json", "w") as fs:
        fs.write(rec)


# 실제 사용할 데이터의 형태를 만들도록 Dataset 클래스를 커스텀 한다.
class MySet(Dataset):
    # 초기화 함수
    def __init__(self, file):
        # 자신과 상속받은 초기화 함수를 실행한다.
        super(MySet, self).__init__()
        # 파일을 불러와 라인을 읽어 변수에 저장한다.
        self.content = open(file).readlines()
        # 이때 변수의 길이만큼을 Numpy 배열을 만들어 변쉥 저장한다.
        indices = np.arange(len(self.content))
        # 20%를 validation 데이터로 만든다. replace인자를 False로 선언해주어 중복 뽑기를 제거한다.
        self.val_indices = np.random.choice(indices, len(self.content) // 5, replace=False)

    # len 메소드를 구현한다.
    def __len__(self):
        return len(self.content)

    # indexing, slicing등을 위한 메소드
    def __getitem__(self, idx):
        # 파일라인을 불러온 정보의 데이터가
        rec = json.loads(self.content[idx])
        # validation 데이터 안에 있다면
        if idx in self.val_indices:
            # 0을 반환하고
            rec['is_train'] = 0
        # 없다면
        else:
            # 1을 반환한다.(True or False)
            rec['is_train'] = 1
        # 적용된 값을 반한환다.
        return rec

def collate_fn(recs):
    # recs(Dictionary)에서 map함수로 Key값을 forward로 가지는 value를 받아 리스트화 한다.
    forward = list(map(lambda x: x['forward'], recs))
    # 위와 동일하지만 Key값은 forward이다.
    backward = list(map(lambda x: x['backward'], recs))

    # Pytorch에서 작동하도록 하기 위해서 데이터 배열을 Tensor화 해준다.
    def to_tensor_dict(recs):
        # 실제 값들의 데이터를 Tensor화
        # 반환받은 데이터를 Tensor화 시킬 때 unsqueeze 함수를 사용하여 끝에 차원을 추가해준다.
        values = torch.FloatTensor(list(map(lambda r: r['values'], recs))).unsqueeze(-1)
        masks = torch.FloatTensor(list(map(lambda r: r['masks'], recs))).unsqueeze(-1)
        deltas = torch.FloatTensor(list(map(lambda r: r['deltas'], recs))).unsqueeze(-1)
        # 원하는 값들의 데이터를 Tensor화(Ground-Truth)
        # 위와 동일하게 차원을 추가해준다.
        evals = torch.FloatTensor(list(map(lambda r: r['evals'], recs))).unsqueeze(-1)
        eval_masks = torch.FloatTensor(list(map(lambda r: r['eval_masks'], recs))).unsqueeze(-1)
        forwards = torch.FloatTensor(list(map(lambda r: r['forwards'], recs))).unsqueeze(-1)
        # 위으 데이터 Tensor들을 가지고 있는 Dictionary 자료형을 반환한다.
        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}

    # 순방향과 역방향에 대한 데이터 Tensor를 Dictionary로 한번더 감싸 저장한다.
    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}
    # 해당 Dictionary에 labels와 is_train에 대한 값을 Tensor화 해서 남겨준다.
    ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs))).unsqueeze(-1)
    ret_dict['is_train'] = torch.FloatTensor(list(map(lambda x: x['is_train'], recs))).unsqueeze(-1)
    # 이러한 데이터 처리과정을 반환한다.
    return ret_dict

# get_loader함수를 만든다.
def get_loader(file, batch_size = 64, shuffle = True):
    # 파일 위치를 받아 데이터셋화 시킨다.
    data_set = MySet(file)
    # 이때의 데이터를 배치사이즈에 맞게 섞고 반환한다. defalt값은 64
    data_iter = DataLoader(dataset = data_set, batch_size = batch_size,num_workers = 4,
                           shuffle = shuffle, pin_memory = True, collate_fn = collate_fn)
    # 이를 data_iter로 반환한다.
    return data_iter

def to_var(var, device):
    if torch.is_tensor(var):
        # 현재 모든 Tensor는 autograd가 적용되어 있으며 그 하위 메소드 또한 적용 된다.
        # var = Variable(var)
        # Tensor에 Device를 설정(차후 GPU사용에 사용)
        var = var.to(device)
        # 해당 Tensor의 상태를 반환
        return var
    # 만약에 var이 int나, float, str일 경우에는
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        # 특별히 적용할 것 없이 반환합니다.
        return var
    # 만약 var이 Dictionary일 경우에는
    if isinstance(var, dict):
        # Key값에 대하여
        for key in var:
            # Value값을 뽑고 재귀함수로 Value값에 to_var함수를 사용한다.
            var[key] = to_var(var[key], device)
        # 이때의 var을 반환한다.
        return var
    # 만약 var이 list인 경우에는
    if isinstance(var, list):
        # list의 요소별로 재귀함수로 to_var함수를 사용한다.
        var = map(lambda x: to_var(x, device), var)
        # 이를 반환한다.
        return var


# 단일 독립변수에 대한 것이므로 Brits_i를 사용함.
class Brits_i(nn.Module):
    # 초기화 함수
    def __init__(self, rnn_hid_size, impute_weight, label_weight, seq_len, device):
        # 자신과 상속받은 초기화 함수를 실행한다.
        super(Brits_i, self).__init__()
        # 받은 인자들을 클래스 내부 변수로 저장
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.seq_len = seq_len
        self.device = device
        # Brits_I를 설계
        self.build()

    # Brits_I 모델의 양방향 설계를 위한 함수
    def build(self):
        # 순방향과 역방향으로 사용할 Rits_I 모델을 Instance로 선언
        self.rits_f = Rits_i(self.rnn_hid_size, self.impute_weight, self.label_weight, self.seq_len, self.device)
        self.rits_b = Rits_i(self.rnn_hid_size, self.impute_weight, self.label_weight, self.seq_len, self.device)

    # model 진행시 사용되는 함수
    def forward(self, data):
        # 선언한 Rits_I 모델을 실행
        ret_f = self.rits_f(data, 'forward')
        # 역방향의 경우 결과를 역방향으로 하는 함수를 실행
        ret_b = self.reverse(self.rits_b(data, 'backward'))
        # 단일의 결과를 반환하기 위하여 둘의 값을 병합해줌.
        ret = self.merge_ret(ret_f, ret_b)
        # 결과를 반환함
        return ret

    # Rits_I의 결과를 병합하는 함수
    def merge_ret(self, ret_f, ret_b):
        # 두 결과의 변수를 각각 저장한다.
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        # 결측값 대체를 한 값들을 이용하여 함수를 통해 서로간의 loss를 구한다.
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])
        # 이때의 각각의 loss율을 구해 최종 loss를 반환한다.
        # 이로소 loss를 낮추기 위해서는 순방향 역방향 loss율이 낮은과 동시에 서로 동일해지도록 유도됨.
        loss = loss_f + loss_b + loss_c
        # 예측값과 imputations된 값들을 더하고 2로 나눠서 반환한다.
        # 두방향에 대한 값을 한개의 모델로서의 값으로 받기 위해서 이다.
        predictions = (ret_f['predictions'] + ret_b['predictions']) / 2
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        # 이렇게 작성된 Brits_I의 값을 반환하도록 한다.
        # 나머지 값은 같은 값이므로 ret_f의 값을 복사한다.
        ret = ret_f.copy()
        # 그 중 위에서 구한 값들만 대체해서 입력한다.
        ret['loss'] = loss
        ret['predictions'] = predictions
        ret['imputations'] = imputations
        # 이 값을 반환한다.
        return ret

    # 두 Imputation 된 값의 차이값을 통하여 loss율을 구하는 함수
    def get_consistency_loss(self, pred_f, pred_b):
        # 각 결과 tensor에서 두 값의 차의 절대값들의 평균값에 1/10을 곱해 loss율을 냄
        # 근데 왜 1/10이지?
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        # 이때의 loss율을 반환함.
        return loss

    # 역방향값 산출을 위한 함수
    def reverse(self, ret):
        # Tensor를 역방향으로 만드는 함수
        def reverse_tensor(tensor_):
            # 만약 인자로 받은 Tensor의 차원이 1차원보다 작거나 같다면
            if tensor_.dim() <= 1:
                # 차원을 반환하고
                return tensor_
            # 만약 크다면 Tensor크기만큼의 index리스트를 역순으로 만들고
            indices = range(tensor_.size()[1])[::-1]
            # 이를 Tensor화 시키며 학습에서 제외시킨다.
            indices = torch.LongTensor(indices).requires_grad_(False).to(self.device)
            # 역순으로 만든 결과에 맞게 Tensor를 역순화 시킨 값을 반환시킨다.
            return tensor_.index_select(1, indices)

        # Key 갯수만큼을
        for key in ret:
            # 위 함수를 사용하여 Tensor를 역순화 시킨다.
            ret[key] = reverse_tensor(ret[key])
        # 이때의 결과를 반환한다.
        return ret

    # 학습시 사용되는 함수
    def run_on_batch(self, data, optimizer, epoch=None):
        # 모델로부터 받은 값을 ret에 저장하고
        ret = self(data)
        # 만약 옵티마이저가 있다면
        if optimizer is not None:
            # 옵티마이저의 값 누적을 피하고
            optimizer.zero_grad()
            # 받은 값`의 loss를 역전파하고
            ret['loss'].backward()
            # 옵티마이저가 최적화한다.
            optimizer.step()
        # 이때의 결과를 다시 반환한다.
        return ret


# Rits_I에서 이진 크로스엔트로피를 구하는 식
def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    # 만약 입력값의 사이즈와 출력값의 사이즈가 같지 않다면
    if not (target.size() == input.size()):
        # 다음 오류를 출력시킨다.
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
    # 입력값의 Tensor를 - 대입하여 음수와 양수를 변경한 다음 이중 0보다 작은 수들을 0으로 바꾼다.
    # input
    max_val = (-input).clamp(min=0)
    # 이진 크로스엔트로피를 구하는 공식을 사용하여 다음을 구함.
    # exp는 지수함수
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    # 기존 가중치가 존재한다면 loss에 가중치를 곱한다.
    if weight is not None:
        loss = loss * weight
    # 만약 reduce가 True가 아닐 경우에
    if not reduce:
        # loss를 반환하고
        return loss
    # size_average가 True일 경우에는
    elif size_average:
        # loss의 평균값을 반환하고
        return loss.mean()
    # 만약 두 조건이 아닐 경우에는
    else:
        # loss의 합을 출력한다.
        return loss.sum()

# Rits_I에 사용될
class TemporalDecay(nn.Module):
    # 초기화 함수
    def __init__(self, input_size, rnn_hid_size):
        # 자신과 상속받은 초기화 함수를 실행한다.
        super(TemporalDecay, self).__init__()
        # 받은 인자들을 클래스 내부 변수로 저장
        self.rnn_hid_size = rnn_hid_size
        # TemporalDecay의 모델을 설계
        self.build(input_size)

    # 모델 설계 함수
    def build(self, input_size):
        # 가중치에 대한 매개변수를 저장하는 변수 선언
        self.W = Parameter(torch.Tensor(self.rnn_hid_size, input_size))
        # 절편에 대한 매개변수를 저장하는 변수 선언
        self.b = Parameter(torch.Tensor(self.rnn_hid_size))
        # 자체 함수로 설정된 두변수를 재설정함
        self.reset_parameters()

    # reset함수
    def reset_parameters(self):
        # 주어진 가중치의 크기만큼의 제곱근을 반환하여 이를 분모로 가지는 변수를 만듬
        stdv = 1. / math.sqrt(self.W.size(0))
        # 이때의 가중치를 다음 값으로 초기화 한다.
        self.W.data.uniform_(-stdv, stdv)
        # 절편이 존재한다면
        if self.b is not None:
            # 다음과 같이 절편도 초기화 해준다.
            self.b.data.uniform_(-stdv, stdv)

    # model 진행시 사용되는 함수
    def forward(self, d):
        # 선형회구분석으로 다음을 분석한 뒤 relu함수로 감마값을 반환받는다.
        gamma = F.relu(F.linear(d, self.W, self.b))
        # 이때의 감마값의 마이너스 값의 지수를 저장하고
        gamma = torch.exp(-gamma)
        # 이 지수값을 반환한다.
        return gamma


class Rits_i(nn.Module):
    # 초기화 함수
    def __init__(self, rnn_hid_size, impute_weight, label_weight, seq_len, device):
        # 자신과 상속받은 초기화 함수를 실행한다.
        super(Rits_i, self).__init__()
        # 받은 인자들을 클래스 내부 변수로 저장
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.seq_len = seq_len
        self.device = device
        # Rits-I 모델을 설계
        self.build()

    # Model설계
    def build(self):
        # 논문에서의 input value. 실질적으로 1개의 값을 받는다.
        self.input_size = 1
        # 논문에서의 recurrent layer. LSTM으로 작성되었으며 기존 RNN의 gradient vanishing 문제를 해결한다.
        # 결측치가 길어지면서 장기기억이 필요할 수도 있기 때문에. (은닉 상태의 값을 반환)
        self.rnn_cell = nn.LSTMCell(self.input_size * 2, self.rnn_hid_size)
        # LSTMcell을 거쳐온 데이터를 선형회귀 시킨다. 대체값을 얻게 된다.(은닉 상태의 값을 예측 벡터로 전환)
        self.regression = nn.Linear(self.rnn_hid_size, self.input_size)
        # 대체값을 이용하여 temporal decay로 감마값을 구한다. 이 감마값을 이용하여 다음 층으로 넘긴다.
        self.temp_decay = TemporalDecay(input_size = self.input_size, rnn_hid_size = self.rnn_hid_size)
        # 최종 출력을 위한 선현회귀 후 다음 층으로 값을 넘긴다.
        self.out = nn.Linear(self.rnn_hid_size, 1)

    # model 진행시 사용되는 함수
    def forward(self, data, direct):
        # Brits-I를 구성하기 위하여 추가로 direct 인자를 받도록 설계. direct는 'forward'와 'backward'를 맡는 부분이다.
        # 데이터의 각 부분에서 해당하는 부분을 가져와 함수의 지역변수로 사용하도록 한다.
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']
        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']
        # x축이 1의 크기를 가진 2차원의, 벡터의 형태인 값으로 labels를 정의한다.
        labels = data['labels'].view(-1, 1)
        # 동일한 형태로 is_train을 정의한다.
        is_train = data['is_train'].view(-1, 1)
        # 은닉상태를 저장할 Tensor를 0을 가진 행렬로 생성(입력을 위하여 은닉층의 크기만큼을 x축으로 가짐)
        h = torch.zeros((values.size()[0], self.rnn_hid_size))
        # 셀상태를 저장할 Tensor를 0을 가진 행렬로 생성(입력을 위하여 은닉층의 크기만큼을 x축으로 가짐)
        c = torch.zeros((values.size()[0], self.rnn_hid_size))
        # 이때 두 층을 GPU를 사용하도록 바꾼다.
        h, c = h.to(self.device), c.to(self.device)
        # x_loss와 y_loss값을 저장하도는 변수 선언
        x_loss = 0.0
        # imputation한 값들을 저장할 리스트 선언
        imputations = []
        # 주어진 데이터의 전체 길이만큼을 반복한다.
        for t in range(self.seq_len):
            # 각 데이터의 부분을 가져와 특정 변수로 지정한다.
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            # 델타값을 temporal decay층을 이용하여 감마값으로 치환한다.
            gamma = self.temp_decay(d)
            # 은닉상태에 감마값을 곱한다.
            h = h * gamma
            # 이때 은닉상태의 값을 선형회귀 시켜 값을 저장한다.
            x_h = self.regression(h)
            # 세포상태의 값을 데이터의 값이 결측값일 때 가중치가 강하게 남도록 설정하여 남긴다.
            x_c = m * x + (1 - m) * x_h
            # 이때의 x_loss값을 구한다.
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)
            # 두번째 차원을 기준으로 x_c와 m의 tensor를 합칩니다.
            inputs = torch.cat([x_c, m], dim=1)
            # 이를 LSTMcell에 집어 넣고 은닉상태값과 셀상태의 값을 받습니다.
            h, c = self.rnn_cell(inputs, (h, c))
            # 셀상태의 값을 차원을 추가하여 리스트에 추가한다.
            imputations.append(x_c.unsqueeze(dim=1))
        # 위 과정을 통해 완성된 리스트를 1차원을 기준으로 전부 합쳐준다.
        imputations = torch.cat(imputations, dim=1)
        # 이 때의 은닉상태를 출력층에 넣어 결과값을 출력 받는다.
        y_h = self.out(h)
        # 이때의 값을 정답과 비교하여 값을 2진 크로스엔트로피의 loss값을 받는다.
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce=False)
        # 주어진 loss값에 학습데이터에 대해서만 계산을 실시한다.(예측값을 loss 계산에 넣지 않도록 하기 위하여)
        y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)
        # 이때의 값을 시그모이드 함수로 은닉상태를 반환시킨다.
        y_h = torch.sigmoid(y_h)
        # 이렇게 주어진 식을 Dictionary형태로 반환한다.
        return {'loss': x_loss * self.impute_weight + y_loss * self.label_weight, 'predictions': y_h,
                'imputations': imputations, 'labels': labels, 'is_train': is_train,
                'evals': evals, 'eval_masks': eval_masks}


# 평가 함수
def evaluate(model, data_iter, device=torch.device("cpu")):
    # model 평가 모드로 전환
    model.eval()
    imputation = np.array([])
    # 배치데이터 만큼을
    for idx, data in enumerate(data_iter):
        # GPU를 설정해주고
        data = to_var(data, device)
        # model에 데이터를 넣고 실행해 준다음.
        ret = model.run_on_batch(data, None)
        imputation = ret['imputations'].data.cpu().numpy()
    # imputation값을 반환한다.
    return imputation


# 결과 예측 함수
def predict_result(model, data_iter, device, df):
    # imputation된 값을 반환받고
    imputation = evaluate(model, data_iter, device)
    scaler = StandardScaler()
    scaler = scaler.fit(df["value"].to_numpy().reshape(-1,1))
    result = scaler.inverse_transform(imputation[0])
    return result[:, 0]