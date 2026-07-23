from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

#실험 환경 설정
#ArgumentParser 생성
parser = argparse.ArgumentParser(description='VAE MNIST Example') #argparse 명령어로 옵션을 입력 받기 위한 파이썬 라이브러리

#batch-size 옵션
# --batch-size라는 옵션을 추가한 것
#e.g.) python main.py --batch-size 64 형태로 사용 가능. default 값은 128.
parser.add_argument('--batch-size', type=int, default=128, metavar='N', 
                    help='input batch size for training (default: 128)') 

#epochs 옵션
# --epochs라는 옵션을 추가한 것
#e.g.) python main.py --epochs 20 형태로 사용 가능. default 값은 10.
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
# --no-accel 옵션. CPU만 사용할 시 사용. default 값은 False.
parser.add_argument('--no-accel', action='store_true', 
                    help='disables accelerator')

#seed 옵션. random seed를 설정하는 옵션. default 값은 1.
# latent sampling, weight initialization 등에서 random seed를 고정하여 재현성을 확보하기 위해 사용.                   
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# 학습 상황 로그를 출력. batch 기준이며 default 값은 10.(batch마다 )                   
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

# args 객체 안에 모든 옵션이 저장됨
args = parser.parse_args()

# no_accel 옵션 사용 && 가속기 사용 가능하면 use_accel = True, 아니면 False
use_accel = not args.no_accel and torch.accelerator.is_available()

torch.manual_seed(args.seed) #재현성 확보


if use_accel:#device 선택
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_accel else {}
train_loader = torch.utils.data.DataLoader( #data를 일정한 크기의 betch로 나누어 모델에 전달하는 객체
    # MNIST 데이터셋 다운로드 및 로드
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),# PIL image -> Tensor  
    batch_size=args.batch_size, shuffle=True, **kwargs) #betch size 128로 설정해둠
#위에선 train데이터 사용. test_loader는 test데이터 사용
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)


class VAE(nn.Module):
    def __init__(self): #생성자
        super(VAE, self).__init__() #부모 클래스인 nn.Module의 초기화 먼저 실행
        #encoder층. 입력차원(x_i),출력차원, 선형변환
        self.fc1 = nn.Linear(784, 400)
        #평균 mu(x_i) 출력 층
        self.fc21 = nn.Linear(400, 20)
        #분산 logvar(x_i) 출력 층
        self.fc22 = nn.Linear(400, 20)

        #decoder층. 입력차원(z_i),출력차원, 선형변환
        self.fc3 = nn.Linear(20, 400)
        #decoder 출력층. 최종이미지 복원.
        self.fc4 = nn.Linear(400, 784)

    # x_i -> z의 분포 파라미터. mu(x_i), logvar(x_i)
    def encode(self, x):
        h1 = F.relu(self.fc1(x)) #fc1층 통과 후 ReLU 활성화 함수 적용. ReLU(a) = max(0,a)
        return self.fc21(h1), self.fc22(h1) #mu(x_i), logvar(x_i) 반환

    #q(z|x) = N(μ, σ²)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) #log(a^2)이니까
        eps = torch.randn_like(std) #std와 동일한 shape의 난수 텐서
        return mu + eps*std

    #z -> x^ 복원
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        #위에 ToTensor()에서 0~1로 정규화->sigmoid로 0~1로 복원
        return torch.sigmoid(self.fc4(h3))
    #x_i -> z -> x^
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784)) #원래 MNIST의 shape은 (batch_size, 1, 28, 28)인데, fc1층에 넣기 위해 (batch_size, 784)로 변환
        z = self.reparameterize(mu, logvar)
        #x^, mu, logvar 반환. mu, logvar는 loss 계산에 사용됨.
        return self.decode(z), mu, logvar


model = VAE().to(device)#모델 객체 생성, 모델의 파라미터 -> device로 이동
optimizer = optim.Adam(model.parameters(), lr=1e-3) #옵티마이저


# Reconstruction + KL divergence losses = 최종 loss
def loss_function(recon_x, x, mu, logvar): #x^[128,784], x[128,1,28,28], mu[128, 20], logvar[128, 20]
    #x-x^, ELBO중 E_q(z;pai)log(p(x|z)) 부분. reconstruction loss
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # -E_q(z;pai)[D_KL(q(z|x)||p(z))].즉, KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # -1/2 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return BCE + KLD #스칼라 텐서


def train(epoch):
    model.train()#학습 모드로 바꿈
    train_loss = 0 #epoch들의 loss 누적저장 변수
    for batch_idx, (data, _) in enumerate(train_loader): #(data, target)을 batch 단위로 제공, _ = target은 사용하지 않음
        data = data.to(device) #data -> device로 이동
        optimizer.zero_grad() #이전 step의 gradient 초기화
        # forward pass
        recon_batch, mu, logvar = model(data) #[128,1,28,28] -> 모델에 맞는 shape으로 변환/ 입력
        loss = loss_function(recon_batch, data, mu, logvar) #batch 하나에 대한 loss
        loss.backward() #학습 가능한 파라미터의 gradient 계산
        train_loss += loss.item() #batch loss들의 누적합
        optimizer.step() #adam optimizer로 파라미터 update
        if batch_idx % args.log_interval == 0: # log_interval 배수마다 로그 출력
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
        #data = batch 사이즈
    # 해당 epoch의 평균 loss 출력   
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

# 성능평가, reconstruction 결과 저장
def test(epoch):
    model.eval() #평가, 추론모드
    test_loss = 0 #전체 test에 대한 loss 누적저장 변수 초기화
    with torch.no_grad(): #추론시 backprop 사용x.
        for i, (data, _) in enumerate(test_loader):# tesr data 10000개, batch=128
            data = data.to(device) #data -> device로 이동
            recon_batch, mu, logvar = model(data) #[128,1,28,28] -> 모델에 맞는 shape으로 변환/ 입력
            #batch에 대한 loss계산 후 누적합
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8) #data = batch 사이즈, 이미지를 몇장 선택할거냐->n
                #원본 이미지 shape으로 변환, 원본 이미지와 recon된 이미지를 이어붙여 저장 8->16장
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                #파일 저장
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)
    #10000장중 장단 평균 loss
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

#실험 시작
if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad(): #VAE의 생성 능력 확인. backpropagation 필요 없음.
            sample = torch.randn(64, 20).to(device) #z 64개 (20차원)
            sample = model.decode(sample).cpu() #sample -> decoder -> x^
            save_image(sample.view(64, 1, 28, 28), #이미지 원본 shape으로 저장
                       'results/sample_' + str(epoch) + '.png')
