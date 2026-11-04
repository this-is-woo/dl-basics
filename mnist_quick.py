import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

if torch.cuda.is_available():
    device = "cuda"                  # NVIDIA GPU (데스크탑)
elif torch.xpu.is_available():
    device = "xpu"                   # Intel GPU (갤럭시북6 Pro)
else:
    device = "cpu"
print("사용 장치:", device)

tf = transforms.ToTensor()
train = datasets.MNIST("data", train=True, download=True, transform=tf)
test = datasets.MNIST("data", train=False, download=True, transform=tf)
train_dl = DataLoader(train, batch_size=128, shuffle=True)
test_dl = DataLoader(test, batch_size=1000)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256), nn.ReLU(),
    nn.Linear(256, 10),
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    print(f"epoch {epoch + 1}: 테스트 정확도 {correct / len(test):.2%}")