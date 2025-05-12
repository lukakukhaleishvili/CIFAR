import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 მონაცემების ჩატვირთვა
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# კლასების სახელები
classes = trainset.classes

# კონვოლუციური ქსელი
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# გაწვრთნა
for epoch in range(10):  # 10 ეპოქა
    net.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.3f}")

# ტესტი
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test images: {accuracy:.2f}%')


def visualize_filters(layer):
    weights = layer.weight.data.cpu().clone()
    if weights.shape[1] > 1:
        weights = weights[:, 0:1, :, :] 
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    grid = torchvision.utils.make_grid(weights, nrow=8, padding=1)
    npimg = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(npimg)
    plt.axis('off')
    plt.title("Visualized Filters")
    plt.show()

visualize_filters(net.conv1)
visualize_filters(net.conv2)


from torchvision.utils import make_grid

def get_activations(image, model, layers):
    activations = {}
    x = image.unsqueeze(0).to(device)

    for name, layer in model.named_children():
        x = layer(x) if name != 'relu' else x
        if name in layers:
            activations[name] = x.detach().cpu()

    return activations

# ერთი სურათის აქტივაცია
dataiter = iter(testloader)
images, labels = next(dataiter)
img = images[0]

activs = get_activations(img, net, ['conv1', 'conv2'])

# ვიზუალიზაცია
for name, act in activs.items():
    fig, axarr = plt.subplots(4, 4, figsize=(8, 8))
    for idx in range(16):
        axarr[idx//4, idx%4].imshow(act[0, idx].numpy(), cmap='hot')
        axarr[idx//4, idx%4].axis('off')
    plt.suptitle(f'Activation Map - {name}')
    plt.show()


# შავბნელი ნაწილი სურათზე
mod_img = img.clone()
mod_img[:, 10:20, 10:20] = 0  # ნაწილი წაიშალა

mod_activs = get_activations(mod_img, net, ['conv1'])

# შევადაროთ
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(activs['conv1'][0, 0].numpy(), cmap='hot')
axs[0].set_title("Original")
axs[1].imshow(mod_activs['conv1'][0, 0].numpy(), cmap='hot')
axs[1].set_title("Modified")
plt.show()


from torch.autograd import Variable

def generate_adversarial_example(model, image, label, epsilon=0.01):
    image.requires_grad = True
    output = model(image.unsqueeze(0).to(device))
    loss = criterion(output, torch.tensor([label]).to(device))
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    perturbed_image = image + epsilon * data_grad.sign()
    return perturbed_image.detach()

adv_img = generate_adversarial_example(net, img.clone().to(device), labels[0].item())

# მოდელის პასუხი
net.eval()
output = net(adv_img.unsqueeze(0))
_, pred = torch.max(output, 1)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(img.permute(1, 2, 0).numpy())
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(adv_img.cpu().permute(1, 2, 0).numpy())
plt.title(f"Adversarial (pred: {classes[pred.item()]})")
plt.axis('off')
plt.show()
