import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MNISTDataset(Dataset):
    def __init__ (self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.iloc[:,0].values
        self.images = self.data.iloc[:,1:].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28,28) / 255.0
        label = self.labels[idx]

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return image, label

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

        self.fc1 = nn.Linear(28*28*1, 128)

        self.fc2 = nn.Linear(128, 64)

        self.fc3 = nn.Linear(64, 10)

        # self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)

        x = torch.relu(self.fc2(x))
        # x = self.dropout(x)

        x = self.fc3(x)
        return x


train_dataset = MNISTDataset("/Users/jakuborzelowski/Desktop/MNIST_Pytorch/datasets/mnist_train.csv")
test_dataset = MNISTDataset("/Users/jakuborzelowski/Desktop/MNIST_Pytorch/datasets/mnist_test.csv")

print(f'Liczba przykładów: ({len(train_dataset)})')
image, label = train_dataset[9999]
print(image.shape)

images, labels = zip(*[train_dataset[i] for i in range(10)])  # Pobieramy 10 obrazów

fig, axes = plt.subplots(1, 10, figsize=(12, 12))
for i in range(10):
    axes[i].imshow(images[i].squeeze(), cmap="gray")
    axes[i].set_title(f'Label: {labels[i]}')
    axes[i].axis('off')
plt.show()

labels = [label for _, label in train_dataset]

df = pd.DataFrame(labels, columns=['class'])

train_class_distribution = df['class'].value_counts(normalize=True) * 100

plt.figure(figsize=(10, 5))
sns.barplot(x=train_class_distribution.index, y=train_class_distribution.values)
plt.title("Procentowy rozkład klas w zbiorze treningowym")
plt.xlabel("Klasa")
plt.ylabel("ilosc")
plt.show()

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = SimpleNN()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.view(-1, 28*28*1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoka [{epoch + 1}/{num_epochs}], Strata: {running_loss / len(train_loader)}')


model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, 28*28*1)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Dokładność na zbiorze testowym: {accuracy:.2f}%')