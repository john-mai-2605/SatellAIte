import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

#DEFINE root dir
root = "../audios"
classes = ['urban', 'rural', 'uninhabited']
num_classes = len(classes)

transform_train = transforms.Compose([
    transforms.Resize((40,40)),
    transforms.RandomCrop((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
])
transform_test = transforms.Compose([
    transforms.Resize((40,40)),
    transforms.CenterCrop((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.20)),
])

trainset = datasets.ImageFolder('./grid/train',transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = datasets.ImageFolder('./grid/test',transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)	

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=32,
    shuffle=False,
)

model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

NUM_EPOCHS = 10
best_accuracy = 0.0
model_path = root + '/model.pt'

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(NUM_EPOCHS):
    
    for images, labels in iter(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    test_error_count = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
    
    test_accuracy = 1.0 - float(test_error_count) / float(len(testset))
    print('Epoch %d: Accuracy is %f' % (epoch, test_accuracy))
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), model_path)
        best_accuracy = test_accuracy
