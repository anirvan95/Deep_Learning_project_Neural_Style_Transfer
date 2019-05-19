import torch
import torchvision
import torchvision.transforms as transforms
import slim
import vgg
import torchvision.models as models

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


def distillation_loss(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(nn.functional.log_softmax(y / T), nn.functional.softmax(teacher_scores / T)) * (
                T * T * 2.0 * alpha) + nn.functional.cross_entropy(y, labels) * (1. - alpha)


import torch.nn as nn
import torch.nn.functional as F

bignet = vgg.__dict__['vgg19']()
bignet.features = torch.nn.DataParallel(bignet.features)
bignet.cuda()
#loading the pretrained weights of vgg on cifar10
checkpoint = torch.load('model_best.pth.tar')
bignet.load_state_dict(checkpoint['state_dict'])

smallnet = slim.SimpleConvNet()
smallnet.cuda()

import torch.optim as optim

criterion = distillation_loss
optimizer = optim.SGD(smallnet.parameters(), lr=0.001, momentum=0.9)

for epoch in range(3):  # loop over the dataset multiple times

    smallnet.train()
    bignet.eval()

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        input_var = torch.autograd.Variable(inputs).cuda()
        target_var = torch.autograd.Variable(labels).cuda()

        # Get output of the teacher model
        with torch.no_grad():
            teacher_output = bignet(input_var).data.cpu().numpy()
            teacher_output = torch.cuda.FloatTensor(teacher_output)

        # zero the parameter gradients


        # forward + backward + optimize
        student_output = smallnet(input_var)
        loss = criterion(student_output, target_var, teacher_output, T=20.0, alpha=0.7)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Distillation Training')

correct = 0
total = 0
#transferring trained model to cpu
smallnet.eval()
smallnet.cpu()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = smallnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
