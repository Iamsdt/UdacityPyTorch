import json
import time
import os
import random
import urllib
import zipfile
import sys

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import torch
from torch import optim
from torch.optim import lr_scheduler


def save_check_point(model, optimizer, scheduler, train_loader, path, epoch, save_cpu=False):
    if save_cpu:
        model = model.to('cpu')
        path = path.split('.')[0] + '-cpu.pth'

    cat_to_name = get_cat_name()
    class_to_idx = train_loader.dataset.class_to_idx
    idx_to_name = {idx: cat_to_name[category] for category, idx in class_to_idx.items()}
    checkpoint = {
        'cat_to_name': cat_to_name,
        'class_to_idx': class_to_idx,
        'idx_to_name': idx_to_name,
        'epochs': epoch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }

    torch.save(checkpoint, path)
    print(f"Model saved at {path}")


def load_checkpoint(filepath, model, lr=0.001, step=2, momentum=0.9, gamma=0.1):
    # Make sure to set parameters as not trainable
    for param in model.parameters():
        param.requires_grad = False

    # Load in checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath,map_location="cpu")


    # Extract classifier
    model.classifier = checkpoint['classifier']
    model.cat_to_name = checkpoint['cat_to_name']
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_name = checkpoint['idx_to_name']
    model.epochs = checkpoint['epochs']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # optimizer = optim.Adadelta(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')
    print(f'Model has been trained for {model.epochs} epochs.')

    return [model, optimizer, scheduler]


def freeze_parameters(model):
    for param in model.features.parameters():
        param.requires_grad = False

    return model


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

    return model


def prepare_loader(data_dir,
                   transform_train,
                   transform_valid,
                   test_transforms,
                   batch_size=20,
                   num_workers=0,
                   valid_size=0.2):
    # data set
    train_data = datasets.ImageFolder(data_dir + '/train', transform=transform_train)

    valid_data = datasets.ImageFolder(data_dir + '/train', transform=transform_valid)

    test_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    print("Train size:{}".format(num_train))
    print("Valid size:{}".format(len(valid_data)))
    print("Test size:{}".format(len(test_data)))

    # mix data
    # index of num of train
    indices = list(range(num_train))
    # random the index
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    # divied into two part
    train_idx, valid_idx = indices[split:], indices[:split]

    # define the sampler
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)

    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size,
        sampler=valid_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return [train_loader, valid_loader, test_loader]


def save_model(model, model_name="model.pt", path=None):
    p = F"/content/gdrive/My Drive/{model_name}"
    if path is not None:
        p = path

    torch.save(model.state_dict(), p)
    print("Saved successfully")


def load_model(model, model_name="model.pt", path=None):
    p = F"/content/gdrive/My Drive/{model_name}"
    if path is not None:
        p = path

    model.load_state_dict(torch.load(path))
    print("Model loaded successfully")
    print(model.classifier)
    return model


def get_cat_name():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name


def train_model(model,
                train_loader,
                valid_loader,
                n_epochs,
                optimizer,
                scheduler,
                criterion,
                name="model.pt",
                path=None):
    # compare overfited
    train_loss_data, valid_loss_data = [], []
    # check for validation loss
    valid_loss_min = np.Inf
    # calculate time
    since = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(n_epochs):
        print("Epoch: {}/{}".format(epoch + 1, n_epochs))
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
        total = 0
        correct = 0
        e_since = time.time()

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        scheduler.step()  # step up scheduler
        for images, labels in train_loader:
            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            log_ps = model(images)
            # calculate the loss
            loss = criterion(log_ps, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * images.size(0)

        ######################
        # validate the model #
        ######################
        print("\t\tGoing for validation")
        model.eval()  # prep model for evaluation
        for data, target in valid_loader:
            # Move input and label tensors to the default device
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss_p = criterion(output, target)
            # update running validation loss
            valid_loss += loss_p.item() * data.size(0)
            # calculate accuracy
            proba = torch.exp(output)
            top_p, top_class = proba.topk(1, dim=1)
            equals = top_class == target.view(*top_class.shape)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        # calculate train loss and running loss
        train_loss_data.append(train_loss * 100)
        valid_loss_data.append(valid_loss * 100)

        print("\tTrain loss:{:.6f}..".format(train_loss),
              "\tValid Loss:{:.6f}..".format(valid_loss),
              "\tAccuracy: {:.4f}".format(correct / total * 100))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('\tValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), name)
            valid_loss_min = valid_loss
            # save to google drive
            if path is not None:
                torch.save(model.state_dict(), path)

        # Time take for one epoch
        time_elapsed = time.time() - e_since
        print('\tEpoch:{} completed in {:.0f}m {:.0f}s'.format(
            epoch + 1, time_elapsed // 60, time_elapsed % 60))

    # compare total time
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # return the model
    return [model, train_loss_data, valid_loss_data]


def calculate_top_accuracy(model, test_loader, criterion, data_dir="flower_data", size=10):
    classes = os.listdir("{}/valid".format(data_dir))
    test_loss = 0
    class_correct = list(0. for i in range(102))
    class_total = list(0. for i in range(102))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        model.eval()  # prep model for evaluation
        for data, target in test_loader:
            # Move input and label tensors to the default device
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update test loss
            test_loss += loss.item()  # *data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            # calculate test accuracy for each object class
            for i in range(size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        # calculate and print avg test loss
        test_loss = test_loss / len(test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))

        for i in range(10):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    str(i), 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))


######################################################
# copy from https://github.com/GabrielePicco/deep-learning-flower-identifier
######################################################

def calc_accuracy(
        model,
        input_image_size=224,
        num_of_worker=0,
        use_google_testset=False,
        testset_path=None,
        batch_size=20,
        norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
    """
    Calculate the mean accuracy of the model on the test test
    :param use_google_testset: If true use the testset derived from google image
    :param testset_path: If None, use a default testset (missing image from the Udacity dataset,
    downloaded from here: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
    :param batch_size:
    :param model:
    :param input_image_size:
    :param norm_mean:
    :param norm_std:
    :return: the mean accuracy
    """
    if use_google_testset:
        testset_path = "./google_test_data"
        url = 'https://www.dropbox.com/s/3zmf1kq58o909rq/google_test_data.zip?dl=1'
        download_test_set(testset_path, url)
    if testset_path is None:
        testset_path = "./flower_data_orginal_test"
        url = 'https://www.dropbox.com/s/da6ye9genbsdzbq/flower_data_original_test.zip?dl=1'
        download_test_set(testset_path, url)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device=device)
    with torch.no_grad():
        batch_accuracy = []
        torch.manual_seed(33)
        torch.cuda.manual_seed(33)
        np.random.seed(33)
        random.seed(33)
        torch.backends.cudnn.deterministic = True
        datatransform = transforms.Compose([transforms.RandomRotation(45),
                                            transforms.Resize(input_image_size + 32),
                                            transforms.CenterCrop(input_image_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(norm_mean, norm_std)])
        image_dataset = datasets.ImageFolder(testset_path, transform=datatransform)
        dataloader = torch.utils.data.DataLoader(
            image_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_of_worker)

        for idx, (inputs, labels) in enumerate(dataloader):
            if device == 'cuda':
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model.forward(inputs)
            _, predicted = outputs.max(dim=1)
            equals = predicted == labels.data
            print("Batch accuracy (Size {}): {}".format(batch_size, equals.float().mean()))
            batch_accuracy.append(equals.float().mean().cpu().numpy())
        mean_acc = np.mean(batch_accuracy)
        print("Mean accuracy: {}".format(mean_acc))
    return mean_acc


def download_test_set(default_path, url):
    """
    Download a testset containing approximately 10 images for every flower category.
    The images were download with the download_testset script and hosted on dropbox.
    :param default_path:
    :return:
    """
    if not os.path.exists(default_path):
        print("Downloading the dataset from: {}".format(url))
        tmp_zip_path = "./tmp.zip"
        urllib.request.urlretrieve(url, tmp_zip_path, download_progress)
        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(default_path)
        os.remove(tmp_zip_path)


def download_progress(blocknum, blocksize, totalsize):
    """
    Show download progress
    :param blocknum:
    :param blocksize:
    :param totalsize:
    :return:
    """
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))
