import json
import time

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets


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
        time_elapsed = time.time() - since
        print('\tOne epoch complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    # compare total time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # return the model
    return [model, train_loss_data, valid_loss_data]
