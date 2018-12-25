import json
import os
import random
import sys
import time
import urllib.request
import zipfile
import matplotlib as plt
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import seaborn as sns


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
                device,
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

    # loop complete
    plt.plot(train_loss_data, label="taining loss")
    plt.plot(valid_loss_data, label="validation loss")
    plt.legend(frameon=False)


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


def plot_solution(image_path, model):
    """
    Plot an image with the top 5 class prediction
    :param image_path:
    :param model:
    :return:
    """
    # Set up plot
    plt.figure(figsize=(6, 10))
    ax = plt.subplot(2, 1, 1)
    # Set up title
    flower_num = image_path.split('/')[3]
    cat_to_name = get_cat_name()
    title_ = cat_to_name[flower_num]
    # Plot flower
    img = process_image(image_path)
    imshow(img, ax, title=title_);
    # Make prediction
    probs, labs, flowers = predict(image_path, cat_to_name, model)
    # Plot bar chart
    plt.subplot(2, 1, 2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    plt.show()


# Class Prediction

def predict(image_path, cat_to_name, model, top_num=5):
    """
    Predict the class of an image, given a model
    :param image_path:
    :param model:
    :param top_num:
    :return:
    """
    # Process image
    img = process_image(image_path)

    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)

    image_tensor.to('cpu')
    model_input.to('cpu')
    model.to('cpu')

    # Probs
    probs = torch.exp(model.forward(model_input))

    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in
                    model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers


def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch
    model, returns an Numpy array
    """
    # Open the image
    from PIL import Image
    img = Image.open(image_path)
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,
                    top_margin))
    # Normalize
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))

    return img


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is first
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax
