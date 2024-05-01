import torch
import wandb
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import transforms, datasets
from Kuzushijidataset import Kuzushijidataset

wandb.init(
    # set the wandb project where this run will be logged
    project="hiragana",
    entity="junta",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "dataset": "k49",
    "epochs": 100
    }
)


# Load pretrained MobileNet
model = models.mobilenet_v3_large(weights=None)

# Modify the classifier
model.features[0][0] = nn.Conv2d(in_channels=1, out_channels=model.features[0][0].out_channels, kernel_size=model.features[0][0].kernel_size)
model.classifier[-1] = torch.nn.Linear(in_features=model.classifier[-1].in_features, out_features=49)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = Kuzushijidataset("k49-dataset", train=True, download=True, transform=transform)
test_data = Kuzushijidataset("k49-dataset", train=False, download=True, transform=transform)

print("Train dataset size: ", len(train_data))
print("Test dataset size: ", len(test_data))

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=128, shuffle=True) 
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=128, shuffle=False)

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Optimizer (You can use Adam or another suitable optimizer)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.
    https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py
    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# Number of epochs
num_epochs = 100
losses = []
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
        epoch_loss = 0
        total_acc = 0
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")
            data, target = data.to(device), target.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
    
            # Forward + backward + optimize
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            
            # acc = multi_acc(outputs, target)
            acc = accuracy_fn(y_true=target, y_pred=outputs.argmax(dim=1))
            total_acc += acc
            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item(), accuracy=f'{acc:.2f}')
        losses.append(epoch_loss)

    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy
    test_loss, test_acc = 0, 0 
    train_loss = epoch_loss / len(train_loader)
    model.eval()
    with torch.inference_mode():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
           
            # 2. Calculate loss (accumatively)
            test_loss += criterion(test_pred, y) # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        
        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_loader)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_loader)


    
    ### Testing diffeerent dataset accuracy
    image_dir = "hiragana_images/"
    dataset = datasets.ImageFolder(root=image_dir, transform=transform)
    # label_map = hiragana_images: k49
    label_map = {0: 0, 1: None, 2: 16, 3: None, 4: 3, 5: 27, 6: 25, 7: 28, 8: 26, 9: 29, 10: 1, 11: None, 12: 5, 13: 8, 14: 6, 15: 9, 16: 7, 17: 30, 18: 33, 19: 31, 20: 34, 21: 32, 22: 47, 23: 20, 24: 23, 25: 21, 26: 24, 27: 22, 28: 4, 29: None, 30: 38, 31: 41, 32: 39, 33: 42, 34: 40, 35: 10, 36: 13, 37: 11, 38: 14, 39: 12, 40: 15, 41: 18, 42: 19, 43: 17, 44: 2, 45: 43, 46: 46, 47: 35, 48: 37, 49: 36}
    
    correct = 0
    total = 0
    model.eval()
    with torch.inference_mode():
        for i in range(len(dataset)):
            img, label = dataset[i]
            img = img.to(device)
            # Continue if inoueMashuu/hiragana-dataset image is not in k49
            if label_map[label] == None:
                continue
            outputs = model(img.unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            # Continue if predicted k49 is not in inoueMashuu/hiragana-dataset
            if predicted.item() not in list(label_map.values()):
                continue 
            total += 1
            correct += 1 if list(label_map.values()).index(predicted.item()) ==  label else 0
    
        # Calculate accuracy
        accuracy = correct / total
        print(f'Accuracy of the model on arbitrary images: {accuracy * 100:.2f}%')
            
        ## Print out what's happening
        # print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")
        wandb.log({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "extern_dataset_accuracy": accuracy * 100
        })


# torch.save(model.state_dict(), "temp-3_large-3-100-0001.pth")

#sns.set_style("dark")
#sns.lineplot(data=losses).set(title="loss change during training", xlabel="epoch", ylabel="loss")
#plt.savefig("loss_graph.png")
#plt.close()

