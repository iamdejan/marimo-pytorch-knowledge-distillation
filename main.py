import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Knolwedge Distillation with PyTorch

    Reference: [Knowledge Distillation Tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load PyTorch
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets

    # Check if the current `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
    # is available, and if not, use the CPU
    device = (
        torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    )
    print(f"Using {device} device")
    return datasets, device, nn, optim, torch, transforms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare Preprocessing and Load Dataset
    """)
    return


@app.cell
def _():
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    return


@app.cell
def _(datasets, transforms):
    transforms_cifar = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Loading the CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transforms_cifar
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transforms_cifar
    )
    return test_dataset, train_dataset


@app.cell
def _(train_dataset):
    train_dataset
    return


@app.cell
def _(test_dataset):
    test_dataset
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DataLoader
    """)
    return


@app.cell
def _(test_dataset, torch, train_dataset):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    return test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model classes
    """)
    return


@app.cell
def _(nn, torch):
    class DeepNN(nn.Module):
        def __init__(self, num_classes=10):
            super(DeepNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    return (DeepNN,)


@app.cell
def _(nn, torch):
    class LightNN(nn.Module):
        def __init__(self, num_classes=10):
            super(LightNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    return (LightNN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Utility functions
    """)
    return


@app.cell
def _(nn, optim):
    def train(model, train_loader, epochs, learning_rate, device):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                # inputs: A collection of batch_size images
                # labels: A vector of dimensionality batch_size with integers denoting class of each image
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
                # labels: The actual labels of the images. Vectors of dimensionality batch_size.
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss = {running_loss / len(train_loader)}")

    return (train,)


@app.cell
def _(torch):
    def test(model, test_loader, device):
        model.to(device)
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    return (test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train and test the model
    """)
    return


@app.cell
def _(DeepNN, device, test, test_loader, torch, train, train_loader):
    torch.manual_seed(42)
    nn_deep = DeepNN(num_classes=10).to(device)
    train(nn_deep, train_loader, epochs=10, learning_rate=0.001, device=device)
    test_accuracy_deep = test(nn_deep, test_loader, device)
    return nn_deep, test_accuracy_deep


@app.cell
def _(LightNN, device, torch):
    torch.manual_seed(42)
    nn_light = LightNN(num_classes=10).to(device)
    return (nn_light,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We instantiate one more lightweight network model to compare their performances. Back propagation is sensitive to weight initialization, so we need to make sure these two networks have the exact same initialization.
    """)
    return


@app.cell
def _(LightNN, device, torch):
    torch.manual_seed(42)
    new_nn_light = LightNN(num_classes=10).to(device)
    return (new_nn_light,)


@app.cell
def _(new_nn_light, nn_light, torch):
    print("Norm of 1st layer of nn_light:", torch.norm(nn_light.features[0].weight).item())
    # Print the norm of the first layer of the new lightweight model
    print("Norm of 1st layer of new_nn_light:", torch.norm(new_nn_light.features[0].weight).item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Print the total number of parameters in each model:
    """)
    return


@app.cell
def _(nn_deep, nn_light):
    total_params_deep = "{:,}".format(sum(p.numel() for p in nn_deep.parameters()))
    print(f"DeepNN parameters: {total_params_deep}")
    total_params_light = "{:,}".format(sum(p.numel() for p in nn_light.parameters()))
    print(f"LightNN parameters: {total_params_light}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Train and test the lightweight network with cross entropy loss:
    """)
    return


@app.cell
def _(device, nn_light, test, test_loader, train, train_loader):
    train(nn_light, train_loader, epochs=10, learning_rate=0.001, device=device)
    test_accuracy_light_ce = test(nn_light, test_loader, device)
    return (test_accuracy_light_ce,)


@app.cell
def _(test_accuracy_deep, test_accuracy_light_ce):
    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    print(f"Student accuracy: {test_accuracy_light_ce:.2f}%")
    return


if __name__ == "__main__":
    app.run()
