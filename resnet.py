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
    # Knowledge Distillation: ResNet50 ➞ ResNet18 on CIFAR-10
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Basic Setup
    """)
    return


@app.cell
def _():
    import os
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import random_split, DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torchvision import models
    import torchvision.transforms as transforms
    from torchvision.datasets import CIFAR10

    print(f"PyTorch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device.type}")
    return (
        CIFAR10,
        DataLoader,
        F,
        device,
        models,
        nn,
        random_split,
        time,
        torch,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Dataset
    """)
    return


@app.cell
def _(CIFAR10, transforms):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 means
                std=[0.2023, 0.1994, 0.2010],
            ),
        ],
    )

    # Load CIFAR-10 train set
    full_trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    full_trainset
    return full_trainset, transform


@app.cell
def _(full_trainset, random_split):
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size

    # Perform split
    train_subset, val_subset = random_split(full_trainset, [train_size, val_size])
    print(f"Train samples: {train_size}")
    print(f"Validation samples: {val_size}")
    return train_subset, val_subset


@app.cell
def _(DataLoader, train_subset, val_subset):
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
    return


@app.cell
def _(CIFAR10, DataLoader, transform):
    test_set = CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    print(f"Test samples: {len(test_set)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup Models
    """)
    return


@app.cell
def _(F, nn):
    class FeatureProjector(nn.Module):
        """
        Feature projector to match student -> teacher feature shapes
        """

        def __init__(self, in_channels, out_channels):
            super(FeatureProjector, self).__init__()

            # define a 1x1 convolutional layer to project feature maps
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x, target_shape):
            # check if the spatial dimensions of the input match the target shape
            if x.shape[2:] != target_shape[2:]:
                # adjust spatial dimensions using adaptive average mapping
                x = F.adaptive_avg_pool2d(x, output_size=target_shape[2:])

            return self.proj(x)

    return (FeatureProjector,)


@app.cell
def _(F, nn, torch):
    class StudentWrapper(nn.Module):
        """
        Wrapper for the student model with projection layers
        """

        def __init__(self, student_model, proj_layers):
            super(StudentWrapper, self).__init__()

            self.model = student_model

            # store projection layers for feature alignment
            self.projections = nn.ModuleList(proj_layers)

        def forward(self, x):
            # collect intermediate features from ResNet blocks
            features = []
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            for i, block in enumerate(
                [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
            ):
                x = block(x)

                # append features from each block
                features.append(x)

            # pool the final feature map and compute logits
            pooled = F.adaptive_avg_pool2d(x, (1, 1))
            flat = torch.flatten(pooled, 1)
            logits = self.model.fc(flat)

            return logits, features

        def project_features(self, features, target_shapes):
            """
            Project student features to match the shapes of teacher features.
            """
            return [
                proj(s_feat, t_shape)
                for s_feat, t_shape, proj in zip(features, target_shapes, self.projections)
            ]

    return (StudentWrapper,)


@app.cell
def _(FeatureProjector, StudentWrapper, models, nn):
    def setup_models(device):
        """
        Setup teacher and student wrapper
        """

        # teacher: ResNet50 pretrained on ImageNet, re-headed for CIFAR-10
        teacher = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        teacher.fc = nn.Linear(2048, 10)
        teacher = teacher.to(device)

        # student: ResNet18 without pretrained weights
        student = models.resnet18(weights=None)
        student.fc = nn.Linear(512, 10)
        student = student.to(device)

        # define the intermediate feature channels for both teacher and student
        teacher_channels = [256, 512, 1024, 2048]
        student_channels = [64, 128, 256, 512]

        # create projection layers to align teacher's feature maps with student's feature maps
        proj_layers = [
            FeatureProjector(in_c, out_c).to(device)
            for in_c, out_c in zip(student_channels, teacher_channels)
        ]

        # wrap the student model with the projection layers
        student_wrapper = StudentWrapper(
            student_model=student,
            proj_layers=proj_layers,
        )

        return teacher, student_wrapper

    return (setup_models,)


@app.cell
def _(F, self, torch):
    def extract_teacher_features(model, x, layers=[1, 2, 3, 4]):
        """
        Extract teacher logits and intermediate features
        """

        # collect intermediate features from ResNet blocks
        features = []
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        for i, block in enumerate(
            [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        ):
            x = block(x)
            if (i + 1) in layers:
                features.append(x)

        # pool the final feature map and compute logits
        pooled = F.adaptive_adaptive_avg_pool2d(x, (1, 1))  # [B, C, 1, 1]
        flat = torch.flatten(pooled, 1)  # [B, C]
        logits = model.fc(flat)  # [B, 10]
        return logits, features

    return


@app.cell
def _(device, setup_models):
    teacher, student_wrapper = setup_models(device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation Functions for Size, Latency and Accuracy
    """)
    return


@app.function
def count_params(model):
    """
    Function to count trainable parameters
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@app.cell
def _(time, times, torch):
    def measure_latency(model, input_size=(1, 3, 32, 32), device="cuda", repetitions=50):
        """
        Function to measure average inference latency over multiple runs
        """

        model.eval()
        inputs = torch.randn(input_size).to(device)
        with torch.no_grad():
            # Warm-up
            for _ in range(10):
                _ = model(inputs)

            # Measure
            durations = []
            for _ in range(repetitions):
                start = time.time()
                _ = model(inputs)
                end = time.time()
                durations.append(end - start)
        return (sum(times) / repetitions) * 1000

    return


@app.cell
def _(device, torch):
    def evaluate_accuracy(model, dataloader):
        """
        Evaluate accuracy given model and loader
        """

        model.eval()
        model.to(device)
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        return accuracy

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fine-tuning the Teacher

    > Note: This fine-tuning is required only in this example since the ResNet50 network was pretrained on ImageNet1K and we need to replace its output layer to match to CIFAR-10.
    """)
    return


if __name__ == "__main__":
    app.run()
