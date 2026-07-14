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
    # Practice: Car Collision Detection

    Dataset: [Organized Car Collision Prediction Dataset](https://www.kaggle.com/datasets/isratjahankhan/organized-car-collision-prediction-dataset/data)
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

    # Check if the current `accelerator <https://pytorch.org/docs/stable/torch.h
    # is available, and if not, use the CPU
    device = (
        torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    )
    print(f"Using {device} device")
    return datasets, device, nn, torch, transforms


@app.cell
def _(torch):
    import numpy as np
    import random
    import os

    def set_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        # Configure CUDA backends
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Force use of deterministic algorithms (PyTorch 1.7+)
        torch.use_deterministic_algorithms(True)

    return os, set_seed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download Dataset
    """)
    return


@app.cell
def _():
    import kagglehub

    # Download latest version
    base_path = kagglehub.dataset_download(
        "isratjahankhan/organized-car-collision-prediction-dataset"
    )

    print("Base path to dataset files:", base_path)
    return (base_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Dataset
    """)
    return


@app.cell
def _(base_path, datasets, transforms):
    train_transform = transforms.Compose(
        [
            # Passing a single integer resizes the shorter side (720) to 224.
            # The longer side automatically scales to ~398, maintaining the 16:9 ratio.
            transforms.Resize(224),
            # Converts PIL Image to PyTorch Tensor (scales pixels to [0.0, 1.0])
            transforms.ToTensor(),
            # Standard ImageNet normalization required for pretrained ResNet weights
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_set = datasets.ImageFolder(f"{base_path}/dataset/train", transform=train_transform)
    train_set
    return (train_set,)


@app.cell
def _(torch, train_set):
    classes, counts = torch.unique(torch.tensor(train_set.targets), return_counts=True)
    all_counts = sum(counts).item()
    print(len(classes))
    weights = [all_counts / (len(classes) * count) for count in counts]
    weights
    return (weights,)


@app.cell
def _(base_path, datasets, transforms):
    validation_testing_transform = transforms.Compose(
        [
            # Passing a single integer resizes the shorter side (720) to 224.
            # The longer side automatically scales to ~398, maintaining the 16:9 ratio.
            transforms.Resize(224),
            # Converts PIL Image to PyTorch Tensor (scales pixels to [0.0, 1.0])
            transforms.ToTensor(),
            # Standard ImageNet normalization required for pretrained ResNet weights
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    validation_set = datasets.ImageFolder(
        f"{base_path}/dataset/valid", transform=validation_testing_transform
    )
    print(validation_set)

    test_set = datasets.ImageFolder(
        f"{base_path}/dataset/test", transform=validation_testing_transform
    )
    print(test_set)
    return test_set, validation_set


@app.cell
def _(test_set, train_set, validation_set):
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)
    return test_loader, train_loader, validation_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fine-Tune Teacher
    """)
    return


@app.cell
def _():
    import subprocess

    subprocess.run(["mkdir", "-p", "models/practice"], capture_output=True, text=True)
    return


@app.cell
def _(nn, torch):
    import torchvision.models as models

    def setup_teacher(num_classes=2, device=torch.device("cpu")):
        """
        Teacher model is ResNet-152 with last replacement on the last layer.
        """

        # Load pre-trained ResNet (e.g., ResNet50)
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

        # Step 1: Freeze all parameters in the network
        for param in model.parameters():
            param.requires_grad = False

        # Step 2: Replace the last fully connected layer (model.fc)
        # The new layer automatically defaults to requires_grad = True
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

        return model.to(device)

    return models, setup_teacher


@app.cell
def _(mo, nn, os, torch, weights):
    BEST_TEACHER_MODEL_PATH = "./models/practice/best_teacher_model.pt"

    def train_teacher(
        model,
        train_loader,
        validation_loader,
        learning_rate=0.001,
        epochs=30,
        device=torch.device("cpu"),
    ):
        if os.path.exists(BEST_TEACHER_MODEL_PATH):
            model.load_state_dict(torch.load(BEST_TEACHER_MODEL_PATH), strict=True)
            return model

        model = model.to(device)
        criterion = nn.CrossEntropyLoss(torch.tensor(weights, device=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_validation_accuracy = 0.0
        with mo.status.progress_bar(total=epochs, title="Training Teacher Model") as bar:
            for epoch in range(epochs):
                running_loss = 0.0
                model.train()
                for inputs, labels in train_loader:
                    # inputs: A collection of batch_size images
                    # labels: A vector of dimensionality batch_size with integers denoting class of each image
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)

                    # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
                    # labels: The actual labels of the images. Vector of dimensionality batch_size
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for inputs, labels in validation_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        preds = outputs.argmax(dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                validation_accuracy = correct / total

                # 2. Dynamically update the bar progress and status subtitle text
                bar.update(
                    increment=1,
                    subtitle=f"Epoch {epoch + 1}/{epochs} | Loss: {running_loss:.4f} | Val Acc: {validation_accuracy:.4f}",
                )
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss = {running_loss / len(train_loader)}, Accuracy on validation set = {validation_accuracy}"
                )

                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    torch.save(model.state_dict(), BEST_TEACHER_MODEL_PATH)

        # load best model
        model.load_state_dict(torch.load(BEST_TEACHER_MODEL_PATH), strict=True)
        return model

    return (train_teacher,)


@app.cell
def _(
    device,
    set_seed,
    setup_teacher,
    train_loader,
    train_teacher,
    validation_loader,
):
    set_seed(42)
    teacher = setup_teacher()
    teacher = train_teacher(teacher, train_loader, validation_loader, device=device)
    return (teacher,)


@app.cell
def _(device, teacher, test_loader, torch):
    from sklearn.metrics import classification_report

    def evaluate_teacher(
        model,
        test_loader,
        device=torch.device("cpu"),
    ):
        all_predictions = []
        all_trues = []

        model.to(device)
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # 1. Forward pass to get model outputs
                outputs = model(inputs)

                # 2. Get predicted class indices (for classification)
                _, predictions = torch.max(outputs, dim=1)

                # 3. Store batch results (move to CPU and convert to numpy)
                all_predictions.extend(predictions.detach().cpu().numpy())
                all_trues.extend(labels.detach().cpu().numpy())

        # Generate report on the full dataset
        print(classification_report(all_trues, all_predictions))

    evaluate_teacher(teacher, test_loader, device=device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distil the Knowledge to Student
    """)
    return


@app.cell
def _(models, nn, torch):
    def setup_student(num_classes=2, device=torch.device("cpu")):
        """
        Student model is ResNet-18 with last replacement on the last layer.
        Unlike the teacher, the student is using empty weights, so no weight freeze is applied here.
        """

        # Load pre-trained ResNet (e.g., ResNet50)
        model = models.resnet18(weights=None)

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

        return model.to(device)

    return (setup_student,)


@app.cell
def _(mo, nn, os, torch, weights):
    BEST_STUDENT_MODEL_PATH = "./models/practice/best_student_model.pt"

    def train_student(
        teacher,
        student,
        train_loader,
        validation_loader,
        learning_rate=0.001,
        epochs=30,
        T=2,
        soft_targets_loss_weight=0.2,
        cross_entropy_loss_weight=0.8,
        device="cpu",
    ):
        device = torch.device(device)

        if os.path.exists(BEST_STUDENT_MODEL_PATH):
            student.load_state_dict(torch.load(BEST_STUDENT_MODEL_PATH), strict=True)
            return student

        teacher.to(device)
        student.to(device)

        teacher.eval()

        criterion = nn.CrossEntropyLoss(torch.tensor(weights, device=device))
        optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
        best_validation_accuracy = 0.0
        with mo.status.progress_bar(total=epochs, title="Training Student Model") as bar:
            for epoch in range(epochs):
                running_loss = 0.0
                student.train()
                for inputs, labels in train_loader:
                    # inputs: A collection of batch_size images
                    # labels: A vector of dimensionality batch_size with integers denoting class of each image
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()

                    with torch.no_grad():
                        teacher_outputs = teacher(inputs)

                    student_outputs = student(inputs)

                    soft_targets = nn.functional.softmax(teacher_outputs / T, dim=-1)
                    soft_prob = nn.functional.log_softmax(student_outputs / T, dim=-1)

                    soft_targets_loss = (
                        torch.sum(soft_targets * (soft_targets.log() - soft_prob))
                        / soft_prob.size()[0]
                        * (T**2)
                    )

                    label_loss = criterion(student_outputs, labels)

                    # Weighted sum of the two losses
                    loss = (
                        soft_targets_loss_weight * soft_targets_loss
                        + cross_entropy_loss_weight * label_loss
                    )

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                student.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for inputs, labels in validation_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = student(inputs)
                        preds = outputs.argmax(dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                validation_accuracy = correct / total

                # 2. Dynamically update the bar progress and status subtitle text
                bar.update(
                    increment=1,
                    subtitle=f"Epoch {epoch + 1}/{epochs} | Loss: {running_loss:.4f} | Val Acc: {validation_accuracy:.4f}",
                )
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss = {running_loss / len(train_loader)}, Accuracy on validation set = {validation_accuracy}"
                )

                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    torch.save(student.state_dict(), BEST_STUDENT_MODEL_PATH)

        # load best model
        student.load_state_dict(torch.load(BEST_STUDENT_MODEL_PATH), strict=True)
        return student

    return (train_student,)


@app.cell
def _(
    device,
    set_seed,
    setup_student,
    teacher,
    train_loader,
    train_student,
    validation_loader,
):
    set_seed(42)
    student = setup_student()
    student = train_student(teacher, student, train_loader, validation_loader, device=device)
    return


if __name__ == "__main__":
    app.run()
