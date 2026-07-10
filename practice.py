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

    return np, os, set_seed


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
    ## Augment Data

    Data augmentation should include:
    - Darken by 5%, 10%, 15%, 20% (under-exposure)
    - Brighen by 5%, 10%, 15%, 20% (over-exposure)
    - Horizontal Flip
    - Motion blur
    - Gaussian blur
    - Salt-and-pepper Noise
    """)
    return


@app.cell
def _(base_path, np, os):
    import cv2
    from pathlib import Path

    # Safely construct folder paths using os.path.join
    dataset_splits = {
        "train": os.path.join(base_path, "dataset", "train"),
        "validation": os.path.join(base_path, "dataset", "valid"),
        "test": os.path.join(base_path, "dataset", "test"),
    }

    def adjust_exposure(image, factor):
        """
        Darkens or brightens the image by a given scale factor.
        Using float32 prevents pixel value overflow/underflow wrapping.
        """

        # Steps:
        # 1. Convert to float
        # 2. Scale pixel values
        # 3. Clip between 0-255
        rescaled = np.clip(image.astype(np.float32) * factor, 0, 255)
        return rescaled.astype(np.uint8)

    def add_salt_and_pepper_noise(image, percentage):
        """Adds a specific percentage of random black and white noise pixels."""
        noisy_image = image.copy()
        h, w, _ = image.shape
        random_matrix = np.random.rand(h, w)

        # salt (white)
        salt_mask = random_matrix < (percentage / 2)
        noisy_image[salt_mask] = [255, 255, 255]

        # pepper (black)
        pepper_mask = (random_matrix >= (percentage / 2)) & (random_matrix < percentage)
        noisy_image[pepper_mask] = [0, 0, 0]

        return noisy_image

    def apply_zoom_motion_blur(image, num_frames=10, zoom_range=0.05):
        """
        Applies a zoom motion blur to an image.
        Simulates a zoom motion blur by scaling the image around its center point multiple times and averaging the frames together.

        Args:
            image (numpy.ndarray): The input image (H, W, C).
            num_frames (int): Number of zoomed layers to stack.
            zoom_range (float): Magnitude of the zoom effect.
        """
        height, width = image.shape[:2]
        center_x = width / 2
        center_y = height / 2

        # Create an empty float32 array to safely accumulate image matrices without clipping
        blur_accumulator = np.zeros_like(image, dtype=np.float32)

        # Generate 10 evenly spaced zoom scale factors from 1.0 (original)
        scales = np.linspace(1.0, 1.0 + zoom_range, num_frames)

        for scale in scales:
            # Get the transformation matrix for scaling around the center (0 degree rotation)
            M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
            # Apply the transformation to the image
            warped = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)
            # Add the frame into our accumulator
            blur_accumulator += warped

        # Divide by total frames to get the mean effect, then convert back to 8-bit image
        zoom_blurred_image = blur_accumulator / num_frames
        return zoom_blurred_image.astype(np.uint8)

    def apply_gaussian_blur(image, kernel_size=7):
        """
        Applies standard Gaussian Blur. Kernel size must be positive and odd.
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)

    def apply_horizontal_flip(image):
        """
        Flips the image horizontally (axis 1).
        """
        return cv2.flip(image, flipCode=1)

    return (
        add_salt_and_pepper_noise,
        adjust_exposure,
        apply_gaussian_blur,
        apply_horizontal_flip,
        apply_zoom_motion_blur,
        cv2,
        dataset_splits,
    )


@app.cell
def _():
    exposure_tasks = {
        "_under_exposure_5p": 0.95,
        "_under_exposure_10p": 0.90,
        "_under_exposure_15p": 0.85,
        "_under_exposure_20p": 0.80,
        "_over_exposure_5p": 1.05,
        "_over_exposure_10p": 1.10,
        "_over_exposure_15p": 1.15,
        "_over_exposure_20p": 1.20,
    }

    noise_tasks = {
        "_noise_5p": 0.05,
        "_noise_10p": 0.10,
        "_noise_15p": 0.15,
        "_noise_20p": 0.20,
    }

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    return exposure_tasks, noise_tasks, valid_extensions


@app.cell
def _(dataset_splits, exposure_tasks, mo, noise_tasks, os, valid_extensions):
    @mo.persistent_cache
    def get_original_images(train_only: bool = True):
        original_images = []
        for split_name, split_path in dataset_splits.items():
            if train_only and split_name != "train":
                continue

            image_count_per_split = 0
            print(f"Gathering original files for {split_name} set")

            if not os.path.exists(split_path):
                print(f"[warning] Path {split_path} does not exist. Skipping...")
                continue

            all_suffixes = (
                list(exposure_tasks.keys())
                + list(noise_tasks.keys())
                + ["_horizontal_flip", "_motion_blur", "_gaussian_blur"]
            )
            for root, dirs, files in os.walk(split_path):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        # Ensure the file doesn't already contain one of our augmentation suffixes
                        if not any(suffix in file for suffix in all_suffixes):
                            original_images.append(os.path.join(root, file))
                            image_count_per_split += 1

            print(f"Found {image_count_per_split} original images in {split_name} set.")
        return original_images

    return (get_original_images,)


@app.cell
def _(
    add_salt_and_pepper_noise,
    adjust_exposure,
    apply_gaussian_blur,
    apply_horizontal_flip,
    apply_zoom_motion_blur,
    cv2,
    exposure_tasks,
    mo,
    noise_tasks,
    os,
):
    @mo.persistent_cache
    def augment_single_image(image_path):
        image = cv2.imread(image_path)
        if image is None:
            return False

        folder_path = os.path.dirname(image_path)
        filename, ext = os.path.splitext(os.path.basename(image_path))

        # 1. Apply Exposure Variations
        for suffix, factor in exposure_tasks.items():
            exposure_image = adjust_exposure(image, factor)
            cv2.imwrite(
                filename=os.path.join(folder_path, f"{filename}{suffix}{ext}"),
                img=exposure_image,
            )

        # 2. Apply Noise Variations
        for suffix, percentage in noise_tasks.items():
            noise_image = add_salt_and_pepper_noise(image, percentage)
            cv2.imwrite(
                filename=os.path.join(folder_path, f"{filename}{suffix}{ext}"),
                img=noise_image,
            )

        # 3. Apply Horizontal Flip
        flip_image = apply_horizontal_flip(image)
        cv2.imwrite(
            filename=os.path.join(folder_path, f"{filename}_horizontal_flip{ext}"),
            img=flip_image,
        )

        # 4. Apply Fixed Zoom Motion Blur
        z_blur_image = apply_zoom_motion_blur(image)
        cv2.imwrite(
            filename=os.path.join(folder_path, f"{filename}_motion_blur{ext}"),
            img=z_blur_image,
        )

        # 5. Apply Gaussian Blur
        g_blur_image = apply_gaussian_blur(image)
        cv2.imwrite(
            filename=os.path.join(folder_path, f"{filename}_gaussian_blur{ext}"),
            img=g_blur_image,
        )

        return True

    return (augment_single_image,)


@app.cell
def _(augment_single_image, get_original_images, mo):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Try to import tqdm for a nice live progress bar; fallback to a basic counter if not installed
    try:
        from tqdm import tqdm

        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False

    @mo.persistent_cache
    def augment_dataset():
        """
        Augment the train set from the dataset.
        """

        original_images = get_original_images(train_only=True)
        total_images = len(original_images)
        print(f"Found {total_images} original images. Starting parallel processing...")

        with ThreadPoolExecutor(max_workers=None) as executor:
            # Submit all image tasks to the pool
            futures = {
                executor.submit(augment_single_image, path): path for path in original_images
            }

            # Monitor execution progress
            if HAS_TQDM:
                # Displays a beautiful live updating progress bar
                for _ in tqdm(as_completed(futures), total=total_images, desc="Augmenting Images"):
                    pass
            else:
                completed_count = 0
                for future in as_completed(futures):
                    completed_count += 1
                    if completed_count % 500 == 0 or completed_count == total_images:
                        print(
                            f"Progress: {completed_count}/{total_images} original images processed."
                        )

        print("All images successfully augmented in parallel!")

    augment_dataset()
    return


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

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    return train_loader, validation_loader


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
def _(epoch_loss, mo, nn, torch):
    BEST_TEACHER_MODEL_PATH = "./models/practice/best_teacher_model.pt"

    @mo.persistent_cache
    def train_teacher(
        model,
        train_loader,
        validation_loader,
        learning_rate=0.001,
        epochs=30,
        feature_map_weight=0.2,
        cross_entropy_weight=0.8,
        device=torch.device("cpu"),
    ):
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
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
                    subtitle=f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {validation_accuracy:.4f}",
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
    model = setup_teacher()
    model = train_teacher(model, train_loader, validation_loader, device=device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distil the Knowledge to Student
    """)
    return


@app.cell
def _(models, nn, torch):
    def setup_student_model(num_classes=2, device=torch.device("cpu")):
        """
        Student model is ResNet-18 with last replacement on the last layer.
        Unlike the teacher, the student is using empty weights, so no weight freeze is applied here.
        """

        # Load pre-trained ResNet (e.g., ResNet50)
        model = models.resnet18(weights=None)

        # Step 2: Replace the last fully connected layer (model.fc)
        # The new layer automatically defaults to requires_grad = True
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

        return model.to(device)

    return


if __name__ == "__main__":
    app.run()
