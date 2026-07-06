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
    return


if __name__ == "__main__":
    app.run()
