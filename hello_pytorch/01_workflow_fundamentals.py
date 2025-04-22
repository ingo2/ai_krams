# https://www.learnpytorch.io/01_pytorch_workflow/

import torch, torch.nn as nn
import matplotlib.pyplot as plt


# Create a Linear Regression model class
class LinearRegressionModel(
    nn.Module  # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(
                1,  # <- start with random weights (this will get adjusted as the model learns)
                dtype=torch.float,  # <- PyTorch loves float32 by default
            ),
            requires_grad=True,  # <- can we update this value with gradient descent?)
        )

        self.bias = nn.Parameter(
            torch.randn(
                1,  # <- start with random bias (this will get adjusted as the model learns)
                dtype=torch.float,  # <- PyTorch loves float32 by default
            ), 
            requires_grad=True,  # <- can we update this value with gradient descent?))
        )

    # Forward defines the computation in the model
    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # <- "x" is the input data (e.g. training/testing features)
        return (
            self.weights * x + self.bias
        )  # <- this is the linear regression formula (y = m*x + b)


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})

    plt.show()


def main() -> None:
    print(f"version: {torch.__version__}")
    print(f"cuda: {torch.cuda.is_available()}")

    # Set manual seed since nn.Parameter are randomly initialized
    torch.manual_seed(42)

    # Create *known* parameters
    weight = 0.7
    bias = 0.3

    # Create data
    start = 0
    end = 1
    step = 0.02
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias

    print(X[:10]), print(y[:10])

    # Create train/test split
    train_split = int(0.8 * len(X))  # 80% of data used for training set, 20% for testing
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    print("\n****Train/Test Split****")
    print(f"{len(X_train)}, {len(y_train)}, {len(X_test)}, {len(y_test)}")

    # Plot the data
    # plot_predictions(X_train, y_train, X_test, y_test)

    # Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
    model_0 = LinearRegressionModel()

    print("\n****Model 0****")
    # Check the nn.Parameter(s) within the nn.Module subclass we created
    print(list(model_0.parameters()))

    # List named parameters
    print(model_0.state_dict())

    print("\n****Model 0 Predictions****")
    # Make predictions with model
    with torch.inference_mode():
        y_preds = model_0(X_test)

    # Check the predictions
    print(f"Number of testing samples: {len(X_test)}")
    print(f"Number of predictions made: {len(y_preds)}")
    print(f"Predicted values:\n{y_preds}")
    print(f"Error:\n{y_test - y_preds}")

    # Plot the data
    # plot_predictions(X_train, y_train, X_test, y_test, y_preds)

    print("\n****Model 0 Optimization****")

    # Create the loss function
    loss_fn = nn.L1Loss()  # MAE loss is same as L1Loss

    # Create the optimizer
    optimizer = torch.optim.SGD(
        # parameters of target model to optimize
        params=model_0.parameters(),
        # learning rate (how much the optimizer should change parameters at each step,
        # higher=more (less stable), lower=less (might take a long time)
        lr=0.01,
    )

    # Set the number of epochs (how many times the model will pass over the training data)
    epochs = 201

    # Create empty loss lists to track values
    train_loss_values = []
    test_loss_values = []
    epoch_count = []

    for epoch in range(epochs):
        ### Training

        # Put model in training mode (this is the default state of a model)
        model_0.train()

        # 1. Forward pass on train data using the forward() method inside
        y_pred = model_0(X_train)
        # print(y_pred)

        # 2. Calculate the loss (how different are our models predictions to the ground truth)
        loss = loss_fn(y_pred, y_train)

        # 3. Zero grad of the optimizer
        optimizer.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Progress the optimizer
        optimizer.step()

        ### Testing

        # Put the model in evaluation mode
        model_0.eval()

        with torch.inference_mode():
            # 1. Forward pass on test data
            test_pred = model_0(X_test)

            # 2. Caculate loss on test data
            test_loss = loss_fn(
                # predictions come in torch.float datatype, so comparisons need to be done with 
                # tensors of the same type
                test_pred,
                y_test.type(torch.float),
            )

            # Print out what's happening
            if epoch % 20 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
    
    # Plot the loss curves
    # plt.plot(epoch_count, train_loss_values, label="Train loss")
    # plt.plot(epoch_count, test_loss_values, label="Test loss")
    # plt.title("Training and test loss curves")
    # plt.ylabel("Loss")
    # plt.xlabel("Epochs")
    # plt.legend()
    # plt.show()

    # Find our model's learned parameters
    print("The model learned the following values for weights and bias:")
    print(model_0.state_dict())
    print("\nAnd the original values for weights and bias are:")
    print(f"weights: {weight}, bias: {bias}")

    # 1. Set the model in evaluation mode
    model_0.eval()

    # 2. Setup the inference mode context manager
    with torch.inference_mode():
      # 3. Make sure the calculations are done with the model and data on the same device
      # in our case, we haven't setup device-agnostic code yet so our data and model are
      # on the CPU by default.
      # model_0.to(device)
      # X_test = X_test.to(device)
      y_preds = model_0(X_test)
    
    # Plot the data
    plot_predictions(X_train, y_train, X_test, y_test, y_preds)

if __name__ == "__main__":
    main()
