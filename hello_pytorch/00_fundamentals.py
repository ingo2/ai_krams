import torch


def main() -> None:
    print(torch.__version__)
    print(f"cuda: {torch.cuda.is_available()}")

    print("\n****Scalar****")
    scalar = torch.tensor(1.0)
    print(f"{scalar}")
    print(f"shape: {scalar.shape}")
    print(f"ndim: {scalar.ndim}")
    print(f"item: {scalar.item()}")

    print("\n****Vector****")
    vector = torch.tensor([1.0, 2.0, 3.0])
    print(f"{vector}")
    print(f"shape: {vector.shape}")
    print(f"ndim: {vector.ndim}")

    print("\n****Matrix****")
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"{matrix}")
    print(f"shape: {matrix.shape}")
    print(f"ndim: {matrix.ndim}")

    print("\n****Tensor****")
    # fmt: off
    tensor = torch.tensor([[[1, 2, 3],
                            [3, 6, 9],
                            [2, 4, 5]]])
    # fmt: on
    print(f"{tensor}")
    print(f"shape: {tensor.shape}")
    print(f"ndim: {tensor.ndim}")

    print("\n****Another Tensor****")
    # fmt: off
    tensor = torch.tensor([[[1.0, 2.0], 
                            [3.0, 4.0]], 
                          [[5.0, 6.0], 
                           [7.0, 8.0]]])
    # fmt: on
    print(f"{tensor}")
    print(f"shape: {tensor.shape}")
    print(f"ndim: {tensor.ndim}")

    print("\n****Random Tensor****")
    random_tensor = torch.rand(size=(3, 4))
    print(random_tensor)
    print(random_tensor.dtype)
    print(random_tensor.device)

    print("\n****Linear Layering****")
    # fmt: off
    tensor_A = torch.tensor(
        [[1, 2],
         [3, 4],
         [5, 6]], dtype=torch.float32)

    tensor_B = torch.tensor(
        [[7, 10],
         [8, 11], 
         [9, 12]], dtype=torch.float32)
    # fmt: on

    torch.manual_seed(42)
    # This uses matrix multiplication
    linear = torch.nn.Linear(
        in_features=2,  # in_features = matches inner dimension of input
        out_features=6,  # out_features = describes outer value
    )
    x = tensor_A
    output = linear(x)
    print(f"Linear layer weights shape: {linear.weight.shape}")
    print(f"Input shape: {x.shape}\n")
    print(f"Output:\n{output}\n\nOutput shape: {output.shape}")

    print(f"Linear layer weights:\n{linear.weight.data}\n")
    print(f"Linear layer bias:\n{linear.bias.data}\n")

    print("\n****Tensor to GPU****")
    print(f"GPU available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor_gpu = tensor_A.to(device)
    print(f"Tensor on GPU:\n{tensor_gpu}\n")


if __name__ == "__main__":
    main()
