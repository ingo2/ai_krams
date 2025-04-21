def quantised_values_rounded(min: float, max: float, steps: int) -> list:
    """Generate a list of evenly spaced values between min and max, rounded."""
    return [round(value) for value in quantised_values_exact(min, max, steps)]


def quantised_values_exact(min: float, max: float, steps: int) -> list:
    """Generate a list of evenly spaced values between min and max, not rounded."""
    return [i * (max - min) / (steps - 1) + min for i in range(steps)]
