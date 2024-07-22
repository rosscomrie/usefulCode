"""An example module demonstrating a simple class with basic operations."""


class ExampleClass:
    """An example class that stores and manipulates an integer value."""

    def __init__(self, value: int) -> None:
        """
        Initialize the ExampleClass with a given value.

        Args:
        ----
            value (int): The initial value to be stored.

        """
        self.value = value

    def increment(self) -> None:
        """Increment the stored value by 1."""
        self.value += 1

    def get_value(self) -> int:
        """
        Retrieve the current stored value.

        Returns
        -------
            int: The current value stored in the class.

        """
        return self.value

    def get_value_str(self) -> str:
        """Nothing."""
        return str(self.value)

    def get_value_wrong_type(self) -> int:
        """Nothing."""
        test = "10"
        return int(test)
