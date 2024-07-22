"""Test module for the ExampleClass in the example_package."""

from src.example_package.example import ExampleClass


def test_example_class() -> None:
    """Test the functionality of the ExampleClass."""
    obj = ExampleClass(5)
    target = 5
    assert obj.get_value() == target
    obj.increment()
    assert obj.get_value() == target + 1
    assert obj.get_value_str() == "6"
    constant = 10
    assert obj.get_value_wrong_type() == constant
