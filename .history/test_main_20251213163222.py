"""Unit tests for main.py functions"""
import pytest
from main import divide, add


@pytest.mark.parametrized("num1, num2", [
    (10, 2),
    (20, 4),
    (30, 5),
])
def test_add():
    """Test add function"""
    assert add(num1, num2) == num1 + num2


def test_divide():
    """Test divide function"""
    with pytest.raises(ValueError, match="You can divide by zero"):
        divide(10, 0)
