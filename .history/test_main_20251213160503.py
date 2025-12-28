"""Unit tests for main.py functions"""
from main import divide, add
import pytest
"""Import the functions to be tested from main.py"""


def test_add():
    """Test add function"""
    assert add(2, 3) == 5
    assert add(3, 4) == 6


def test_divide():
    """Test divide function"""
    with pytest.raises(ValueError, match="You can't divide by zero"):
        assert divide(10, 0) == 3
