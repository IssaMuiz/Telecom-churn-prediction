"""Unit tests for main.py functions"""
import pytest
from main import divide, add


def test_add():
    """Test add function"""
    assert add(2, 3) == 5
    assert add(3, 4) == 7


def test_divide():
    """Test divide function"""
    with pytest.raises(ValueError, match="You can't divide by zero"):
        divide(10, 0)
