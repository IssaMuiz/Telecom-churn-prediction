"""Unit tests for main.py functions"""
import pytest
from main import divide, add


@pytest.mark.parametrized("num1, num2, expected", [
    (10, 2, 12),
    (20, 4, 24),
    (30, 5, 35),
])
def test_add(num1, num2, expected):
    """Test add function"""
    assert add(num1, num2) == expected
