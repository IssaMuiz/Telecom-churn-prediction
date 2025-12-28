from src.my_module import add


def test_add():
    result = add(2, 8)
    assert result == 10
