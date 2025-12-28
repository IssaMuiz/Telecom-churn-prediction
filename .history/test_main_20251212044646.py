from main import get_weather


def test_get_weather():
    assert get_weather(25) == 'It is hot'
