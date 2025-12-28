from main import get_weather


def test_get_weather():
    assert get_weather(20) == 'It is hot'
