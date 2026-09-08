from evaluate_model import parse_numeric


def test_takes_last_number_not_first():
    # The real bug this guards against: a first-match regex on this
    # exact kind of output grabbed the year 2022, not the answer.
    assert parse_numeric("2022: 33,364\n2023: 30,425") == 30425.0


def test_takes_last_number_after_shown_arithmetic():
    assert parse_numeric("8,625 + 66,871 = 75,496") == 75496.0


def test_strips_commas_and_dollar_sign():
    assert parse_numeric("$76,559,000,000") == 76559000000.0


def test_negative_number():
    assert parse_numeric("The change was -1234.5") == -1234.5


def test_decimal_eps_value():
    assert parse_numeric("diluted EPS is 3.27") == 3.27


def test_no_number_returns_none():
    assert parse_numeric("I cannot determine this from the table.") is None


def test_whitespace_only_returns_none():
    assert parse_numeric("   ") is None
