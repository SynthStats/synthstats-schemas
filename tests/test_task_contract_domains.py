"""Domain acceptance for values a float cannot hold.

A Python int is always finite, so a domain must not decide an integer's
finiteness by converting it to a float: the conversion overflows (and the
domain then refuses a valid value) once the integer is past float's range.
"""

import synthstats_schemas as schemas


def test_integer_domain_accepts_an_integer_beyond_float_range():
    domain = schemas.IntegerDomain(kind="integer")

    assert domain.accepts(2**2000) is True


def test_integer_domain_still_applies_its_bounds_beyond_float_range():
    inside = schemas.IntegerDomain(kind="integer", minimum=0, maximum=2**4000)
    outside = schemas.IntegerDomain(kind="integer", maximum=10)

    assert inside.accepts(2**2000) is True
    assert outside.accepts(2**2000) is False


def test_integer_domain_still_refuses_booleans_and_non_integers():
    domain = schemas.IntegerDomain(kind="integer")

    assert domain.accepts(True) is False
    assert domain.accepts(1.5) is False
    assert domain.accepts("3") is False


def test_a_numeric_level_beyond_float_range_is_matched():
    domain = schemas.NumericLevelsDomain(kind="numeric_levels", levels=[2**2000])

    assert domain.accepts(2**2000) is True


def test_real_domain_accepts_an_integer_beyond_float_range():
    domain = schemas.RealDomain(kind="real")

    assert domain.accepts(2**2000) is True
