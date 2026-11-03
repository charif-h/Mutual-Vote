import pytest

from mutualvote.opinion import Opinion


def test_initial_state_matches_table_2() -> None:
    o = Opinion()
    assert (o.belief, o.disbelief, o.uncertainty) == (0.0, 0.0, 1.0)


def test_positive_then_negative_matches_table_2() -> None:
    o = Opinion().positive(1.0)
    assert (o.r, o.s, o.w) == (1.0, 0.0, 2.0)
    assert o.belief == pytest.approx(1 / 3)
    assert o.uncertainty == pytest.approx(2 / 3)
    o = o.negative(1.0)
    assert (o.r, o.s, o.w) == (1.0, 1.0, 2.0)
    assert (o.belief, o.disbelief, o.uncertainty) == pytest.approx((1 / 4, 1 / 4, 2 / 4))


def test_masses_sum_to_one() -> None:
    o = Opinion().positive(0.3).negative(0.7).positive(0.9)
    assert o.belief + o.disbelief + o.uncertainty == pytest.approx(1.0)


def test_paper_example_ab_about_c() -> None:
    # score([A,B],[C]) from the worked example: r = 0.8 + 0.9, w = 2 + 0.2 + 0.1
    o = Opinion().positive(0.8).combine(Opinion().positive(0.9))
    assert (o.r, o.s, o.w) == pytest.approx((1.7, 0.0, 2.3))
    assert o.belief == pytest.approx(0.425)
    assert o.disbelief == pytest.approx(0.0)
    assert o.uncertainty == pytest.approx(0.575)
    assert o.expected() == pytest.approx(0.7125)
    assert o.score() == pytest.approx(0.425)


def test_paper_example_ab_about_d() -> None:
    # A has no opinion about D -> one hostile interaction; B->D is positive 0.9
    o = Opinion().positive(0.9).combine(Opinion().negative(0.5))
    assert (o.r, o.s, o.w) == pytest.approx((0.9, 0.5, 2.6))
    assert o.score() == pytest.approx(0.1)


def test_hostile_count_equals_repeated_negative() -> None:
    o = Opinion().hostile(0.3, 4)
    p = Opinion()
    for _ in range(4):
        p = p.negative(0.3)
    assert (o.r, o.s, o.w) == pytest.approx((p.r, p.s, p.w))


def test_combine_counts_prior_once() -> None:
    assert Opinion().combine(Opinion()) == Opinion()


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        Opinion().positive(1.5)
    with pytest.raises(ValueError):
        Opinion().negative(-0.1)
    with pytest.raises(ValueError):
        Opinion(a=0.3).combine(Opinion(a=0.5))
