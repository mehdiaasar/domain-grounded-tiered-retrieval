from dgtr.claims import parse_claims, strip_numbering


class TestParseClaims:
    def test_keeps_numbered_claims(self):
        assert parse_claims("1. A is B.\n2. C is D.") == ["1. A is B.", "2. C is D."]

    def test_drops_model_preamble_and_trailing_commentary(self):
        raw = (
            "Here is the decomposed output:\n"
            "1. The Port Authority of Thailand exists.\n"
            "2. Attaphol Buspakom played for it.\n"
            "I followed the critical rules for extraction:\n"
            "* Decontextualization: I replaced the pronoun."
        )
        assert parse_claims(raw) == [
            "1. The Port Authority of Thailand exists.",
            "2. Attaphol Buspakom played for it.",
        ]

    def test_unnumbered_output_is_kept_rather_than_dropped(self):
        assert parse_claims("A is B.\nC is D.") == ["A is B.", "C is D."]

    def test_accepts_paren_numbering_and_a_list(self):
        assert parse_claims("1) A is B.") == ["1) A is B."]
        assert parse_claims(["intro:", "1. A is B."]) == ["1. A is B."]

    def test_blank_input(self):
        assert parse_claims("") == []
        assert parse_claims("   \n  ") == []


def test_strip_numbering():
    assert strip_numbering("1. A is B.") == "A is B."
    assert strip_numbering("A is B.") == "A is B."
