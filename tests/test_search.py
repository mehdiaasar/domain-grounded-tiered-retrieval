from dgtr.search import format_results, split_documents
from dgtr.sources import DOMAINS, TRUSTED_SOURCES, allowlist_for


class TestFormatResults:
    def test_dict_payload(self):
        payload = {"results": [{"url": "https://who.int/a", "content": "Insulin lowers glucose."}]}
        context, urls = format_results(payload)
        assert context == "Source: https://who.int/a\nInsulin lowers glucose."
        assert urls == ["https://who.int/a"]

    def test_bare_list_payload(self):
        context, urls = format_results([{"url": "u", "content": "c"}])
        assert urls == ["u"]
        assert "c" in context

    def test_entries_without_content_are_skipped(self):
        context, urls = format_results({"results": [{"url": "u"}, "junk"]})
        assert context == ""
        assert urls == []

    def test_unexpected_payload_yields_empty_context(self):
        # Never prose -- an apology here would be graded and cited as a document.
        assert format_results(None) == ("", [])
        assert format_results("some string") == ("", [])


class TestSplitDocuments:
    def test_roundtrip(self):
        payload = {
            "results": [
                {"url": "https://a.test", "content": "first"},
                {"url": "https://b.test", "content": "second"},
            ]
        }
        context, _ = format_results(payload)
        docs = split_documents(context)
        assert len(docs) == 2
        assert docs[0].startswith("Source: https://a.test")

    def test_empty(self):
        assert split_documents("") == []


class TestAllowlists:
    def test_unknown_domain_falls_back_to_general(self):
        assert allowlist_for("astrology") == TRUSTED_SOURCES["general"]
        assert allowlist_for(None) == TRUSTED_SOURCES["general"]

    def test_paper_declares_seven_domains(self):
        assert len(DOMAINS) == 7
        assert "general" in DOMAINS

    def test_every_domain_has_sources(self):
        assert all(TRUSTED_SOURCES[d] for d in DOMAINS)
