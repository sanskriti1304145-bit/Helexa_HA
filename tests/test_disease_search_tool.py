"""
test_disease_search_tool.py - OFFLINE tests for tools/disease_search_tool.py

Run from the project root:
    python -m tests.test_disease_search_tool

These tests never touch the internet. Instead of really calling MedlinePlus,
we swap requests.get for a fake that returns a SAVED real reply from
tests/fixtures/ (or pretends there is a timeout / no internet).
That is called "mocking". The live check is a separate script:
    python -m tests.live_check_medlineplus
"""

import os
from pathlib import Path

import requests

from tests.test_helpers import (
    use_fresh_test_data, check, fingerprint, passed_count, ORIGINAL_DATA_DIR, TEST_DATA_DIR,
)

use_fresh_test_data()

import config  # noqa: E402
from tools import disease_search_tool  # noqa: E402
from tools.disease_search_tool import (  # noqa: E402
    search_medlineplus, parse_medlineplus_xml, clean_html,
)
from tools.logger import read_logs  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
CKD_XML = (FIXTURES / "medlineplus_ckd_sample.xml").read_text(encoding="utf-8")
NO_RESULTS_XML = (FIXTURES / "medlineplus_no_results_sample.xml").read_text(encoding="utf-8")
SPELLING_XML = (FIXTURES / "medlineplus_spelling_sample.xml").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fakes that replace requests.get
# ---------------------------------------------------------------------------

class FakeResponse:
    """Looks enough like a requests Response for our tool."""
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Server Error")


last_call = {}   # remembers what the tool sent, so we can check the request


def fake_get_returning(text, status_code=200):
    def fake_get(url, params=None, timeout=None):
        last_call.update(url=url, params=params, timeout=timeout)
        return FakeResponse(text, status_code)
    return fake_get


def fake_get_raising(error):
    def fake_get(url, params=None, timeout=None):
        raise error
    return fake_get


def use_fake(fake):
    disease_search_tool.requests.get = fake


# ---------------------------------------------------------------------------

def test_clean_html():
    print("clean_html")
    check("removes highlight spans",
          clean_html('<span class="qt0">Chronic</span> <span class="qt1">Kidney</span> Disease')
          == "Chronic Kidney Disease")
    check("paragraphs -> lines, list items -> '- item'",
          clean_html("<p>Intro.</p><ul><li>Less salt</li><li>Exercise</li></ul>")
          == "Intro.\n- Less salt\n- Exercise")
    check("heading before <p> gets its own line (MedlinePlus style)",
          clean_html('What is <span class="qt0">asthma</span>?<p>Asthma is a lung disease.</p>')
          == "What is asthma?\nAsthma is a lung disease.")
    check("HTML entities decoded, extra spaces removed",
          clean_html("salt  &amp;   sugar") == "salt & sugar")
    check("empty / None -> ''", clean_html(None) == "" and clean_html("") == "")


def test_parse_saved_sample():
    print("parse_medlineplus_xml (saved real reply)")
    results, total, suggestion = parse_medlineplus_xml(CKD_XML)
    check("3 documents parsed, total count 79", len(results) == 3 and total == 79, (len(results), total))
    first = results[0]
    check("first title cleaned: 'Chronic Kidney Disease'", first["title"] == "Chronic Kidney Disease", first["title"])
    check("first link is the MedlinePlus page",
          first["link"] == "https://medlineplus.gov/chronickidneydisease.html", first["link"])
    check("text has no HTML left", "<" not in first["text"] and "qt0" not in first["text"])
    check("text keeps the treatment facts",
          "dialysis" in first["text"] and "lower blood pressure" in first["text"])
    check("list items kept as '- ' lines", "\n- Choose foods with less salt (sodium)" in first["text"])
    check("source names MedlinePlus", "MedlinePlus" in first["source"])
    check("every result has title, link, text, source",
          all(set(r) == {"title", "link", "text", "source"} and all(r.values()) for r in results))
    check("no spelling suggestion here", suggestion is None)


def test_search_success():
    print("search_medlineplus (mocked)")
    use_fake(fake_get_returning(CKD_XML))
    r = search_medlineplus("  chronic   kidney disease ")
    check("status success with 3 results", r["status"] == "success" and len(r["results"]) == 3, r["message"])
    check("request sent to the documented endpoint", last_call["url"] == "https://wsearch.nlm.nih.gov/ws/query")
    check("request params: db, cleaned term, retmax, tool",
          last_call["params"] == {"db": "healthTopics", "term": "chronic kidney disease",
                                  "retmax": 3, "tool": "agentic_healthcare_capstone"}, last_call)
    check("a timeout is always set", last_call["timeout"] == config.REQUEST_TIMEOUT_SECONDS)


def test_search_no_results():
    print("no results")
    use_fake(fake_get_returning(NO_RESULTS_XML))
    r = search_medlineplus("xyzqqqnotadisease")
    check("count 0 -> no_results, empty list", r["status"] == "no_results" and r["results"] == [], r)

    use_fake(fake_get_returning(SPELLING_XML))
    r = search_medlineplus("diabetis")
    check("misspelling -> no_results with 'Did you mean diabetes?'",
          r["status"] == "no_results" and r["suggestion"] == "diabetes"
          and "Did you mean 'diabetes'" in r["message"], r)


def test_search_failures():
    print("network problems")
    use_fake(fake_get_raising(requests.exceptions.Timeout()))
    r = search_medlineplus("asthma")
    check("timeout -> status timeout, clear message",
          r["status"] == "timeout" and "did not answer" in r["message"] and r["results"] == [], r)

    use_fake(fake_get_raising(requests.exceptions.ConnectionError("no route")))
    r = search_medlineplus("asthma")
    check("no internet -> status no_connection",
          r["status"] == "no_connection" and "internet" in r["message"], r)

    use_fake(fake_get_returning("Service Unavailable", status_code=503))
    r = search_medlineplus("asthma")
    check("HTTP 503 -> status error", r["status"] == "error" and "503" in r["message"], r)

    use_fake(fake_get_returning("<html>not xml"))
    r = search_medlineplus("asthma")
    check("garbled reply -> status error (no crash)", r["status"] == "error", r)


def test_invalid_input():
    print("invalid input")
    use_fake(fake_get_raising(AssertionError("should not call the internet")))
    check("empty query -> invalid", search_medlineplus("   ")["status"] == "invalid")
    check("max_results 0 -> invalid", search_medlineplus("asthma", max_results=0)["status"] == "invalid")
    check("max_results 50 -> invalid", search_medlineplus("asthma", max_results=50)["status"] == "invalid")
    check("max_results 'abc' -> invalid", search_medlineplus("asthma", max_results="abc")["status"] == "invalid")


def test_cache():
    print("24-hour cache")
    os.environ["APP_NOW"] = "2026-10-21 12:00"
    use_fake(fake_get_returning(CKD_XML))
    r = search_medlineplus("Asthma", use_cache=True)
    check("first search -> from MedlinePlus, not cache", r["status"] == "success" and r["from_cache"] is False, r["message"])
    check("  saved one cache file in test_data/cache/medlineplus",
          len(list((config.CACHE_DIR / "medlineplus").glob("*.json"))) >= 1
          and config.CACHE_DIR == TEST_DATA_DIR / "cache")

    use_fake(fake_get_raising(AssertionError("internet should not be used")))
    r = search_medlineplus("  asthma ")
    check("same search again -> from cache, no internet call",
          r["status"] == "success" and r["from_cache"] is True and len(r["results"]) == 3, r["message"])
    check("  message says it came from the cache", "from 24-hour cache" in r["message"], r["message"])
    use_fake(fake_get_returning(CKD_XML))
    check("different max_results -> separate cache entry (fetched)",
          search_medlineplus("asthma", max_results=2)["from_cache"] is False)
    use_fake(fake_get_raising(AssertionError("internet should not be used")))

    os.environ["APP_NOW"] = "2026-10-22 11:59"
    check("23h59m later -> still from cache", search_medlineplus("asthma")["from_cache"] is True)
    os.environ["APP_NOW"] = "2026-10-22 12:01"
    use_fake(fake_get_returning(CKD_XML))
    check("after 24 hours -> fetched again", search_medlineplus("asthma")["from_cache"] is False)

    use_fake(fake_get_raising(requests.exceptions.Timeout()))
    search_medlineplus("migraine")
    use_fake(fake_get_returning(CKD_XML))
    check("a timeout is NOT cached (next search tries again)",
          search_medlineplus("migraine")["from_cache"] is False)

    check("use_cache=False always asks MedlinePlus",
          search_medlineplus("asthma", use_cache=False)["from_cache"] is False)

    use_fake(fake_get_returning(NO_RESULTS_XML))
    search_medlineplus("xyzqqqnotadisease")
    use_fake(fake_get_raising(AssertionError("internet should not be used")))
    check("no_results answers are cached too",
          search_medlineplus("xyzqqqnotadisease")["from_cache"] is True)

    for cache_file in (config.CACHE_DIR / "medlineplus").glob("*.json"):
        cache_file.write_text("{ damaged")
    use_fake(fake_get_returning(CKD_XML))
    check("damaged cache file -> ignored, fetched again",
          search_medlineplus("asthma")["from_cache"] is False)
    del os.environ["APP_NOW"]


def test_logging():
    print("logging")
    use_fake(fake_get_returning(CKD_XML))
    count = len(read_logs())
    search_medlineplus("chronic kidney disease")
    use_fake(fake_get_raising(requests.exceptions.Timeout()))
    search_medlineplus("gout")   # a term that is not in the cache
    rows = read_logs()[count:]
    check("each search writes exactly one log row",
          [r["tool_name"] for r in rows] == ["search_medlineplus"] * 2, rows)
    check("statuses logged: success, timeout",
          [r["status"] for r in rows] == ["success", "timeout"], rows)
    check("tool is wrapped by the logger", hasattr(search_medlineplus, "__wrapped__"))


if __name__ == "__main__":
    originals_before = fingerprint(ORIGINAL_DATA_DIR)
    real_get = requests.get

    test_clean_html()
    test_parse_saved_sample()
    test_search_success()
    test_search_no_results()
    test_search_failures()
    test_invalid_input()
    test_cache()
    test_logging()

    disease_search_tool.requests.get = real_get
    print("safety")
    check("original data/ files are unchanged", fingerprint(ORIGINAL_DATA_DIR) == originals_before)
    print(f"\nAll {passed_count['passed']} checks passed (no internet used).")
