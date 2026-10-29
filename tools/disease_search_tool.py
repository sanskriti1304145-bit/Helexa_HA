"""
disease_search_tool.py - search MedlinePlus for disease and treatment information.

MedlinePlus is the U.S. National Library of Medicine's consumer health
website. Its free web service needs no API key.
Docs: https://medlineplus.gov/about/developers/webservices/

Functions in this file:
    search_medlineplus(query, max_results)  -> calls the web service (logged tool)
    parse_medlineplus_xml(xml_text)          -> turns the XML reply into clean results
    clean_html(text)                         -> strips HTML tags from summary text

How the web service works:
    We send a GET request such as
        https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term=chronic+kidney+disease
    and get back XML like this (shortened):
        <nlmSearchResult>
          <count>79</count>
          <list>
            <document rank="0" url="https://medlineplus.gov/chronickidneydisease.html">
              <content name="title">Chronic Kidney Disease</content>
              <content name="FullSummary">&lt;p&gt;You have two kidneys...</content>
              <content name="snippet">...</content>
            </document>
          </list>
        </nlmSearchResult>
    The title and summary contain HTML (paragraphs, lists, and
    <span class="qt0"> tags that highlight search words), so we clean it.

Every result we return has: title, link, text, source.
The "link" is the MedlinePlus page, so answers can cite where facts came from.

24-hour cache:
    NLM asks apps to cache results for 12-24 hours. After a successful
    search we save the answer as a small JSON file in cache/medlineplus/.
    The same search within 24 hours is answered from that file (the result
    then has "from_cache": True). Errors and timeouts are never cached, so
    the next search tries the internet again.

Status values: "success", "no_results", "timeout", "no_connection",
               "invalid", "error".
"""

import hashlib
import html
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import requests

import config
from tools.logger import log_tool_call

SOURCE_NAME = "MedlinePlus (U.S. National Library of Medicine)"
MAX_RESULTS_ALLOWED = 10   # the service returns 10 by default; we never ask for more
CACHE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
CACHEABLE_STATUSES = ("success", "no_results")   # never cache errors or timeouts


# ---------------------------------------------------------------------------
# 1. Clean the HTML inside the summaries
# ---------------------------------------------------------------------------

def clean_html(text):
    """Turn MedlinePlus HTML into plain, readable text.

    "<p>CKD means...</p><ul><li>Less salt</li></ul>"  ->  "CKD means...\n- Less salt"
    """
    if not text:
        return ""
    text = re.sub(r"<li[^>]*>", "\n- ", text)              # list items -> "- item"
    # Paragraph starts AND ends -> new line. (MedlinePlus puts headings such as
    # "What is asthma?" as plain text right before a <p>, so the start matters too.)
    text = re.sub(r"<p[^>]*>|</p>|<br\s*/?>|</ul>|</ol>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)                    # remove every other tag
    text = html.unescape(text)                             # "&amp;" -> "&"
    # Tidy spaces inside each line and drop empty lines.
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------
# 2. Read the XML reply
# ---------------------------------------------------------------------------

def parse_medlineplus_xml(xml_text):
    """Turn the web service's XML into a list of clean result dicts.

    Kept separate from the internet call so tests can feed it a saved sample.
    Returns (results, total_count, spelling_suggestion).
    Raises ET.ParseError if the text is not valid XML.
    """
    root = ET.fromstring(xml_text)

    total_count = int(root.findtext("count", default="0") or 0)
    # MedlinePlus sometimes suggests a corrected spelling for the query.
    suggestion = clean_html(root.findtext("spellingCorrection", default="")) or None

    results = []
    for document in root.iter("document"):
        # Each <content name="..."> holds one field; collect them in a dict.
        fields = {}
        for content in document.findall("content"):
            name = content.get("name")
            if name not in fields:   # some names (altTitle) repeat; keep the first
                fields[name] = content.text or ""

        # Use the full summary; fall back to the short snippet if missing.
        text = clean_html(fields.get("FullSummary") or fields.get("snippet", ""))
        results.append({
            "title": clean_html(fields.get("title", "")),
            "link": document.get("url", ""),
            "text": text,
            "source": SOURCE_NAME,
        })

    return results, total_count, suggestion


# ---------------------------------------------------------------------------
# 3. The 24-hour cache
# ---------------------------------------------------------------------------

def _cache_file(query, max_results):
    """Each different search gets its own cache file.

    The file name is a short "hash" (fingerprint) of the search, so
    "Chronic Kidney Disease" and "chronic kidney disease" share one file.
    """
    key = f"{config.MEDLINEPLUS_DB}|{query.lower()}|{max_results}"
    name = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return config.CACHE_DIR / "medlineplus" / f"{name}.json"


def _read_cache(query, max_results):
    """Return the cached result if it exists and is younger than 24 hours, else None."""
    path = _cache_file(query, max_results)
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved_at = datetime.strptime(saved["saved_at"], CACHE_TIME_FORMAT)
    except (FileNotFoundError, ValueError, KeyError):
        return None   # no cache yet, or a damaged file: just search again
    if config.get_now() - saved_at > timedelta(hours=config.SEARCH_CACHE_HOURS):
        return None   # too old
    return saved["result"]


def _write_cache(query, max_results, result):
    """Save a result to the cache. A failure here never breaks the search."""
    path = _cache_file(query, max_results)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = {"saved_at": config.get_now().strftime(CACHE_TIME_FORMAT),
                 "query": query, "result": result}
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        os.replace(temp_path, path)
    except Exception as error:
        print(f"WARNING: could not save MedlinePlus cache: {error}")


# ---------------------------------------------------------------------------
# 4. The tool: search MedlinePlus
# ---------------------------------------------------------------------------

@log_tool_call
def search_medlineplus(query, max_results=3, use_cache=True):
    """Search MedlinePlus health topics and return clean results.

    Example: search_medlineplus("chronic kidney disease")
    Returns {"status": "success", "message": ..., "from_cache": False, "results": [
                {"title": ..., "link": ..., "text": ..., "source": ...}, ...]}

    use_cache=False skips the cache and always asks MedlinePlus.
    """
    query = " ".join(str(query or "").split())
    if not query:
        return {"status": "invalid", "message": "Please give a disease or topic to search for.",
                "results": []}
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        return {"status": "invalid", "message": "max_results must be a whole number.",
                "results": []}
    if not 1 <= max_results <= MAX_RESULTS_ALLOWED:
        return {"status": "invalid",
                "message": f"max_results must be between 1 and {MAX_RESULTS_ALLOWED}.",
                "results": []}

    # --- 1. Try the cache first ---
    if use_cache:
        cached = _read_cache(query, max_results)
        if cached is not None:
            return {**cached, "from_cache": True,
                    "message": cached["message"] + " (from 24-hour cache)"}

    # --- 2. Ask MedlinePlus ---
    result = _fetch_from_medlineplus(query, max_results)
    result["from_cache"] = False

    # --- 3. Save good answers for next time ---
    if result["status"] in CACHEABLE_STATUSES:
        _write_cache(query, max_results, result)
    return result


def _fetch_from_medlineplus(query, max_results):
    """Make the real web request and turn the reply into a result dict."""
    params = {
        "db": config.MEDLINEPLUS_DB,
        "term": query,                 # requests URL-encodes this for us
        "retmax": max_results,
        "tool": config.MEDLINEPLUS_TOOL_NAME,
    }
    try:
        response = requests.get(config.MEDLINEPLUS_URL, params=params,
                                timeout=config.REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()    # turns HTTP errors (404, 500...) into exceptions
    except requests.exceptions.Timeout:
        return {"status": "timeout",
                "message": f"MedlinePlus did not answer within {config.REQUEST_TIMEOUT_SECONDS} "
                           "seconds. Please try again later.",
                "results": []}
    except requests.exceptions.ConnectionError:
        return {"status": "no_connection",
                "message": "Could not reach MedlinePlus. Please check the internet connection.",
                "results": []}
    except requests.exceptions.RequestException as error:
        return {"status": "error", "message": f"MedlinePlus request failed: {error}",
                "results": []}

    # --- Read the reply ---
    try:
        results, total_count, suggestion = parse_medlineplus_xml(response.text)
    except ET.ParseError as error:
        return {"status": "error",
                "message": f"MedlinePlus sent a reply we could not read: {error}",
                "results": []}

    if not results:
        message = f"MedlinePlus has no health topics matching '{query}'."
        if suggestion:
            message += f" Did you mean '{suggestion}'?"
        return {"status": "no_results", "message": message, "results": [],
                "suggestion": suggestion}

    titles = ", ".join(r["title"] for r in results)
    return {"status": "success",
            "message": f"Found {len(results)} of {total_count} MedlinePlus topic(s) "
                       f"for '{query}': {titles}.",
            "results": results,
            "suggestion": suggestion}
