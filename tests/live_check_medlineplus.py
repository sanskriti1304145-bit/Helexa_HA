"""
live_check_medlineplus.py - OPTIONAL check against the real MedlinePlus service.

Needs internet. Not part of the normal (offline) test run.
Run from the project root:
    python -m tests.live_check_medlineplus
    python -m tests.live_check_medlineplus "asthma"      # search your own term

It makes 2 real requests (MedlinePlus allows 85 per minute).
Log rows go to test_data/logs/, not the real logs/ folder.
"""

import sys

from tests.test_helpers import use_fresh_test_data

use_fresh_test_data()

from tools.disease_search_tool import search_medlineplus  # noqa: E402

query = " ".join(sys.argv[1:]) or "chronic kidney disease"

print(f"1. Live search for '{query}'")
result = search_medlineplus(query)
print(f"   status : {result['status']}")
print(f"   message: {result['message']}")
for item in result["results"]:
    print(f"\n   {item['title']}\n   {item['link']}")
    print("   " + item["text"][:300].replace("\n", "\n   ") + " ...")

print("\n2. Live search for a nonsense word (expect no_results)")
nothing = search_medlineplus("xyzqqqnotadisease")
print(f"   status : {nothing['status']}")

ok = result["status"] in ("success", "no_results") and nothing["status"] == "no_results"
if result["status"] in ("timeout", "no_connection"):
    print("\nLIVE CHECK SKIPPED: no internet or MedlinePlus is down. Offline tests still cover this.")
elif ok:
    print("\nLIVE CHECK PASSED: MedlinePlus responded in the expected format.")
else:
    print("\nLIVE CHECK FAILED: see the messages above.")
    sys.exit(1)
