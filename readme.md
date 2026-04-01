search api eval via prediction markets

does better search make an llm a better forecaster?
33 resolved prediction market questions. brier score. the only variable is the search api.

setup:

  cp .env.example .env
  pip install -r requirements.txt
  python3 -m search

why this works

prediction markets give you free, unambiguous ground truth. yes or no, no annotation needed.
brier score is a proper scoring rule. you can't game it, you can only be well-calibrated.
the agentic mode tests search apis the way they're actually used in 2025: as tools inside ai agents.

date filtering is where it gets interesting. search results are filtered to 14 days before resolution
so the llm can't just read a headline confirming the outcome. it has to reason from pre-event
reporting. exa supports this natively via end_published_date. the api itself enforces the cutoff.
tavily doesn't, so we filter client-side, which is inherently lossy.

issues

misdated articles are the main integrity risk. during testing we found exa returning articles
titled "Biden drops out" and "Trump wins the White House" with published dates weeks or months
before the events actually happened. no date offset can fix metadata that's wrong by months.

our mitigations:
- 14-day offset on all search and llm knowledge cutoffs
- undated articles are dropped entirely (they bypass date filters)
- llm prompt explicitly states "the outcome has not yet been determined"
- tavily drops articles with missing or unparseable dates instead of including them

this is an inherent limitation of any search-based eval. misdated articles are rare but exist
in every search index. the 14-day buffer catches most near-resolution leakage, but wildly
misdated articles are a known edge case we can't fully eliminate.

findings

search helps. both exa and tavily beat the no-search baseline on brier score.
agentic mode (llm writes its own queries) outperforms single-query for both providers.
calibration improves dramatically in agentic mode. ece drops from ~0.2 to ~0.12.
search helps most on questions where the baseline llm is uncertain (near 50/50),
which is exactly where you'd want a search api to add value.

exa's native date filtering is a concrete product advantage for time-sensitive workflows.
tavily's lack of it means client-side filtering, which drops articles with missing metadata
and can return fewer usable results.
