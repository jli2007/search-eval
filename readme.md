search api eval via prediction markets

the core idea: resolved prediction markets give you free, unambiguous ground truth.
a question either resolved yes or no. no human judges, no annotations, no subjectivity.
that's the hardest part of any eval and prediction markets just hand it to you.

we score with brier score — a proper scoring rule from the forecasting community.
you can't game it. the only way to minimize it is to output calibrated probabilities.

the pipeline:

1. take a resolved question (e.g. "will bitcoin hit $100k by end of 2024?" — yes)
2. date-filter search results to before the resolution so the llm can't just read the answer
3. run three modes:
   - no search — llm guesses from training data (baseline)
   - single query — question fed directly as a search query
   - agentic — llm writes its own 1-3 queries, gets all results, synthesizes
4. score predicted probability against actual outcome

same llm, same prompts, same questions. only the search api changes.
any difference in score is purely from search quality.

33 questions across 7 categories (politics, geopolitics, crypto, economics, tech, science, sports).
all resolved in 2024, sourced from polymarket, metaculus, and manifold.

setup:

  cp .env.example .env
  pip install -r requirements.txt

usage:

  python3 -m search                                      # full run, exa vs baseline
  python3 -m search --limit 5                            # quick test
  python3 -m search --providers none exa tavily    # compare providers
  python3 -m search --modes no_search agentic             # agentic only

outputs a results table to terminal, saves raw json + charts to search/results/.

adding a new search provider is one class with a name and a search method.
