search api eval via prediction markets

the core idea: resolved prediction markets give you free, unambiguous ground truth.
a question either resolved yes or no. no human judges, no annotations, no subjectivity.
that's the hardest part of any eval and prediction markets just hand it to you.

we use brier score — a proper scoring rule from the forecasting community.
the only way to minimize it is to output calibrated probabilities. you can't game it.

the pipeline:

1. take a resolved question (e.g. "will tiktok be banned in the us by jan 19 2025?" — yes)
2. date-filter search results to before the resolution so the llm can't just read the answer
3. run three modes:
   - no search — llm guesses from training data (baseline)
   - single query — question fed directly as a search query
   - agentic — llm writes its own 1-3 queries, gets all results, synthesizes
4. score predicted probability against actual outcome with brier score

same llm, same prompts, same questions. only the search api changes.
any difference in score is purely from search quality.

run:

cp .env.example .env (fill in your api keys)
pip install -e .
predsearch --providers exa tavily brave --mode agentic
