### GrokRuit: xAI Recruiter

Elon Musk personally met with the first 3,000 hires at SpaceX. Sourcing is the primary bottleneck for recruitment: finding the right engineers can make or break a company. Getting from 100K candidates to 10 requires human recruiters to search, rank, and evaluate one by one.

X is the best place to find underrated engineers on the Internet. GrokRuiter is a different approach to traditional recruiting - it uses the revealed preferences of xAI engineers and social network interactions to identify promising candidates. An engineer who has made open source contributions and has few but highly-respected followers/interactions is often overlooked but they're exactly who we want to surface and rank highly.

Our approach uses social network interactions, knowledge graphs, and Grok as a judge for sourcing. Recruiters can add sources from their network they want to expand in knowledge graph. We first use X API to crawl and curate a social graph of potential candidates, then rank using personalized Pagerank+Grok fast 4.1 to converge upon those with high signal, and finally use grok 4 fast with reasoning for in-depth eval + tool use web search to score candidates. In addition, our method can be used for online improvement of the score function based on recruiter feedback.

The entire process, from source to automated outreach, takes just a few minutes. We aim to make this immediately deployable and useful as an end-to-end solution to xAI recruitment process.

Try it live! https://grok-underrated-recruiter.vercel.app/

## Inspiration

One of Grok’s first product uses will be as an AI recruiter. We were inspired by the fact that this will be useful as a product and can automate recruitment pipelines, one of the first cases of autonomous decision making and self-improvement in deployed settings. When discussing with Christina and xAI engineers, it was brought up that sourcing is the primary bottleneck. Automating sourcing is the first and most crucial step to AI recruiting and online improvement.

## What it does

GrokRank addresses this challenge using PageRank + Grok as LLM judge to find underrated candidates.

Human recruiters use explore (discover candidates through existing connections) and exploit (in-depth analysis on top K candidates). GrokRank automates the process using a 4-stage pipeline:

1. Social Knowledge Graph Construction

- Start with initial connections accounts (e.g. xAI engineers)
- Expand via follows, retweets, replies to use existing connections to build interaction edges with weighted signals
- Following edges weighted 5.0, retweets 3.0, replies 2.5, likes 1.0 (passive engagement)
- Result: 17,323 nodes, 28,841 edges, 12,050 candidates

2. Fast LLM Screen (grok-4-1-fast-non-reasoning)

- Screen bios for technical relevance
- Filter out xAI/X employees (already hired), organization accounts, non-technical accounts, first-pass to output engineers
- 16K → 500 candidates in minutes (~45% filter rate)

3. PageRank + Learnable Score

- Personalized PageRank with seeds as the personalization vector
- Boosts low profile count but high ptential which may be missed by recruiters (PageRank / log(followers))
- PageRank has learnable components which can be updated with online feedback from recruiters

4. Deep Evaluation (grok-4-1-fast-reasoning + xAI Search Tools)

- Autonomous information gathering via web_search and x_search tools
- Searches GitHub, LinkedIn, and X posts for each candidate
- Scores on 5-criterion rubric (0-100 scale): Technical Depth (25%), Project Evidence (25%), Mission Alignment (20%), Exceptional Ability (20%), Communication (10%)

## How we built it

- X API (OAuth1 + Bearer Token) for graph crawling
- xAI API with multiple Grok models:
  - grok-4-1-fast-non-reasoning for fast screening
  - grok-4-1-fast-reasoning + tool calls for deep evaluation
  - grok-3-mini for natural language search to show thinking trace
- xAI SDK with web_search() and x_search() tools
- NetworkX for PageRank computation
- FastAPI backend with SSE streaming
- React/TypeScript frontend with force-graph visualization
- SQLite for saved candidates and DM history
- Vercel for deployment

## Challenges we ran into

1. X API rate limits: Aggressive caching at every layer (X responses, Grok evaluations, graph state) to minimize API calls
2. X interaction are noisy, high volume of accounts: use Grok fast 4.1 as first-pass to aggressively filter down candidates
3. Making the tool useable/useful: getting the UI working, backend and frontend with social graphs, query should produce candidates which we think are good hires...

## Accomplishments that we're proud of

We worked hard to make something which can be immediately useful and deployed.

- End-to-end automation: From X handle to scored candidate profile with GitHub/LinkedIn links
- Real-time graph expansion: Add new sources and run screening directly from the UI
- AI-generated personalized DMs: Using candidate evaluation data to craft relevant outreach

## What's next for GrokRuit

Finetune open weight Grok/LLM adapter to learn from recruiter feedback (whether or not the candidate made it through, +/-1 reward)
Integrate with interviewer preparation and automate the rest of the recruiting pipeline
Eval on synthetic dataset

### Try It

Live demo: https://grok-underrated-recruiter.vercel.app/

Search for a specific role or set of skills, and get the highest-scored candidates.
