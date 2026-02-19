# \# Week 1 — Transformers Notes

# 

# \## 10 bullets

# 

# 1\. \*\*Self-attention\*\* lets each token look at other tokens in the same sequence to build a context-aware representation of itself.

# 2\. In self-attention, each token produces three vectors: \*\*Query (Q)\*\*, \*\*Key (K)\*\*, and \*\*Value (V)\*\* by linear projections of its embedding.

# 3\. The \*\*attention score\*\* between token \*i\* and token \*j\* is computed by a similarity function, typically the dot product: `Q\_i · K\_j`.

# 4\. Scores are scaled by `1/sqrt(d\_k)` to keep values numerically stable before applying softmax.

# 5\. \*\*Softmax\*\* turns scores into weights that sum to 1, so we get a weighted average of values.

# 6\. The output for token \*i\* is the weighted sum of \*\*V\*\* vectors from all tokens: it “mixes” information from other tokens into token \*i\*.

# 7\. \*\*Q (Query)\*\* represents what this token is looking for; it encodes the current token’s “question.”

# 8\. \*\*K (Key)\*\* represents what each token offers; it’s used to match against queries.

# 9\. \*\*V (Value)\*\* is the information actually passed along once a token is attended to.

# 10\. \*\*Multi-head attention\*\* runs several attention mechanisms in parallel so the model can learn different kinds of relationships at once (e.g., syntax, coreference, topic), then concatenates and mixes them.

# 

# \## Quick intuition

# \- Attention = "which tokens matter for this token right now?"

# \- Multi-head = "learn multiple views of relevance at the same time."



