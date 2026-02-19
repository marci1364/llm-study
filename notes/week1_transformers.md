# Week 1 — Transformers Notes

## 10 bullets (self-attention + multi-head + Q/K/V)

1. **Self-attention** lets each token mix information from other tokens to build a context-aware representation.
2. For each token embedding \(x_i\), we compute **Query, Key, Value** via linear projections:
   \[
   q_i = x_i W_Q,\quad k_i = x_i W_K,\quad v_i = x_i W_V
   \]
3. The **attention score** from token \(i\) to token \(j\) is typically a dot product:
   \[
   s_{ij} = q_i^\top k_j
   \]
4. We apply **scaled dot-product attention** to keep scores stable when dimensions grow:
   \[
   s_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}}
   \]
5. We convert scores into weights with **softmax** (weights sum to 1 over \(j\)):
   \[
   \alpha_{ij} = \mathrm{softmax}(s_{ij})
   \]
6. The output representation for token \(i\) is a **weighted sum of values**:
   \[
   o_i = \sum_j \alpha_{ij} v_j
   \]
7. **Query (Q)** represents what token \(i\) is “looking for” in other tokens.
8. **Key (K)** represents what token \(j\) “offers”; it’s what queries match against.
9. **Value (V)** is the actual information that gets aggregated when token \(j\) is attended to.
10. **Multi-head attention** runs attention multiple times in parallel with different projections so each head can capture different relationships:
   \[
   \text{head}_h = \mathrm{Attention}(XW_Q^h,\; XW_K^h,\; XW_V^h)
   \]
   and then combines them:
   \[
   \mathrm{MHA}(X) = \mathrm{Concat}(\text{head}_1,\ldots,\text{head}_H) W_O
   \]

## Quick intuition (in plain language)
- Attention answers: “Which other tokens matter for this token right now?”
- Multi-head attention means: “Look for multiple kinds of relationships at once.”
