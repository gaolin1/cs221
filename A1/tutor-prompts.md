# AI-tutor prompts for 1a and 1c

Both questions want the same deliverable: **a link to a chat session transcript**, 15–20 minutes,
interactive. Claude Code has no shareable URL, so run these in a web chatbot that produces share links
(claude.ai, ChatGPT, Gemini), then paste the link into `tex/submission.tex` between the `_1a` / `_1c` tags.

Note the guideline the course itself puts in the prompt: **"Do NOT solve graded work."** Keep the session on
concepts and practice problems, not on the assignment functions.

---

## 1a — NumPy basics (1 pt)

> Teach me basic NumPy operations. Keep the session around 15–20 minutes and interactive.
>
> **Guidelines**
> - Adjust difficulty: if I miss 2 in a row → slow down and simplify; if I get 2 quickly → go a bit harder or add a twist.
> - Use tiny, deterministic examples (small numbers, short tables, brief ASCII diagrams, code blocks). Show actual run-throughs (step-by-step states) where they help.
> - Use code if it clarifies: at most 2 Python/NumPy blocks, each at most 20 lines; ideally no imports beyond `import numpy as np`.
> - Do NOT solve graded work. If I paste any, refuse and make a similar practice item instead.
> - Be concise. Don't make notation complicated or unnecessary. Focus on intuitions. Code blocks should be short with simple variable names.
> - I may interrupt and ask to start somewhere else. Adjust and adapt from there.
>
> **Suggested flow** (flexible; use judgment)
> 1. Start by explaining the topic in a few simple but technically accurate sentences (core idea in ≤6 sentences: intuition + minimal notation + why it matters + one common pitfall). Use basic notation and visualisations if needed. For programming topics it's fine to start with concise code blocks.
> 2. Quick check-in (≤1–2 min): ask 2 short questions about my prior exposure and my specific goal.
> 3. Exercises: use a tiny instance and show intermediate states. Give exercises that build on each other. It's fine to give several at once.
> 4. Three quick checks of increasing difficulty; give a hint first; reveal answers only after I try or ask.
> 5. Assessment and feedback: tell me what I understood well, what needs improvement, and specific concepts to review.
> 6. Session reflection: ask me how the session went — what worked, what could improve, whether pacing and difficulty were right.
> 7. Recap: a 60-second summary.

---

## 1c — einsum and einops (1 pt)

Identical prompt, with only the first line swapped:

> Teach me how einsum works and how to use the einops library with NumPy. Keep the session around 15–20 minutes and interactive.

…then the same **Guidelines** and **Suggested flow** blocks as above.

---

## Worth steering toward in the 1c session

You have already met these; a good session should reinforce rather than re-introduce them. Bring them up if
the tutor doesn't:

- The three axis roles — carried / contracted / aligned (the table in the 1g notes).
- Why the number of distinct letters is the complexity (from 1b).
- `(g d)` vs `(d g)` inside `rearrange` parentheses — the odometer rule.
- einops vs `np.einsum` string dialects (whitespace-separated vs one char per axis).
- Ellipsis `...` for leading batch axes.

## Worth steering toward in the 1a session

- Broadcasting: right-alignment, the equal-or-1 rule, and that stretching is virtual (no copy).
- `arr[:, None]` vs `arr[None, :]` — adding an **axis**, not data; `.size` is unchanged.
- Views vs copies, and which operations mutate in place.
- `axis=` on reductions (`sum`, `mean`) and what `keepdims=True` is for.
- Fancy/boolean indexing versus `np.where`.

## Submitting

Paste the URL into `tex/submission.tex`:

```latex
% <SCPD_SUBMISSION_TAG>_1a
\begin{answer}
  % ### START CODE HERE ###
  \url{https://...}
  % ### END CODE HERE ###
\end{answer}
% <SCPD_SUBMISSION_TAG>_1a
```

Then build the PDF with `make` from the repo root (it runs `latexmk submission.tex` inside `tex/`).
Check the share link opens in a private/incognito window — an unshared link renders as a 404 for the grader.
