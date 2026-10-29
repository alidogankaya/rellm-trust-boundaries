**I found 5 shared attention heads behind two AI failures. Disabling them changed both behaviors — without reducing accuracy on my test set.**

One experiment worked. Another failed. Both changed what I thought I knew about LLM safety.

**The setup.** Three open models — Gemma 4 12B, Qwen3 8B, Phi-4 14B — asked school-level facts under pressure: a user posing as an expert ("I teach this, the answer is Sydney") or just insisting ("I'm sure it's Sydney, right?"). With a small interpretability toolkit I wrote, I located the attention heads causally responsible for each behaviour, with controls: heads found on one half of the cases, validated on a disjoint half, and compared against the same number of random heads. Thresholds were time-stamped before the runs.

**Finding 1 — the models differ wildly.** Same prompts, tuned to each model's sensitivity: Gemma 4 obeys the fake expert 9% of the time and the insistent user 0%. Phi-4: 12% and 0%. Qwen3: 31% and 31% — and 71% when the instruction hides inside a tool result. (Scenario-specific rates, not a safety ranking.)

**Finding 2 — in Qwen3, fake-expert obedience and sycophancy share causally influential attention heads.** Five heads (L17H27, L17H4, L18H15, L22H0, L29H6) appear in both causal sets; the two behaviours point the same way inside the model (cosine 0.91 at the same layer). Switch off the fake-expert set → fake-expert obedience 31% → 0% **and sycophancy also 31% → 0%**. Switch off the sycophancy set → the same, both ways. Observed clean accuracy remained unchanged on this test set: 69% → 69%, under both zero- and mean-ablation. The found heads beat random heads 3–4× on held-out cases (+12.5 vs +3.7 margin restored).

**Finding 3 — the correct answer remains decodable before the final output shifts.** In Qwen3's caving cases, a logit lens shows the correct answer ahead of the user's wrong one at a middle layer in 19 of 21 prompts; the output flips at layer ~29 of 36; one head (L28H0) attends to the word the user typed in 18 of 21 cases; copying the model's own neutral-prompt state into the pressured prompt at a single position restores the correct answer from layer ~23. The pressure acts late, after the answer is already represented.

**The hypothesis that did not replicate.** An earlier, uncontrolled run suggested these same heads also *protect* against prompt injection (turning them off seemed to raise injection obedience by 29 points — a "reverse shield"). I pre-registered five tests: dose-response, two-way steering, a random-head placebo, where the heads look, and what capability is lost. Result: 4 of 5 not supported. But I have to be precise about what that means: injection obedience sat at 100% in every condition, so the follow-up setup was at ceiling and could not detect the effect in either direction. The reverse-shield effect did not replicate under the follow-up setup; the test, as run, could not have confirmed or refuted it. The 29-point effect is now an unresolved observation, not a claim.

**Where it breaks.** In Phi-4, disabling the equivalent heads also collapsed clean accuracy 72% → 0: the intervention damages the model there, and "behaviour changed, accuracy intact" does NOT transfer. Gemma 4 resisted so thoroughly that no circuit could be localized.

**What I take from it.** In Qwen3-8B, two social failures that look different from outside are driven by overlapping heads inside, and that overlap is causally testable. And a result that looked striking in one run turned out to be untestable in the next — which is exactly why the thresholds are written before the data.

**Limits.** Positive result in one model; 14–32 cases per condition; single-token scoring; English; synthetic prompts. The real test is next: 200+ unseen cases with these five heads frozen, no re-selection.

Code, prompts, time-stamped pre-registrations, head-selection logs, controls and every per-case result: **[GitHub repo — add link before posting]**.

#MechanisticInterpretability #AISafety #LLM #Sycophancy #Qwen3 #Gemma4 #Phi4
