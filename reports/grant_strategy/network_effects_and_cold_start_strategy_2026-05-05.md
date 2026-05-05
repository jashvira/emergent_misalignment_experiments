# Grant Network Effects And Cold-Start Strategy

Date: 2026-05-05

## Bottom Line

The user's concern is correct: many public AI safety grants to "independent researchers" are not truly cold-start grants. They often rely on a prior trust graph: MATS mentor, former advisor, known collaborator, regrantor endorsement, or public track record in the field.

This does not make applying pointless, but it changes the strategy. A cold applicant should not lead with a large broad ask. The highest-probability route is to convert current artifacts into trust: public writeup, exact repo, clean plots, small compute ask, and targeted feedback from 3-6 relevant researchers before requesting serious runway.

## Evidence From Public Grants

### Manifund Is Strongly Trust-Graph Driven

Manifund is structurally built around regrantors with personal expertise and independent budgets. Its own regranting page says regrantors can fund hidden opportunities through personal networks, with grants made in days rather than months.

Public evidence:

- `Exploring novel research directions in prosaic AI alignment`: funded at `$30k`. The comments show explicit reliance on endorsements:
  - Marcus Abramovitch says he funded after seeing Neel's and Evan's endorsements.
  - Neel Nanda says Lawrence is experienced and he trusts his judgement.
  - Evan Hubinger says he was recommended the grant by someone he trusts and had encouraged Lawrence to post it.
- `Scaling Training Process Transparency`: funded at `$5,150`. Evan Hubinger's comment says he knew Rob was capable because he had mentored him previously, and the conflict section says Rob was his SERI MATS mentee.
- `Mitigating Reward Hacking Through RL Training Interventions`: funded at `$7,900`. Neel Nanda says the project was done with him during MATS and discloses a mild conflict of interest.
- `Compute and other expenses for LLM alignment research`: funded at `$400,100`. The donor comment says Ethan Perez reached out directly, the donor respected Ethan's competence, and Ethan was mentoring the projects.

Interpretation:

Manifund can fund cold proposals, but the examples most similar to us were not cold. They were de-risked by known mentors, prior MATS affiliation, or respected endorsers.

Cold-start implication:

Manifund should be treated as a public credibility-building route, not just an application form. The proposal needs public artifacts and ideally 1-2 comments from people who have actually looked at the work.

### BlueDot Is Community-Gated

BlueDot Rapid Grants explicitly targets BlueDot course participants, alumni, facilitators, and active community members. Their blog says most grants went to course participants/facilitators doing technical AI safety work.

Interpretation:

BlueDot is a good route only if the user is already in or can quickly enter the BlueDot community. If not, its apparent speed does not apply.

Cold-start implication:

If the user is not BlueDot-connected, deprioritize BlueDot unless joining a course/community channel is easy and honest.

### LTFF Is Less Publicly Network-Gated, But References Matter

LTFF has a formal application route and funds technical AI safety research, researcher training, and independent work. The public page lists grants such as a `$40k` LM-tools alignment stipend and a `$98k` cybersecurity/alignment risk-assessment org.

Interpretation:

LTFF is more plausible for a cold applicant than Manifund because it has a private committee process rather than public regrantor comments. But the same basic trust problem remains: reviewers need confidence the applicant can execute, especially for runway.

Cold-start implication:

LTFF is still viable, but the ask should be smaller and sharper: 6-10 weeks of work, not a vague 6-month research programme. Include exact artifacts, public repo, current pilot results, a concrete budget, and ideally references from credible ML/security people even if not AI-safety famous.

### Coefficient Career Development Can Work For Cold Transition

Coefficient's career-development page explicitly supports people transitioning into AI safety or global-catastrophic-risk careers. Past recipients include people moving from software engineering, chemistry/math, chemical engineering, etc.

Interpretation:

This is one of the more honest cold-start routes if framed as career transition. It does not require the project to already be vouched for by an AI safety insider, but it does require a credible personal trajectory and a clear career-capital story.

Cold-start implication:

Use this if asking for personal runway. Do not frame it as "fund my experiment"; frame it as "fund my focused transition into empirical AI safety, with this experiment as the concrete output."

### Schmidt / AISI Are Not Cold-Solo Friendly

Schmidt Trustworthy AI and Interpretability RFPs are intellectually relevant but large. They ask for rigorous, scoped proposals; Schmidt explicitly wants core questions, validity arguments, and concrete baselines. AISI-style programmes are similarly high-bar and more natural for teams.

Interpretation:

These are not impossible, but cold solo applications without an established collaborator are low probability.

Cold-start implication:

Do not spend the first push on Schmidt/AISI unless a collaborator or institutional sponsor is found. Use the pilot to get that collaborator.

## Revised Ranking Under No-Network Assumption

If the user genuinely knows no one in AI safety:

1. **LTFF small, private, artifact-heavy ask**: best serious cold route.
2. **Coefficient career-transition ask**: best personal-runway route if the user's story fits.
3. **Manifund public microgrant**: viable only if used to attract comments/endorsements, not as a pure cold post.
4. **BlueDot**: strong only if the user is in the community; otherwise lower priority.
5. **Foresight**: possible if framed as open-source infrastructure and node participation is real.
6. **Schmidt/AISI**: prepare later, after external validation or collaborator.

## Recommended Cold-Start Playbook

### Step 1: Produce A Public Technical Brief

One short, serious brief:

- 2-page PDF or LessWrong/Alignment Forum post.
- Core hypothesis.
- Current Betley table.
- Exact repo/report links.
- What would falsify it.
- What funding buys.

This is not a grant proposal. It is a trust object.

### Step 2: Ask For Review, Not Money

Send the brief to 8-12 people in adjacent areas:

- Emergent misalignment authors.
- Persona features authors.
- School of Reward Hacks authors.
- AI safety evals researchers.
- MATS stream mentors.
- Security/evals people who can judge the experiment.

Ask:

> I have a small pilot result on awareness-stratified emergent misalignment. Could you sanity-check whether the causal question is worth pursuing and whether there is a better eval/matching design?

Do not ask first for money. Ask for technical criticism.

### Step 3: Convert Any Useful Reply Into Credibility

Possible useful outcomes:

- A public comment.
- Permission to quote a private sentence.
- A small design suggestion.
- A referral to another researcher.
- A "this is interesting but flawed" note that improves the proposal.

One serious external comment changes the application substantially.

### Step 4: Apply Small First

Cold applicants should not ask first for `$75k` unless the proposal has external validation.

Better sequence:

1. `$3k-$8k` compute/API microgrant.
2. Run one clean replication.
3. Publish update.
4. Apply for `$40k-$75k` serious runway.

### Step 5: Use Existing Non-AIS Credibility

If the user has ML/security/software credentials outside AI safety, use them. For this project, execution credibility matters:

- Working codebase.
- HF-uploaded adapters.
- OpenAI Batch judge pipeline.
- W&B training runs.
- Exact Betley eval reproduction.
- Prompt provenance discipline.

This partly substitutes for insider trust.

## Practical Conclusion

The current project is fundable, but not best approached as "submit cold to every funder."

The right move is:

1. Turn results into a public technical brief.
2. Get 2-3 technical sanity checks.
3. Apply to LTFF/Manifund with those checks attached or quoted.
4. Keep the first ask modest unless we get a credible endorser.

The strongest honest framing:

> I do not yet have deep AI safety network trust, but I have already run a concrete pilot with public artifacts. I am asking for a small amount of funding to test whether the result survives a cleaner replication.

