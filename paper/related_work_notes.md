# Related work: source notes for §2

Neutral summaries to draw on when writing §2. They were written from general knowledge, **not** checked against the
publications in this session. **Every summary is marked "verify"**: check its content and bibliographic details
against the source before citing it. BibTeX keys refer to `paper/refs.bib`, where every entry is also marked
UNVERIFIED. The last column of each heading says how the work bears on this paper. That is a suggestion for the
author, not a claim made in the paper.

---

## Brock & Durlauf (2001), "Discrete choice with social interactions" — `brock2001discrete` — **verify**

Brock and Durlauf model individuals making a binary choice whose utility includes a private term, a social term that
depends on the expected choices of a reference group, and a random term with an extreme-value distribution. That gives
a logit choice probability. Under rational expectations, the mean-field equilibrium solves a self-consistency equation
of the same form as the Curie–Weiss mean-field Ising model. When the social interaction is strong relative to the
randomness, there can be multiple equilibria. The paper also discusses when social effects can be identified from data.
*Relevance:* the agent update rule in this paper is a logit choice with a social term, and Brock and Durlauf's
multiple equilibria are the "tipping points" here.

## Manski (1993), "Identification of endogenous social effects: The reflection problem" — `manski1993identification` — **verify**

Manski separates three reasons why people in a group behave alike:
- endogenous effects (behaviour responds to the group's behaviour);
- contextual effects (behaviour responds to the group's characteristics);
- correlated effects (members share unobserved traits or environments).

In the linear-in-means model, the group's mean outcome is itself determined by members' behaviour. Endogenous effects
then cannot in general be told apart from contextual effects using observational data on outcomes and group
characteristics. This is the "reflection problem".

*Relevance:* this paper sets each agent's contacts' positions experimentally, which sidesteps the reflection problem
for the single-agent response function.

## Kirman (1993), "Ants, rationality, and recruitment" — `kirman1993ants` — **verify**

Kirman builds a stochastic model of agents choosing between two equivalent options, motivated by experiments in which
ants choose between two identical food sources. Agents switch options either at random (self-conversion) or by meeting
another agent and copying its choice (recruitment). Depending on the relative rates, the share of agents on one option
either fluctuates around one half or spends long periods near the extremes and switches between them occasionally. The
model is used to explain herding and fluctuating aggregate behaviour in markets without changes in fundamentals.

*Relevance:* this paper adds a first-contact "voter" (copying) step to the surrogate where it fitted better. Contact-order
primacy in LLM agents resembles recruitment by a randomly met contact.

## Bradley & Terry (1952); Luce (1959) — `bradley1952rank`, `luce1959individual` — **verify**

Bradley and Terry propose a model for paired comparisons: each item i has a positive worth π_i, and i is preferred to j
with probability π_i/(π_i + π_j). Equivalently, the log-odds of preferring i to j is the difference of the two items'
log-worths. Luce's *Individual Choice Behavior* sets out the choice axiom (independence from irrelevant alternatives)
and shows that it implies choice probabilities proportional to item scale values, of which Bradley–Terry is the
two-item case. Together these underpin the multinomial logit and much later work on ranking from pairwise preferences.

*Relevance:* the binary logit used for the agents' A/B response belongs to this family. Labels and option order act
as additive "worth" offsets, corresponding to the letter and order fields.

## Smets' Transferable Belief Model (Smets & Kennes 1994) — `smets1994transferable` — **verify**

The Transferable Belief Model represents uncertain belief with belief functions, in the Dempster–Shafer tradition,
rather than probabilities. It separates two levels:
- a "credal" level, where beliefs are held and combined (for example by an unnormalised Dempster rule, allowing mass
  on the empty set);
- a "pignistic" level, where beliefs are turned into probabilities only when a decision has to be made.

The model is justified axiomatically and does not depend on an underlying probability model.

*Relevance (to be decided by the author):* a possible frame for distinguishing what an agent "believes" about its
contacts from the probability it acts on; the paper does not currently use it.

## Viana & Bray (1985), "Phase diagrams for dilute spin glasses" — `viana1985phase` — **verify**

Viana and Bray study Ising spin glasses on dilute random graphs, where each spin has a finite mean number of
neighbours and the couplings are random (including the ±J case with a fraction of negative bonds). Using replica
methods, they derive the phase diagram in the plane of temperature and dilution or bond bias. It contains
paramagnetic, ferromagnetic and spin-glass phases, with transition lines that depend on the mean connectivity and the
bond distribution. The model became the standard example of a finite-connectivity (diluted) mean-field spin glass.

*Relevance:* this is the symmetric baseline of §3.1. The cavity transition lines used for validation are of the
Viana–Bray type.

## Crisanti & Sompolinsky (1987), "Dynamics of spin systems with randomly asymmetric bonds" — `crisanti1987dynamics` — **verify**

Crisanti and Sompolinsky study spin-glass dynamics when the couplings J_ij and J_ji are not equal: each bond has a
symmetric and an asymmetric random part. They use Langevin dynamics and a solvable spherical model. With asymmetric
bonds there is no energy function, and they find that even a small amount of asymmetry destroys the spin-glass
freezing transition at finite temperature. Slow glassy dynamics survive only over limited timescales.

*Relevance:* the non-reciprocity sweep of §4.1 reproduces the qualitative result that frozen camps melt under
asymmetry, and the paper presents this as known.

## Friedkin & Johnsen (1990), "Social influence and opinions" — `friedkin1990social` — **verify**

Friedkin and Johnsen propose a model of opinion formation in a network. Each person's opinion is updated as a weighted
average of the opinions of those who influence them, combined with that person's initial opinion through a
susceptibility (anchoring) weight. Unlike DeGroot averaging, the model does not in general reach consensus: persistent
disagreement can be an equilibrium when actors stay partly anchored to their initial positions. Later work by the same
authors extended and tested the model with small-group data.

*Relevance:* there are parallels with the averaging (α ≈ 1) and inertia (γ) found in LLM agents. The model is
continuous-opinion, whereas the task here is binary.

## Argyle et al. (2023), "Out of one, many: Using language models to simulate human samples" — `argyle2023out` — **verify**

Argyle and colleagues condition a large language model (GPT-3) on demographic backstories and compare its responses
with those of real survey respondents from the matching subgroups. They call this "silicon sampling". They argue that
the model's outputs show "algorithmic fidelity": they reproduce fine-grained patterns of association between
attitudes and demographics in American political survey data. They propose criteria for when such samples can stand
in for human subjects in social-science research.

*Relevance:* this is the single-agent "LLM as respondent" line of work. Our study asks what happens when such agents
interact.

## Park et al. (2023), "Generative agents: Interactive simulacra of human behavior" — `park2023generative` — **verify**

Park and colleagues build an architecture in which LLM-driven agents have:
- a memory stream of experiences in natural language;
- reflection, which turns memories into higher-level inferences;
- planning.

They run 25 such agents in a sandbox town. They report believable individual and emergent social behaviour, such as
the spread of information and coordinating attendance at a party. Ablations indicate that observation, reflection
and planning each contribute to believability as judged by human raters.

*Relevance:* this is the "populations of LLM agents" line of work. Our agents are deliberately memoryless and the task
is binary, so that the population's behaviour can be compared with a measured kernel.

---

## Also cited in the paper (not part of the requested list)

- **Kline & Santos (2012)**, "A score based approach to wild bootstrap inference" — `kline2012score` — **verify**.
  Used for the H3 score test with wild bootstrap (Amendment 1).
- **Five unrefereed preprints (September 2026)**, author "OpenAI", PDFs in `papers/` — `openai2026rokhlin`,
  `openai2026mezardparisi`, `openai2026nopercolation`, `openai2026cubic`, `openai2026nonuniqueness` — **verify**
  (titles and dates are taken from the PDF title pages; no archival identifier is known). Cite them as unrefereed
  preprints and as motivation only.
