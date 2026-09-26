"""Model-cost routing with a budget guarantee (PRD 11.5).

A team serves most traffic from a self-hosted open-weight model on vLLM and
pays for a frontier API call only when the cheap tier's answer is genuinely
uncertain. ``Route`` runs the cascade with a calibrated, per-query escalation
rule learned on ``cost-routing-v1`` (historical queries with both tiers'
outputs, costs and correctness recorded), and states a cost guarantee:

    P(cost per request <= 0.4 cents) >= 1 - alpha = 90%

over requests exchangeable with the profile's historical sample. It bounds
the long-run fraction of over-budget requests; it does not promise that any
particular request, or any short run of requests, stays under budget.

The request has no top-level backend: a Route query declares its backends in
``cascade``.

Run offline (the default when CLI_API_KEY is unset):

    python examples/model_cascade_routing.py --mock
"""

from __future__ import annotations

from _mock import open_client, parse_args

from cli_sdk import OpenAIBackend, Route, RouteAnswer, VLLMBackend

PROFILE = "cost-routing-v1"
BUDGET_CENTS = 0.4

# engine_version and quantization are part of the backend fingerprint:
# change either and the profile must be recalibrated.
CHEAP_TIER = VLLMBackend(model="llama-3.3-70b", base_url="http://internal-vllm:8000",
                         engine_version="0.11.0", quantization="fp8")
STRONG_TIER = OpenAIBackend(model="gpt-4.1-2025-04-14")

QUERIES = [
    "What are your support hours?",
    "How do I reset my password?",
    "Translate 'order shipped' into German.",
    "Summarize this ticket: customer cannot find the invoice download link.",
    "Compare the tax treatment of annual versus monthly plans for EU customers, step by step.",
    "What is the capital of Australia?",
    "Draft a two-sentence apology for a delayed delivery.",
    "Does the contract allow us to terminate early, and what are the legal trade-offs?",
    "List three ways to speed up a slow SQL query.",
    "Convert 72 degrees Fahrenheit to Celsius.",
]


def main() -> None:
    args = parse_args(__doc__)
    route = Route(
        cascade=[{"backend": CHEAP_TIER}, {"backend": STRONG_TIER}],
        calibration_profile=PROFILE,
        guarantee="cost_budget",
        target_cents=BUDGET_CENTS,
        alpha=0.10,
    )

    with open_client(args.server) as client:
        profile = client.calibration_profiles.get(PROFILE)
        print(f"Profile '{profile.name}': {profile.method}, alpha={profile.alpha}, n={profile.n} "
              f"historical queries, fingerprint={profile.backend_fingerprint}\n")

        print(f"{'query':<60} {'served by':<22} {'cost':>6}")
        answers: list[RouteAnswer] = []
        for text in QUERIES:
            result = client.evaluate(context={"query": text}, queries={"answer": route})
            answer = result.answers["answer"]
            assert isinstance(answer, RouteAnswer)
            answers.append(answer)
            if answer.is_heuristic:
                # No formal cost guarantee: fall back to the conservative policy
                # (here, always the strong tier) and record it.
                print(f"{text[:58]:<60} heuristic answer, using strong tier")
                continue
            served = answer.served_by.get("model", "?")
            flag = " escalated" if answer.escalated else ""
            print(f"{text[:58]:<60} {served:<22} {answer.cost_cents:>5.2f}c{flag}")

        card = answers[-1].guarantee
        costs = [a.cost_cents or 0.0 for a in answers]
        on_cheap = sum(not a.escalated for a in answers)
        over = sum(c > BUDGET_CENTS for c in costs)
        print(f"\nGuarantee: {card.describe()}")
        print(f"   type={card.type}, method={card.method}, target_cents={card.target_cents}, "
              f"alpha={card.alpha}, calibration_n={card.calibration_n}")
        print(f"\nThis batch: {on_cheap} of {len(answers)} served by the self-hosted tier, "
              f"mean cost {sum(costs) / len(costs):.3f}c, max {max(costs):.3f}c")
        print(f"   over budget: {over} of {len(answers)}. The guarantee bounds the long-run over-budget rate at "
              f"{card.alpha:.0%}; a 10-request sample can sit above or below that by chance.")


if __name__ == "__main__":
    main()
