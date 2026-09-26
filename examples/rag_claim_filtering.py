"""RAG answer factuality with a claim-level guarantee on Claude (PRD 11.2).

A support copilot drafts long-form answers from retrieved documentation.
Product's requirement: at most 5% of shipped answers may contain a claim the
documentation does not back. ``Claim`` decomposes each draft into atomic
claims, scores each claim's support, and keeps only the subset that clears a
threshold calibrated on ``rag-factuality-v1``, so that, over answers
exchangeable with that profile's annotated examples, the probability that
every retained claim is supported is at least 1 - alpha = 95%.

Anthropic's Messages API returns text only (access level L0), so support is
scored from resampled generations and their agreement. That costs extra
backend calls, reported in ``usage.backend_calls``, but the guarantee has the
same shape as on a backend that exposes log-probabilities.

Acting on the answer:
  - ship only the retained claims, never the raw draft;
  - if the filter removed most of the draft, hand the question to a person
    rather than show a fragment;
  - if the answer carries no formal guarantee, ship nothing automatically.

Run offline (the default when CLI_API_KEY is unset):

    python examples/rag_claim_filtering.py --mock
"""

from __future__ import annotations

from _mock import open_client, parse_args

from cli_sdk import AnthropicBackend, Claim, ClaimAnswer

PROFILE = "rag-factuality-v1"

# effort is part of the scoring function: keep it identical between
# calibration and serving. sample_count is the number of resamples per query.
BACKEND = AnthropicBackend(model="claude-sonnet-5", effort="low", sample_count=10)

DOCS = [
    "Annual plans can be refunded in full within 30 days of purchase.",
    "After 30 days, annual plans are not refundable, but you can cancel renewal at any time.",
    "Refunds are issued to the original payment method.",
    "Monthly plans are not refundable; cancelling stops the next charge.",
]

QUESTIONS = [
    {
        "question": "What's your refund policy for annual plans?",
        "retrieved_docs": DOCS,
        "draft_answer": (
            "Annual plans can be refunded in full within 30 days of purchase. "
            "Refunds are issued to the original payment method. "
            "Refunds typically arrive within 2 business days. "
            "After 30 days you can still cancel renewal at any time. "
            "You can also pause an annual plan for up to 3 months instead."
        ),
    },
    {
        "question": "Can I move my annual plan to a different workspace?",
        "retrieved_docs": DOCS,
        "draft_answer": (
            "Yes, workspace owners can transfer a plan from the billing settings page. "
            "Transfers take effect at the start of the next billing cycle. "
            "Annual plans can be refunded in full within 30 days of purchase."
        ),
    },
]

MIN_RETENTION = 0.5  # below this, the filtered answer is too thin to show on its own


def act_on(answer: ClaimAnswer) -> str:
    if answer.is_heuristic:
        return "HAND OFF: no formal guarantee on this profile; route to a support agent"
    if not answer.retained_claims or answer.retention_rate < MIN_RETENTION:
        return "HAND OFF: most of the draft was unsupported; route to a support agent with the retained notes"
    return "SHIP the filtered answer"


def main() -> None:
    args = parse_args(__doc__)
    with open_client(args.server, backend=BACKEND) as client:
        profile = client.calibration_profiles.get(PROFILE)
        print(f"Profile '{profile.name}': method={profile.method}, alpha={profile.alpha}, n={profile.n}, "
              f"status={profile.status}, last audit={profile.last_audit}")
        print("Claim-level profiles are small (50 to 200 fully annotated answers) because every claim"
              " in every example is labelled by hand.\n")

        for context in QUESTIONS:
            result = client.evaluate(
                context=context,
                queries={
                    "filtered_answer": Claim(
                        instructions="Filter the draft answer to only fully-supported claims.",
                        calibration_profile=PROFILE,
                        alpha=0.05,
                    ),
                },
            )
            answer = result.answers["filtered_answer"]
            assert isinstance(answer, ClaimAnswer)

            print(f"Q: {context['question']}")
            print(f"   retained {len(answer.retained_claims)} of "
                  f"{len(answer.retained_claims) + len(answer.dropped_claims)} claims "
                  f"(retention {answer.retention_rate:.0%})")
            for claim in answer.retained_claims:
                print(f"     keep  {claim}")
            for dropped in answer.dropped_claims:
                score = f" (support {dropped.score:.2f})" if dropped.score is not None else ""
                print(f"     drop  {dropped.text}  [{dropped.reason}{score}]")
            print(f"   guarantee: {answer.guarantee.statement or answer.guarantee.describe()}")
            print(f"              method={answer.guarantee.method}, target={answer.guarantee.target}, "
                  f"calibration_n={answer.guarantee.calibration_n}")
            print(f"   backend calls: {result.usage.backend_calls} "
                  f"(L0 backend: support is scored by resampling)")
            decision = act_on(answer)
            print(f"   -> {decision}")
            if decision.startswith("SHIP"):
                print(f"      \"{answer.as_text()}\"")
            print()


if __name__ == "__main__":
    main()
