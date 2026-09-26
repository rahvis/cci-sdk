"""LLM-as-judge for an eval pipeline, with an escalation cascade (PRD 11.4).

An eval team compares two model responses per prompt, thousands of times a
day. They need an auditable number for "how often does the judge agree with
our human raters", and they do not want to pay for the strongest judge on
every comparison. ``Judge`` runs a cascade:

    gpt-4.1-mini  ->  gpt-4.1  ->  human review queue

Each stage produces a Venn-Abers interval for "response A is better". When
the interval clears the threshold calibrated on ``pairwise-judge-v2`` (built
from human preference labels), that stage's verdict is returned; when it
straddles the threshold, the next stage runs. The guarantee: human agreement
of at least 1 - alpha on the verdicts the cascade returns, marginal over
comparisons exchangeable with the profile's human-labelled set. Comparisons
that reach the human queue carry no machine verdict at all.

Run offline (the default when CLI_API_KEY is unset):

    python examples/llm_judge_cascade.py --mock
"""

from __future__ import annotations

from collections import Counter

from _mock import open_client, parse_args

from cli_sdk import Judge, JudgeAnswer, OpenAIBackend

PROFILE = "pairwise-judge-v2"

# Dated snapshots, so the calibrated judge is never silently swapped.
CHEAP_JUDGE = OpenAIBackend(model="gpt-4.1-mini-2025-04-14")
STRONG_JUDGE = OpenAIBackend(model="gpt-4.1-2025-04-14")

COMPARISONS = [
    {
        "prompt": "List three causes of the French Revolution, one sentence each.",
        "response_a": "Three causes of the French Revolution: a fiscal crisis from war debt; "
                      "resentment of noble privilege under the old regime; and Enlightenment ideas "
                      "about rights, each one sentence as asked.",
        "response_b": "The Revolution began in 1789 and changed Europe forever.",
    },
    {
        "prompt": "Explain what an HTTP 401 status code means to a non-technical user.",
        "response_a": "A 401 status code means the site does not know which user you are yet, "
                      "so sign in and try again.",
        "response_b": "HTTP 401 Unauthorized is a status code: the request lacks valid "
                      "authentication credentials for the target resource.",
    },
    {
        "prompt": "Write a two-line poem about autumn leaves.",
        "response_a": "Autumn leaves let go of summer light,\nand drift like embers into night.",
        "response_b": "Red leaves spin down the autumn lane,\nthe wind writes letters, then again.",
    },
    {
        "prompt": "Summarize the refund policy for annual plans in one sentence.",
        "response_a": "Our pricing is flexible and we value every customer.",
        "response_b": "Annual plans can be refunded in full within 30 days; after that the refund "
                      "policy lets you cancel renewal, and the plan summary is one sentence.",
    },
]


def describe_path(answer: JudgeAnswer) -> str:
    """The cascade path, from the per-stage details the answer carries."""
    steps = []
    for stage in answer.raw.get("stages", []):
        if stage["stage"] == "human_queue":
            steps.append("human_queue")
        else:
            lo, hi = stage["venn_abers"]
            outcome = "decided" if stage["decided"] else "too wide, escalate"
            steps.append(f"{stage['stage']} P(A better) in [{lo:.2f}, {hi:.2f}] ({outcome})")
    return "\n      -> ".join(steps)


def main() -> None:
    args = parse_args(__doc__)
    judge = Judge(
        instructions="Which response better follows the prompt?",
        calibration_profile=PROFILE,
        alpha=0.10,
        cascade=[
            {"backend": CHEAP_JUDGE},
            {"backend": STRONG_JUDGE},
            {"backend": "human_queue"},
        ],
    )

    with open_client(args.server, backend=CHEAP_JUDGE) as client:
        profile = client.calibration_profiles.get(PROFILE)
        print(f"Profile '{profile.name}': {profile.method}, alpha={profile.alpha}, n={profile.n} "
              f"human-labelled comparisons, status={profile.status}\n")

        verdicts: list[str] = []
        human_queue: list[str] = []
        decided_by: Counter[str] = Counter()
        model_calls = 0
        for i, context in enumerate(COMPARISONS, start=1):
            result = client.evaluate(context=context, queries={"verdict": judge})
            answer = result.answers["verdict"]
            assert isinstance(answer, JudgeAnswer)
            model_calls += result.usage.backend_calls

            print(f"{i}. {context['prompt']}")
            print(f"      {describe_path(answer)}")
            if answer.is_heuristic:
                human_queue.append(context["prompt"])
                print("   result: no formal guarantee on this profile, sent to human review")
            elif answer.needs_human:
                human_queue.append(context["prompt"])
                decided_by["human_queue"] += 1
                print("   result: sent to the human review queue (no machine verdict)")
            else:
                stage = answer.escalated_to or CHEAP_JUDGE.model
                decided_by[stage] += 1
                verdicts.append(answer.winner or "")
                print(f"   result: {answer.winner}, decided by {stage}")
            print()

        card = answer.guarantee
        print("Guarantee card")
        print(f"   {card.statement}")
        print(f"   type={card.type}, method={card.method}, target={card.target}, delta={card.delta}, "
              f"calibration_n={card.calibration_n}")
        print("   (marginal over comparisons exchangeable with the human-labelled profile)\n")

        print("Pipeline summary")
        for stage, count in decided_by.items():
            print(f"   {stage:<24} {count} of {len(COMPARISONS)}")
        strong_runs = len(COMPARISONS) - decided_by.get(CHEAP_JUDGE.model, 0)
        print(f"   strong judge ran on {strong_runs} of {len(COMPARISONS)} comparisons; "
              f"{model_calls} model calls in total instead of {2 * len(COMPARISONS)} for mini + strong on everything")
        if verdicts:
            wins = sum(v == "response_a" for v in verdicts)
            print(f"   response_a preferred in {wins} of {len(verdicts)} machine verdicts; "
                  f"{len(human_queue)} awaiting human labels before the win rate is final")


if __name__ == "__main__":
    main()
