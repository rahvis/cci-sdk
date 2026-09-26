"""Support-ticket auto-routing with a Set and an FDR-controlled Gate (PRD 11.1).

The ops lead wants tickets routed to billing / technical / sales with no human
in the loop, but only while the fraction of auto-routed tickets that land in
the wrong queue stays at or below 5%. This example walks the whole lifecycle:

  1. Create (or reuse) the calibration profile ``support-routing-v3``,
     with per-account-tier (Mondrian) calibration.
  2. Show that a profile with no examples fails closed: its answers are
     labelled ``heuristic`` and nothing is auto-routed.
  3. Add 1,200 historical, human-labelled tickets and read back the profile's
     size, minimum, recommendation and coverage interval.
  4. Audit the profile on 300 fresh labelled tickets (what ``cli calibration
     audit`` runs in CI to block a deploy).
  5. Evaluate incoming tickets with a ``Set`` (coverage 90%) and a ``Gate``
     (``guarantee="fdr"``, ``target=0.05``) and act on the guarantee.
  6. Create drift monitors and poll them for alerts.

The guarantees are marginal over tickets exchangeable with the calibration
set: "at most 5% of auto-routed tickets are expected to be misrouted", not a
promise about any single ticket.

Run offline (the default when CLI_API_KEY is unset):

    python examples/support_routing_gate.py --mock
    python examples/support_routing_gate.py --mock --simulate-drift

Against the live API this writes synthetic examples into the profile, so point
it at a scratch workspace.
"""

from __future__ import annotations

import hashlib
import random

from _mock import open_client, parse_args

from cli_sdk import (
    AzureOpenAIBackend,
    CalibrationExample,
    EvaluateResponse,
    Gate,
    GateAnswer,
    Set,
    SetAnswer,
)

PROFILE = "support-routing-v3"
TEAMS = {"billing": None, "technical": None, "sales": None}
SET_INSTRUCTIONS = "Which team should handle this ticket?"
GATE_INSTRUCTIONS = "Auto-route this ticket without human review?"

# The deployment behind the profile. Pin the deployment's version-upgrade
# policy to NoAutoUpgrade: a silent model upgrade breaks exchangeability.
BACKEND = AzureOpenAIBackend(model="gpt-4.1", deployment="support-routing-gpt41")

# Hashing the instructions ties the profile to the exact prompt it was
# calibrated with; changing either instruction means recalibrating.
PROMPT_HASH = "sha256:" + hashlib.sha256((SET_INSTRUCTIONS + "\n" + GATE_INSTRUCTIONS).encode()).hexdigest()


class RoutingResponse(EvaluateResponse):
    """Typed access: ``result.department`` and ``result.route``, checked at parse time."""

    department: SetAnswer
    route: GateAnswer


# -- synthetic ticket history (stands in for your labelled ticket export) ----

_TEMPLATES = {
    "billing": [
        "I was charged twice for {month}, please refund one of the payments.",
        "Invoice {num} shows the wrong VAT number for our company.",
        "Why was my card billed after I cancelled the subscription?",
        "Our renewal went through at the old rate, can you issue a partial refund?",
        "I need a receipt for the payment made on {month} {day}.",
        "My payouts have been failing for {day} days and the money has not arrived.",
    ],
    "technical": [
        "The webhook endpoint returns 500 errors since last night.",
        "API requests time out after {day} seconds from our EU servers.",
        "I cannot log in with SSO since this morning.",
        "The CSV export is broken for reports larger than {num} rows.",
        "Our Salesforce integration stopped syncing contacts on {month} {day}.",
        "Hi, I've been trying to connect my Stripe account for {day} days.",
    ],
    "sales": [
        "How much would {num} extra seats cost on the Enterprise plan?",
        "Do you offer nonprofit pricing for annual contracts?",
        "We want to upgrade to the Enterprise plan before {month}.",
        "Can we get a quote for {num} users with a two-year contract?",
        "Is there a discount if we pay annually instead of monthly?",
        "Could someone from your team give us a demo next week?",
    ],
}
# Tickets whose wording points at one team while a human labelled another:
# the realistic cases a calibrated set has to cover.
_CROSSED = [
    ("technical", "The invoice PDF download button is broken on Safari."),
    ("billing", "The upgrade went through but I was billed for the wrong plan."),
    ("sales", "Before we connect our systems, what does the Enterprise plan cost?"),
    ("technical", "Payment page shows an error when customers enter a card."),
    ("billing", "Can you cancel the extra seats and refund the difference?"),
    ("technical", "The refund button just shows a blank page."),
    ("billing", "Why does the API cost so much more this month?"),
    ("technical", "The charge report is empty after the latest update."),
    ("billing", "The API login page crashed while I was updating my card."),
    ("sales", "Our SSO integration is broken, so we are reconsidering the contract."),
]
_MONTHS = ["January", "March", "May", "July", "September", "November"]
_TIERS = ["enterprise"] * 2 + ["pro"] * 5 + ["free"] * 3


def ticket_examples(n: int, seed: int) -> list[CalibrationExample]:
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        if rng.random() < 0.15:
            team, text = rng.choice(_CROSSED)
        else:
            team = rng.choice(list(_TEMPLATES))
            text = rng.choice(_TEMPLATES[team]).format(
                month=rng.choice(_MONTHS), day=rng.randint(2, 28), num=rng.randint(10, 9000))
        context = {"ticket": text, "account_tier": rng.choice(_TIERS)}
        examples.append(CalibrationExample(context=context, label=team, source="human"))
    return examples


# -- the routing decision ----------------------------------------------------


def routing_queries() -> dict:
    return {
        "department": Set(instructions=SET_INSTRUCTIONS, options=TEAMS,
                          calibration_profile=PROFILE, alpha=0.10, method="APS"),
        "route": Gate(instructions=GATE_INSTRUCTIONS, calibration_profile=PROFILE,
                      guarantee="fdr", target=0.05),
    }


def decide(result: RoutingResponse, auto_routing_enabled: bool) -> str:
    """Turn the two answers into one action, failing closed at every step."""
    department, gate = result.department, result.route
    if gate.is_heuristic or department.is_heuristic:
        return f"human triage (no formal guarantee yet), shortlist {department.set}"
    if not auto_routing_enabled:
        return f"human triage (auto-routing paused), shortlist {department.set}"
    if gate.approved:
        return f"auto-route to {department.top}"
    if gate.decision == "abstain":
        return "ask the customer for details (no team is plausible enough)"
    return f"human triage, shortlist {department.set}"


def print_card(label: str, guarantee) -> None:
    print(f"  {label}")
    print(f"    {guarantee.describe()}")
    fields = [("type", guarantee.type), ("method", guarantee.method), ("alpha", guarantee.alpha),
              ("target", guarantee.target), ("realized_upper_bound", guarantee.realized_upper_bound),
              ("calibration_n", guarantee.calibration_n), ("coverage_ci", guarantee.coverage_ci),
              ("last_audited", guarantee.last_audited)]
    print("    " + ", ".join(f"{k}={v}" for k, v in fields if v is not None))


def main() -> None:
    args = parse_args(__doc__, drift_flag=True)
    incoming = [
        {"ticket": "I was charged twice for September and need one refunded.", "account_tier": "pro"},
        {"ticket": "Hi, I've been trying to connect my Stripe account for 3 days.", "account_tier": "enterprise"},
        {"ticket": "How much would 40 more seats cost on the Enterprise plan?", "account_tier": "pro"},
        {"ticket": "My payouts have been failing for 3 days.", "account_tier": "enterprise"},
        {"ticket": "The webhook endpoint returns 500 errors since your deploy last night.", "account_tier": "free"},
        {"ticket": "hello?? anyone there", "account_tier": "free"},
    ]

    with open_client(args.server, backend=BACKEND) as client:
        profiles = client.calibration_profiles

        # 1. Create the profile, or reuse it on a re-run.
        print("1. Calibration profile")
        existing = {p.name: p for p in profiles.list()}
        if PROFILE in existing:
            profile = existing[PROFILE]
            print(f"   reusing '{PROFILE}' (n={profile.n})")
        else:
            profile = profiles.create(name=PROFILE, backend=BACKEND, method="APS", alpha=0.10,
                                      group_by="account_tier", prompt_template_hash=PROMPT_HASH)
            print(f"   created '{PROFILE}': method={profile.method}, alpha={profile.alpha}, "
                  f"group_by={profile.group_by}, fingerprint={profile.backend_fingerprint}")

            # 2. An empty profile cannot back a guarantee: CLI answers, but labels it heuristic.
            print("\n2. Evaluating before any examples exist")
            early = client.evaluate(context=incoming[0], queries=routing_queries(),
                                    response_model=RoutingResponse)
            print(f"   guarantee type: {early.route.guarantee.type} ({early.route.guarantee.statement})")
            print(f"   action: {decide(early, auto_routing_enabled=True)}")

        # 3. Add labelled history until the profile reaches the recommended size.
        print("\n3. Adding labelled examples")
        if profile.n < (profile.recommended_n or 1000):
            history = ticket_examples(1200, seed=3)
            for start in range(0, len(history), 400):
                profile = profiles.add_examples(PROFILE, history[start:start + 400])
                print(f"   +{len(history[start:start + 400])} examples -> n={profile.n}")
        profile = profiles.get(PROFILE)
        ci = profile.realized_coverage_ci
        print(f"   n={profile.n} (hard minimum {profile.minimum_n}, recommended {profile.recommended_n}), "
              f"status={profile.status}, serving guarantees: {profile.can_serve_guarantees}")
        if ci:
            print(f"   realized-coverage interval implied by n: [{ci[0]:.3f}, {ci[1]:.3f}]")
        for group, status in profile.groups.items():
            print(f"   group {group:<10} n={status.n:<4} {status.status}")

        # 4. Audit on fresh labelled tickets before trusting it in production.
        print("\n4. Audit on 300 fresh labelled tickets")
        audit = profiles.audit(PROFILE, fresh_examples=ticket_examples(300, seed=11))
        print(f"   result={audit.result}, realized coverage {audit.realized_coverage:.3f} "
              f"(95% CI [{audit.ci_lower:.3f}, {audit.ci_upper:.3f}]) vs target {audit.target:.2f}")
        auto_routing_enabled = audit.passed

        # 5. Route incoming tickets and act on the guarantee.
        print("\n5. Routing incoming tickets")
        results = []
        for context in incoming:
            result = client.evaluate(context=context, queries=routing_queries(), response_model=RoutingResponse)
            results.append(result)
            print(f"   {context['ticket'][:58]:<58}")
            print(f"     set={result.department.set} gate={result.route.decision}"
                  f" -> {decide(result, auto_routing_enabled)}")

        auto = sum(r.route.approved for r in results)
        print(f"\n   auto-routed {auto} of {len(results)}; the rest went to people with a calibrated shortlist")
        print("\n   Guarantee cards (from the last request):")
        print_card("department (Set)", results[-1].department.guarantee)
        print_card("route (Gate)", results[-1].route.guarantee)

        # 6. Watch the guarantee in production with anytime-valid monitors.
        print("\n6. Drift monitors")
        monitors = profiles.monitors
        monitors.create(PROFILE, type="coverage", target=0.90, false_alarm_rate=0.05, labelled_sample_rate=0.02)
        monitors.create(PROFILE, type="risk", target=0.05, false_alarm_rate=0.05, labelled_sample_rate=0.02)
        monitors.create(PROFILE, type="fingerprint")
        for monitor in monitors.list(PROFILE):
            target = f", target={monitor.target}" if monitor.target is not None else ""
            print(f"   {monitor.id}: {monitor.type}{target}, status={monitor.status}")
        alerts = list(monitors.poll(PROFILE))
        if not alerts:
            print("   no alerts: the running e-processes show no evidence against the guarantee")
        for alert in alerts:
            print(f"   ALERT [{alert.severity}] {alert.type}: {alert.message}")
            if alert.type in ("coverage", "risk", "fingerprint"):
                auto_routing_enabled = False
        if not auto_routing_enabled:
            print("   auto-routing paused: every ticket now goes to human triage until the profile is recalibrated")


if __name__ == "__main__":
    main()
