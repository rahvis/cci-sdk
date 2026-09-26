"""Bring your own model: a CustomBackend that sends evidence, not the model.

``ToyIntentModel`` is a tiny local bag-of-words classifier standing in for an
in-house model. It declares access level L1 (it can return a probability for
each option) and implements ``score_options`` and ``sample``. The SDK calls
those methods in this process, attaches the resulting numbers to each query
as ``evidence``, and sends the request with backend provider
``client_evidence``. CLI then calibrates that evidence against the named
profile exactly as it would calibrate a hosted model's log-probabilities.

The example prints the request body captured on the wire, so you can see
precisely what leaves your process:

  - sent: the context, each query's instructions and options, the backend's
    name and access level, and the evidence (probabilities or samples);
  - not sent: the model, its weights or vocabulary, and any prompt template
    it uses internally.

It then repeats the call at L0 (sampling only), where the evidence becomes a
list of sampled answers instead of probabilities.

Run offline (the default when CLI_API_KEY is unset):

    python examples/custom_backend_evidence.py --mock
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, ClassVar, Optional

import httpx
from _mock import open_client, parse_args

from cli_sdk import CustomBackend, Gate, Set

PROFILE = "toy-intent-v1"
OPTIONS = {"cancel": "Cancel an order", "track": "Where is my order", "return": "Return an item"}
INTENT_QUESTION = "What does the customer want?"
AUTO_RESOLVE_QUESTION = "Resolve this automatically without an agent?"


class ToyIntentModel(CustomBackend):
    """A local model: keyword weights plus a softmax. Nothing about it is sent."""

    access_level = "L1"
    name = "toy-intent-bow"

    WEIGHTS: ClassVar[dict[str, dict[str, float]]] = {
        "cancel": {"cancel": 2.2, "stop": 1.4, "mistake": 0.9, "don't": 0.6},
        "track": {"where": 1.6, "track": 2.2, "arrive": 1.3, "shipped": 1.2, "late": 0.9},
        "return": {"return": 2.2, "refund": 1.2, "broken": 1.0, "size": 0.9, "wrong": 0.8},
    }

    def _probabilities(self, text: str, options: list[str]) -> dict[str, float]:
        words = text.lower().split()
        logits = {o: 0.2 + sum(w for k, w in self.WEIGHTS.get(o, {}).items()
                               if any(t.startswith(k) for t in words)) for o in options}
        top = max(logits.values())
        exp = {o: math.exp(v - top) for o, v in logits.items()}
        total = sum(exp.values())
        return {o: round(v / total, 4) for o, v in exp.items()}

    def score_options(self, context: Any, instructions: Any, options: dict[str, Optional[str]]) -> dict[str, float]:
        text = context["message"]
        if set(options) == {"true", "false"}:
            # A yes/no question (Gate, Belief): answer with the model's own
            # confidence in its top intent. CLI calibrates this raw number;
            # it is never used as a guarantee by itself.
            confidence = max(self._probabilities(text, list(OPTIONS)).values())
            return {"true": confidence, "false": round(1 - confidence, 4)}
        return self._probabilities(text, list(options))

    def sample(self, context: Any, instructions: Any, n: int) -> list[str]:
        # sample() receives no option list, so answer in the vocabulary the
        # question expects: option keys for the intent question, "true" /
        # "false" for the yes/no auto-resolve question.
        probs = self._probabilities(context["message"], list(OPTIONS))
        rng = random.Random(f"{context['message']}|{instructions}")  # deterministic per question
        if instructions == AUTO_RESOLVE_QUESTION:
            confidence = max(probs.values())
            return rng.choices(["true", "false"], weights=[confidence, 1 - confidence], k=n)
        return rng.choices(list(probs), weights=list(probs.values()), k=n)


class ToyIntentModelL0(ToyIntentModel):
    """The same model exposed at L0: CLI only ever sees sampled answers."""

    access_level = "L0"


def queries() -> dict:
    return {
        "intent": Set(instructions=INTENT_QUESTION, options=OPTIONS,
                      calibration_profile=PROFILE, alpha=0.10),
        "auto_resolve": Gate(instructions=AUTO_RESOLVE_QUESTION,
                             calibration_profile=PROFILE, guarantee="risk", target=0.10),
    }


def main() -> None:
    args = parse_args(__doc__)
    captured: list[dict[str, Any]] = []

    def capture(request: httpx.Request) -> None:
        if request.url.path.endswith("/evaluate"):
            captured.append(json.loads(request.content))

    context = {"message": "Where is my order? It shipped a week ago and still hasn't arrived."}

    print("1. L1 backend: evidence is the model's option probabilities\n")
    with open_client(args.server, backend=ToyIntentModel(), event_hooks={"request": [capture]}) as client:
        result = client.evaluate(context=context, queries=queries())

    body = captured[-1]
    print("   Request body sent to POST /v1/evaluate:")
    print("   " + json.dumps(body, indent=2).replace("\n", "\n   "))
    intent, gate = result.answers["intent"], result.answers["auto_resolve"]
    print(f"\n   set={intent.set}, gate={gate.decision}")
    print(f"   {intent.guarantee.describe()}")
    print(f"   {gate.guarantee.describe()}")
    print(f"   model calls made by CLI: {result.usage.backend_calls} (the evidence was computed locally)")

    print("\n2. L0 backend: the same model, evidence is sampled answers only\n")
    with open_client(args.server, backend=ToyIntentModelL0(), sample_count=8,
                     event_hooks={"request": [capture]}) as client:
        result = client.evaluate(context=context, queries=queries())
    for qid, query in captured[-1]["queries"].items():
        print(f"   {qid}.evidence = {json.dumps(query['evidence'])}")
    print(f"   set={result.answers['intent'].set}, gate={result.answers['auto_resolve'].decision}")
    print("\n   Eight samples carry less information than a probability vector, so the calibrated set is")
    print("   wider. A profile must also be calibrated at the access level it serves: L1 probabilities")
    print("   and L0 sample frequencies are different scores and need separate calibration profiles.")


if __name__ == "__main__":
    main()
