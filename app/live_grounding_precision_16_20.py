from app.ai import (
    reset_ai_usage,
    get_ai_usage_summary,
)

from app.services.response_grounding_validator import (
    ResponseGroundingValidator,
)


KNOWLEDGE = """
The three terminals of the BJT are called the Base (B),
the Collector (C) and the Emitter (E).

With the voltage V_BE and V_CB as shown, the
Base-Emitter (B-E) junction is forward biased and the
Base-Collector (B-C) junction is reverse biased.

Due to the large differences in the doping concentrations
of the emitter and the base regions, electrons are injected
into the base region from the emitter region.

The number of electrons injected into the collector region
is directly related to the electrons injected into the base
region from the emitter region.

Therefore, the collector current is related to the emitter
current which is in turn a function of the B-E voltage.

The voltage between two terminals controls the current
through the third terminal.
"""


CASES = [
    {
        "id": "S1",
        "label": "Explicit terminal claim",
        "response": (
            "The BJT has three terminals: Base, Collector, "
            "and Emitter."
        ),
        "allowed": {"supported"},
    },
    {
        "id": "S2",
        "label": "Explicit junction bias claim",
        "response": (
            "In the shown NPN bias condition, the B-E "
            "junction is forward biased and the B-C "
            "junction is reverse biased."
        ),
        "allowed": {"supported"},
    },
    {
        "id": "S3",
        "label": "Explicit current relationship",
        "response": (
            "The collector current is related to the "
            "emitter current, which is a function of "
            "the B-E voltage."
        ),
        "allowed": {"supported"},
    },
    {
        "id": "U1",
        "label": "Unsupported beta claim",
        "response": (
            "The base current is amplified by beta to "
            "become the collector current."
        ),
        "allowed": {
            "partially_supported",
            "unsupported",
        },
    },
    {
        "id": "U2",
        "label": "Unsupported exact VBE claim",
        "response": (
            "The B-E junction becomes forward biased "
            "only when V_BE is exactly 0.7 V."
        ),
        "allowed": {
            "partially_supported",
            "unsupported",
        },
    },
    {
        "id": "U3",
        "label": "Unsupported doping claim",
        "response": (
            "The base region is heavily doped."
        ),
        "allowed": {
            "partially_supported",
            "unsupported",
        },
    },
    {
        "id": "C1",
        "label": "Contradiction",
        "response": (
            "In the shown NPN bias condition, the B-E "
            "junction is reverse biased and the B-C "
            "junction is forward biased."
        ),
        "allowed": {"unsupported"},
    },
    {
        "id": "Q1",
        "label": "Question-only claim behavior",
        "response": (
            "What exact V_BE value is required to turn "
            "the transistor on?"
        ),
        "allowed": {"supported"},
    },
]


def main():
    validator = ResponseGroundingValidator()

    passed = 0
    failed = 0

    reset_ai_usage()

    print()
    print(
        "Step 16.20.2A Semantic Claim-Source "
        "Precision Matrix"
    )
    print()

    for case in CASES:

        result = validator.validate(
            response=case["response"],
            knowledge_context=KNOWLEDGE,
            task_name="response_validator",
        )

        ok = result.status in case["allowed"]

        if ok:
            passed += 1
        else:
            failed += 1

        print("=" * 70)
        print(
            case["id"],
            case["label"],
        )
        print("=" * 70)

        print("RESPONSE:")
        print(case["response"])
        print()

        print("STATUS     =", result.status)
        print("CONFIDENCE =", result.confidence)
        print("REASON     =", result.reason)
        print("ISSUES     =", result.issues)
        print("EXPECTED   =", sorted(case["allowed"]))
        print(
            "RESULT     =",
            "PASS" if ok else "FAIL",
        )
        print()

    usage = get_ai_usage_summary()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("Passed:", passed)
    print("Failed:", failed)
    print(
        "LLM Calls:",
        usage.get("calls"),
    )

    print(
        "Tasks:",
        [
            record.get("component")
            for record
            in usage.get("records", [])
        ],
    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()