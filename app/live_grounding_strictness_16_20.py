from app.services.response_grounding_validator import (
    ResponseGroundingValidator,
)


KNOWLEDGE = """
The three terminals of the BJT are called the Base (B),
the Collector (C) and the Emitter (E).

The Base-Emitter (B-E) junction is forward biased and
the Base-Collector (B-C) junction is reverse biased.

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
    (
        "S1",
        "Source-level relationship",
        (
            "The collector current is related to the "
            "emitter current."
        ),
        {"supported"},
    ),
    (
        "S2",
        "Conservative electron injection",
        (
            "Electrons are injected from the emitter "
            "region into the base region."
        ),
        {"supported"},
    ),
    (
        "P1",
        "Base-current proportionality",
        (
            "The collector current is proportional to "
            "the base current."
        ),
        {
            "partially_supported",
            "unsupported",
        },
    ),
    (
        "P2",
        "Collector doping comparison",
        (
            "The base is less heavily doped than both "
            "the emitter and the collector."
        ),
        {
            "partially_supported",
            "unsupported",
        },
    ),
    (
        "P3",
        "Majority carrier qualifier",
        (
            "Most electrons injected into the base "
            "continue into the collector."
        ),
        {
            "partially_supported",
            "unsupported",
        },
    ),
    (
        "P4",
        "Exact voltage threshold",
        (
            "The transistor requires exactly 0.7 V "
            "between base and emitter to operate."
        ),
        {
            "partially_supported",
            "unsupported",
        },
    ),
    (
        "P5",
        "Unsupported base-current control",
        (
            "The collector current is controlled by "
            "the base current."
        ),
        {
            "partially_supported",
            "unsupported",
        },
    ),
]


def main():
    validator = ResponseGroundingValidator()

    passed = 0
    failed = 0

    print()
    print(
        "Step 16.20.2C Strict Grounding "
        "Adversarial Matrix"
    )
    print()

    for (
        case_id,
        label,
        response,
        allowed,
    ) in CASES:

        result = validator.validate(
            response=response,
            knowledge_context=KNOWLEDGE,
            task_name="response_validator",
        )

        ok = result.status in allowed

        if ok:
            passed += 1
        else:
            failed += 1

        print("=" * 70)
        print(
            case_id,
            label,
        )
        print("=" * 70)

        print(
            "RESPONSE   =",
            response,
        )

        print(
            "STATUS     =",
            result.status,
        )

        print(
            "CONFIDENCE =",
            result.confidence,
        )

        print(
            "REASON     =",
            result.reason,
        )

        print(
            "ISSUES     =",
            result.issues,
        )

        print(
            "EXPECTED   =",
            sorted(allowed),
        )

        print(
            "RESULT     =",
            "PASS" if ok else "FAIL",
        )

        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        "Passed:",
        passed,
    )

    print(
        "Failed:",
        failed,
    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
