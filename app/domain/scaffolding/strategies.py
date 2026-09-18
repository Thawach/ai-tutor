from dataclasses import dataclass


@dataclass(frozen=True)
class ScaffoldingStrategy:
    """
    นิยามกลยุทธ์ scaffolding หนึ่งรูปแบบ
    """

    level: int
    name: str
    description: str
    instruction: str


GUIDING_QUESTION = ScaffoldingStrategy(
    level=1,
    name="guiding_question",
    description="ใช้คำถามนำเพื่อกระตุ้นการคิดและความรู้เดิม",
    instruction="""
Use a guiding-question strategy.

Your goal is to activate the learner's prior knowledge
and encourage the learner to think independently.

Rules:
- Ask exactly ONE main guiding question.
- Do not reveal the answer.
- Do not provide a direct hint yet.
- Do not explain the complete concept.
- Connect the question to the learner's current topic.
- Keep the response concise.
- Wait for the learner to respond.
"""
)


HINT = ScaffoldingStrategy(
    level=2,
    name="hint",
    description="ให้คำใบ้เพียงเล็กน้อยโดยไม่เปิดเผยคำตอบทั้งหมด",
    instruction="""
Use a hint strategy.

Your goal is to move the learner exactly ONE small step closer
to the answer while leaving meaningful reasoning for the learner.

Rules:

- Give exactly ONE small clue or attention-directing cue.
- Reveal only ONE useful piece of information.
- Do not reveal the complete answer.
- Do not state the complete structure, sequence, classification,
  formula, mapping, or solution.
- Do not fill in all missing information for the learner.
- Do not give the answer inside the hint and then ask the learner
  to repeat, identify, or confirm that same answer.
- The learner must still need to reason from the clue.
- Use the learner's previous response as the starting point.
- After the clue, ask exactly ONE short follow-up question.
- Ask only about the next small reasoning step.
- Do not ask a compound question.
- Do not provide a full explanation.
- Keep the response concise.

LANGUAGE RULE:

- Do not switch languages for labels such as "Hint:".
- If the learner writes in Thai, the entire response should be in Thai,
  except for necessary technical terms such as NPN, PNP, Base,
  Emitter, Collector, n-type, and p-type.

TOO MUCH HELP:

"NPN has an n-type emitter, a p-type base, and an n-type collector.
Which layer is the base?"

This is NOT an appropriate hint because the answer is already revealed.

APPROPRIATE HINT:

"ลองสังเกตชั้นที่อยู่ตรงกลางของโครงสร้าง NPN
คุณคิดว่าชั้นนั้นควรเป็นชนิดใด?"

This is appropriate because it directs attention to only one feature
and still requires the learner to reason toward the answer.
"""
)


CONCEPTUAL_SUPPORT = ScaffoldingStrategy(
    level=3,
    name="conceptual_support",
    description="อธิบายเฉพาะแนวคิดสำคัญที่ผู้เรียนยังขาด",
    instruction="""
Use conceptual support.

Your goal is to explain only the key concept
that the learner appears to be missing.

Rules:
- Explain the missing concept briefly and accurately.
- Do not explain the entire topic.
- Connect the explanation to the learner's previous response.
- Correct misconceptions when necessary.
- Do not immediately solve the entire original problem.
- After the explanation, ask the learner to try again.
- Ask exactly ONE main follow-up question.
"""
)


EXAMPLE = ScaffoldingStrategy(
    level=4,
    name="example",
    description="ใช้ตัวอย่างหรือกรณีใกล้เคียงเพื่อช่วยให้ผู้เรียนเห็นแนวคิด",
    instruction="""
Use an example or modeling strategy.

Your goal is to demonstrate the concept using
ONE similar example without directly solving
the learner's original problem.

Rules:
- Provide only one relevant example.
- Make the connection between the example and the concept clear.
- Do not simply give the final answer to the original question.
- After the example, ask the learner to apply the idea
  to the original problem.
- Ask exactly ONE main follow-up question.
"""
)


DIRECT_EXPLANATION = ScaffoldingStrategy(
    level=5,
    name="direct_explanation",
    description="ให้คำอธิบายที่ถูกต้องและครบเมื่อผู้เรียนยังติดขัด",
    instruction="""
Use direct explanation.

The learner has already received substantial support
and still needs explicit help.

Rules:
- Give the correct explanation clearly.
- Explain the reasoning, not only the final answer.
- Use the verified learning content when available.
- Avoid unnecessary detail.
- After explaining, ask exactly ONE reflection question
  to check whether the learner understands the reasoning.
"""
)


STRATEGIES_BY_LEVEL = {
    1: GUIDING_QUESTION,
    2: HINT,
    3: CONCEPTUAL_SUPPORT,
    4: EXAMPLE,
    5: DIRECT_EXPLANATION,
}


def get_strategy(level: int) -> ScaffoldingStrategy:
    """
    คืน ScaffoldingStrategy ตามระดับ 1-5
    """

    level = max(1, min(level, 5))

    return STRATEGIES_BY_LEVEL[level]