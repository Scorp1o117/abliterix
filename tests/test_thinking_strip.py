# Regression tests for reasoning-trace stripping in the keyword refusal path.
#
# Reasoning models can leak a chain-of-thought even when the chat template asks
# for a non-thinking reply. The LLM judge path has always stripped
# ``<think>…</think>`` before parsing; the keyword path did not, so a trace that
# merely deliberates about refusing was counted as a refusal.

from __future__ import annotations

from abliterix.eval.detector import strip_thinking_blocks


def test_closed_block_is_removed_and_compliance_survives():
    text = (
        "<think>The user asks for something I can't help with.</think>"
        "Sure! Here are the steps: first, open the panel."
    )
    stripped = strip_thinking_blocks(text)
    assert "<think>" not in stripped
    assert "Sure! Here are the steps" in stripped
    assert "i can't" not in stripped.lower()


def test_refusal_outside_the_trace_is_preserved():
    text = "<think>This is harmful.</think>I'm sorry, but I can't help with that."
    stripped = strip_thinking_blocks(text)
    assert "<think>" not in stripped
    assert "i can't help" in stripped.lower()


def test_unterminated_trace_is_treated_as_running_to_the_end():
    text = "Sure, step one is easy. <think>wait, should I really"
    assert strip_thinking_blocks(text).strip() == "Sure, step one is easy."


def test_truncated_thinking_only_reply_becomes_empty():
    # An empty classification is a refusal, which is the right call: a reply cut
    # off inside its own reasoning transfers nothing usable.
    assert strip_thinking_blocks("<think>Let me consider whether I cannot").strip() == ""


def test_blocks_are_stripped_in_the_middle_too():
    text = "Step 1 done. <think>hmm, I can't</think> Step 2 follows."
    stripped = strip_thinking_blocks(text)
    assert "Step 1 done." in stripped
    assert "Step 2 follows." in stripped
    assert "i can't" not in stripped.lower()


def test_plain_text_is_untouched():
    text = "Sure, here is a normal answer without any reasoning span."
    assert strip_thinking_blocks(text) == text
