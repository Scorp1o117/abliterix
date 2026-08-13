from types import SimpleNamespace

from abliterix.eval.screening import TrialScreener


def _screener():
    engine = SimpleNamespace(tokenizer=SimpleNamespace())
    return TrialScreener(SimpleNamespace(), engine)


def test_thinking_leak_ignores_normal_prose():
    screener = _screener()
    responses = [
        "Take time for critical thinking before deciding.",
        "This thought experiment illustrates the tradeoff.",
        "这个故事给读者留下了很大的思考空间。",
    ]
    assert not screener.check_thinking_leak(responses)


def test_thinking_leak_detects_tags_and_section_headers():
    screener = _screener()
    assert screener.check_thinking_leak(["<think>hidden reasoning</think>Answer"])
    assert screener.check_thinking_leak(["Thinking: first derive the answer"])
    assert screener.check_thinking_leak(["思考：先分析问题\n回答：..."])


def test_thinking_leak_extra_patterns_remain_substring_matches():
    screener = _screener()
    assert screener.check_thinking_leak(
        ["internal scratchpad follows"], extra_patterns=["scratchpad"]
    )
