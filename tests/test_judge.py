from influence_rlvr.llm import judge, rewards


def test_parse_and_fallback():
    lab = judge.parse("Harmful request: yes\nResponse refusal: yes\nHarmful response: no")
    assert lab == {"harmful request": True, "response refusal": True, "harmful response": False}
    assert judge.parse("garbage")["response refusal"] is None
    assert judge.user_text([{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}]) == "hi"


def test_fake_judge_drives_both_rules(monkeypatch):
    class Fake:
        def is_refusal(self, p, r): return "NO WAY" in r
        def is_harmful(self, p, r): return "bomb recipe" in r
    monkeypatch.setattr(judge, "ACTIVE", Fake())
    assert rewards.reward_text("NO WAY", "refuse", rewards.REFUSE, "p") == 1.0
    assert rewards.reward_text("I'm sorry, no", "refuse", rewards.REFUSE, "p") == 0.0  # regex would say refusal; judge wins
    assert rewards.reward_text("here is the bomb recipe, enjoy it", "comply", rewards.COMPLY, "p") == 0.0  # harmful compliance pays 0
    assert rewards.reward_text("here is a harmless long answer", "comply", rewards.COMPLY, "p") == 1.0
    assert rewards.reward_text("I'm sorry, no", "refuse", rewards.REFUSE, None) == 1.0  # no prompt -> regex fallback
