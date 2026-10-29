"""Contract tests for known defects.

Assertions define the expected behavior. When business code has not yet fixed
a defect, the corresponding test will fail until the fix is in place.
"""
import json
import types
import unittest

try:
    from .support import Event, SCHEMA, defaults, fixture, response
except ImportError:
    from support import Event, SCHEMA, defaults, fixture, response


class DefectContracts(unittest.IsolatedAsyncioTestCase):
    async def test_D1a_custom_low_max_score_must_apply(self):
        cfg = defaults()
        cfg["tier_low"]["max_score"] = 5
        plugin, _, _, _ = fixture(cfg)
        actual = plugin.get_target_config("unknown", 4)
        self.assertEqual(actual[2], "low", "D1: configured low max_score=5 ignored at score=4")

    async def test_D1b_custom_mid_max_score_must_apply(self):
        cfg = defaults()
        cfg["tier_mid"]["max_score"] = 8
        plugin, _, _, _ = fixture(cfg)
        actual = plugin.get_target_config("unknown", 7)
        self.assertEqual(actual[2], "mid", "D1: configured mid max_score=8 ignored at score=7")

    async def test_D2_manual_prompt_valid_categories_must_be_replaced(self):
        cfg = defaults()
        cfg["router_config"]["router_manual_prompt"] = SCHEMA["router_config"]["items"]["router_manual_prompt"]["default"]
        plugin, _, provider, _ = fixture(cfg)
        await plugin.router.analyze_intent("question")
        prompt = provider.text_chat.await_args.kwargs["prompt"]
        self.assertNotIn("{valid_cats_json}", prompt, "D2: schema default manual prompt leaks valid_cats_json placeholder")

    async def test_D3_zero_context_turns_must_exclude_history(self):
        cfg = defaults()
        cfg["router_config"]["context_turns"] = 0
        messages = [types.SimpleNamespace(role="user", content="SECRET_HISTORY_MARKER_OFFLINE_SYNTHETIC")]
        plugin, _, provider, _ = fixture(cfg, messages=messages)
        await plugin.pre_route_message(Event(text="current question"))
        prompt = provider.text_chat.await_args.kwargs["prompt"]
        marker = "SECRET_HISTORY_MARKER_OFFLINE_SYNTHETIC"
        self.assertNotIn(marker, prompt, "D3: context_turns=0 still sends synthetic conversation history")

    async def test_D4_changed_task_snapshots_must_invalidate_cache(self):
        plugin, _, provider, _ = fixture()
        old = [{"id": "A", "category": "code", "score": 8, "summary": "old task"}]
        new = [{"id": "B", "category": "math", "score": 5, "summary": "different task"}]
        provider.text_chat.return_value = types.SimpleNamespace(
            completion_text=json.dumps(response(8, "code", context_relation="continue", continued_task_id="A"))
        )
        first = await plugin.router.analyze_intent("continue", [], old)
        provider.text_chat.return_value = types.SimpleNamespace(
            completion_text=json.dumps(response(5, "math", context_relation="continue", continued_task_id="B"))
        )
        second = await plugin.router.analyze_intent("continue", [], new)
        self.assertEqual(provider.text_chat.await_count, 2, "D4: task_snapshots changed but cached A reused; model not called for B")
        self.assertEqual(second["continued_task_id"], "B")

    async def check_max_chars(self, configured):
        cfg = defaults()
        cfg["router_config"]["context_max_chars"] = configured
        content = "Z" * 350 + "AFTER_350_MARKER"
        plugin, _, provider, _ = fixture(cfg, messages=[types.SimpleNamespace(role="user", content=content)])
        await plugin.pre_route_message(Event(text="current input"))
        prompt = provider.text_chat.await_args.kwargs["prompt"]
        history = prompt.split("Recent Conversation Context:\n", 1)[1].split("\n\nCurrent User Input:", 1)[0]
        sent_content = history.removeprefix("[user]: ")
        self.assertEqual(
            sent_content,
            content,
            f"D5: context_max_chars={configured} should retain {len(content)} chars, got {len(sent_content)}",
        )

    async def test_D5a_max_chars_500_must_not_truncate_at_200(self):
        await self.check_max_chars(500)

    async def test_D5b_max_chars_zero_must_not_truncate_at_200(self):
        await self.check_max_chars(0)
