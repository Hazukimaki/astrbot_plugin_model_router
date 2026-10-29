"""Regression contract tests for v0.6.1 fixes and new requirements.

Covers:
- User real thresholds low=2, mid=7 (1,2->low; 3,6,7->mid; 8,9->high)
- Default thresholds low=3, mid=7 (1..3->low; 4..7->mid; 8,9->high)
- Context turns = 4 sending 8 messages when conversation > 6 messages
- Context turns = 0 skipping conversation history read & omitting old snapshots
- Context max chars = 500, 0, 50 applying to both dict and str histories
- All effective contexts (not just last two) participating in cache key
- Cache invalidation on task_snapshots content/id changes and router provider config changes
- Conservative fallback on invalid JSON and provider exceptions
"""
import copy
import json
import types
import unittest

try:
    from .support import Event, defaults, fixture, response
except ImportError:
    from support import Event, defaults, fixture, response


class ThresholdContractTests(unittest.TestCase):
    def test_user_real_thresholds_low2_mid7(self):
        """User real threshold requirement: low=2, mid=7."""
        cfg = defaults()
        cfg["tier_low"]["max_score"] = 2
        cfg["tier_mid"]["max_score"] = 7
        plugin, _, _, _ = fixture(cfg)

        test_cases = [
            (1, "low"),
            (2, "low"),
            (3, "mid"),
            (6, "mid"),
            (7, "mid"),
            (8, "high"),
            (9, "high"),
        ]
        for score, expected_tier in test_cases:
            with self.subTest(score=score, expected_tier=expected_tier):
                provider, tier = plugin.get_target_config("chat", score)
                self.assertEqual(
                    tier,
                    expected_tier,
                    f"Score {score} with low=2, mid=7 should route to {expected_tier}, got {tier}",
                )
                self.assertEqual(provider, f"{expected_tier}-provider")

    def test_default_thresholds_low3_mid7(self):
        """Default threshold requirement: low=3, mid=7 across full 1-9 scale."""
        plugin, _, _, _ = fixture()

        test_cases = [
            (1, "low"),
            (2, "low"),
            (3, "low"),
            (4, "mid"),
            (5, "mid"),
            (6, "mid"),
            (7, "mid"),
            (8, "high"),
            (9, "high"),
        ]
        for score, expected_tier in test_cases:
            with self.subTest(score=score, expected_tier=expected_tier):
                provider, tier = plugin.get_target_config("unknown", score)
                self.assertEqual(
                    tier,
                    expected_tier,
                    f"Score {score} with default low=3, mid=7 should route to {expected_tier}, got {tier}",
                )
                self.assertEqual(provider, f"{expected_tier}-provider")


class ContextHistoryContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_turns_four_sends_eight_messages(self):
        """When conversation has > 6 messages, context_turns=4 must send 8 messages (4 turns)."""
        cfg = defaults()
        cfg["router_config"]["context_turns"] = 4
        # Seed 10 historical messages
        messages = [
            types.SimpleNamespace(
                role="user" if i % 2 == 0 else "assistant",
                content=f"HIST_MSG_INDEX_{i}",
            )
            for i in range(10)
        ]
        plugin, _, provider, _ = fixture(cfg, messages=messages)
        await plugin.pre_route_message(Event(text="current prompt"))
        provider.text_chat.assert_awaited_once()
        prompt = provider.text_chat.await_args.kwargs["prompt"]

        # Oldest 2 messages must NOT be in prompt
        self.assertNotIn("HIST_MSG_INDEX_0", prompt)
        self.assertNotIn("HIST_MSG_INDEX_1", prompt)
        # Sliced 8 recent messages must all be in prompt
        for i in range(2, 10):
            self.assertIn(
                f"HIST_MSG_INDEX_{i}",
                prompt,
                f"HIST_MSG_INDEX_{i} should be included in prompt when context_turns=4",
            )

    async def test_zero_context_turns_skips_history_read_and_omits_snapshots(self):
        """context_turns=0 must completely skip reading conversation history and omit old task snapshots."""
        cfg = defaults()
        cfg["router_config"]["context_turns"] = 0
        messages = [types.SimpleNamespace(role="user", content="LEAKED_HISTORY_MARKER")]
        plugin, context, provider, _ = fixture(cfg, messages=messages)

        sid = "test:private:1"
        plugin.task_snapshots[sid] = {
            "old_task_seeded": {
                "score": 8,
                "category": "code",
                "summary": "Previous coding session",
                "turn_count": 0,
            }
        }

        await plugin.pre_route_message(Event(text="current input", sid=sid))

        # 1. Conversation manager history should not even be looked up
        context.conversation_manager.get_curr_conversation_id.assert_not_called()

        # 2. History must not leak into model prompt
        provider.text_chat.assert_awaited_once()
        prompt = provider.text_chat.await_args.kwargs["prompt"]
        self.assertNotIn("LEAKED_HISTORY_MARKER", prompt)
        self.assertNotIn("Recent Conversation Context:", prompt)

        # 3. Old task snapshots must not be sent
        self.assertNotIn("old_task_seeded", prompt)
        self.assertNotIn("Previous coding session", prompt)

    async def test_context_max_chars_effective_for_dict_and_str(self):
        """context_max_chars=500, 0, and 50 must apply correctly to both dict and str contexts."""
        cases = [
            (500, "X" * 300 + "_TAIL_500", True),     # 309 chars, retained (< 500)
            (0, "Y" * 300 + "_TAIL_UNLIMITED", True), # 315 chars, retained (0 = no limit)
            (50, "Z" * 50 + "_TAIL_SHOULD_CUT", False), # > 50 chars, tail must be cut
        ]

        for limit, content, should_contain_tail in cases:
            tail = content.split("_TAIL_")[1]
            tail_marker = "_TAIL_" + tail

            # Test dict format: {"role": "user", "content": content}
            with self.subTest(limit=limit, format="dict"):
                cfg = defaults()
                cfg["router_config"]["context_max_chars"] = limit
                plugin, _, provider, _ = fixture(cfg)
                await plugin.router.analyze_intent("q", contexts=[{"role": "user", "content": content}])
                prompt = provider.text_chat.await_args.kwargs["prompt"]
                if should_contain_tail:
                    self.assertIn(tail_marker, prompt, f"Limit {limit} dict should retain tail")
                else:
                    self.assertNotIn(tail_marker, prompt, f"Limit {limit} dict should truncate tail")

            # Test str format: content directly
            with self.subTest(limit=limit, format="str"):
                cfg = defaults()
                cfg["router_config"]["context_max_chars"] = limit
                plugin, _, provider, _ = fixture(cfg)
                await plugin.router.analyze_intent("q", contexts=[content])
                prompt = provider.text_chat.await_args.kwargs["prompt"]
                if should_contain_tail:
                    self.assertIn(tail_marker, prompt, f"Limit {limit} str should retain tail")
                else:
                    self.assertNotIn(tail_marker, prompt, f"Limit {limit} str should truncate tail")


class CacheContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_effective_contexts_participate_in_cache(self):
        """All effective history messages (not just the last two) must participate in cache key."""
        plugin, _, provider, _ = fixture()
        ctx1 = [
            {"role": "user", "content": "HISTORY_VARIANT_A"},
            {"role": "assistant", "content": "SHARED_2"},
            {"role": "user", "content": "SHARED_3"},
            {"role": "assistant", "content": "SHARED_4"},
        ]
        ctx2 = [
            {"role": "user", "content": "HISTORY_VARIANT_B"},
            {"role": "assistant", "content": "SHARED_2"},
            {"role": "user", "content": "SHARED_3"},
            {"role": "assistant", "content": "SHARED_4"},
        ]

        key1 = plugin.router._get_cache_key("same query", ctx1)
        key2 = plugin.router._get_cache_key("same query", ctx2)
        self.assertNotEqual(
            key1,
            key2,
            "Cache key must differ when an earlier (non-last-two) context message differs",
        )

        await plugin.router.analyze_intent("same query", ctx1)
        await plugin.router.analyze_intent("same query", ctx2)
        self.assertEqual(
            provider.text_chat.await_count,
            2,
            "Calling analyze_intent with different earlier history must not hit cache",
        )

    async def test_cache_invalidated_on_task_snapshots_id_or_content_change(self):
        """Changing task_snapshots ID or content must invalidate cache / alter cache key."""
        plugin, _, provider, _ = fixture()
        base = [{"id": "TASK_1", "category": "code", "score": 8, "summary": "initial summary"}]
        diff_id = [{"id": "TASK_2", "category": "code", "score": 8, "summary": "initial summary"}]
        diff_content = [{"id": "TASK_1", "category": "math", "score": 4, "summary": "changed summary"}]

        key_base = plugin.router._get_cache_key("query", [], base)
        key_diff_id = plugin.router._get_cache_key("query", [], diff_id)
        key_diff_content = plugin.router._get_cache_key("query", [], diff_content)

        self.assertNotEqual(key_base, key_diff_id, "Cache key must differ when snapshot ID changes")
        self.assertNotEqual(key_base, key_diff_content, "Cache key must differ when snapshot content changes")

        await plugin.router.analyze_intent("query", [], base)
        await plugin.router.analyze_intent("query", [], diff_id)
        await plugin.router.analyze_intent("query", [], diff_content)
        self.assertEqual(
            provider.text_chat.await_count,
            3,
            "Each snapshot change must result in a fresh model call",
        )

    async def test_cache_invalidated_on_router_config_model_or_provider_change(self):
        """Changing router provider must invalidate cache, but legacy router_model must be inert."""
        cfg1 = defaults()
        plugin1, _, provider1, _ = fixture(cfg1)
        key1 = plugin1.router._get_cache_key("test input", [])

        # Legacy model changed -> ignored in cache key
        cfg2 = copy.deepcopy(cfg1)
        cfg2["router_config"]["router_model"] = "model-beta"
        plugin2, _, _, _ = fixture(cfg2)
        key2 = plugin2.router._get_cache_key("test input", [])
        self.assertEqual(key1, key2, "Cache key must remain identical when legacy router_model changes")

        # Provider changed -> alters cache key
        cfg3 = copy.deepcopy(cfg1)
        cfg3["router_config"]["router_provider"] = "different-router-provider"
        plugin3, _, _, _ = fixture(cfg3)
        key3 = plugin3.router._get_cache_key("test input", [])
        self.assertNotEqual(key1, key3, "Cache key must differ when router_provider changes")


class ConservativeFallbackContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_json_conservative_fallback(self):
        """Invalid JSON from provider must return None, not crash, not pollute cache or event."""
        plugin, _, _, _ = fixture(raw="{corrupted: not_valid_json]")
        event = Event(extras={"selected_provider": "pre-existing-provider", "selected_model": "pre-existing-model"})
        await plugin.pre_route_message(event)

        # Pre-existing defaults remain untouched
        self.assertEqual(event.get_extra("selected_provider"), "pre-existing-provider")
        self.assertEqual(event.get_extra("selected_model"), "pre-existing-model")
        self.assertEqual(len(plugin.router._cache), 0, "Invalid JSON should not be cached")

    async def test_provider_exception_conservative_fallback(self):
        """Provider exception during routing must return None, not crash or set extras."""
        plugin, _, provider, _ = fixture()
        provider.text_chat.side_effect = RuntimeError("Offline connection reset")
        event = Event()
        await plugin.pre_route_message(event)

        self.assertNotIn("selected_provider", event.extras)
        self.assertNotIn("selected_model", event.extras)
        self.assertEqual(len(plugin.router._cache), 0, "Exceptions must not populate cache")
