"""Positive/functional behavior regression tests for model router plugin."""
import json
import types
import unittest
from unittest.mock import AsyncMock, patch

try:
    from .support import Event, defaults, fixture, response, routing
except ImportError:
    from support import Event, defaults, fixture, response, routing


class ParsingAndProviders(unittest.IsolatedAsyncioTestCase):
    async def check_raw(self, raw, expected):
        plugin, _, provider, _ = fixture(raw=raw)
        actual = await plugin.router.analyze_intent("input")
        self.assertEqual(actual, expected)
        provider.text_chat.assert_awaited_once()

    async def test_json(self):
        await self.check_raw(json.dumps(response()), response())

    async def test_fenced_json(self):
        await self.check_raw("```json\n" + json.dumps(response()) + "\n```", response())

    async def test_fenced_without_language(self):
        await self.check_raw("```\n" + json.dumps(response()) + "\n```", response())

    async def test_outer_whitespace(self):
        await self.check_raw(" \n" + json.dumps(response()) + "\n ", response())

    async def test_invalid_json(self):
        await self.check_raw("not JSON", None)

    async def test_empty_output(self):
        await self.check_raw("", None)

    async def test_empty_fence_is_contained(self):
        await self.check_raw("```", None)

    async def test_provider_exception(self):
        plugin, _, provider, _ = fixture()
        provider.text_chat.side_effect = RuntimeError("offline simulated model failure")
        self.assertIsNone(await plugin.router.analyze_intent("input"))
        self.assertEqual(len(plugin.router._cache), 0)

    async def test_no_router_provider(self):
        cfg = defaults()
        cfg["router_config"]["router_provider"] = ""
        plugin, context, provider, _ = fixture(cfg)
        self.assertIsNone(await plugin.router.analyze_intent("input"))
        context.get_provider_by_id.assert_not_called()
        provider.text_chat.assert_not_awaited()

    async def test_router_provider_not_found(self):
        cfg = defaults()
        cfg["router_config"]["router_provider"] = "absent"
        plugin, _, provider, _ = fixture(cfg)
        self.assertIsNone(await plugin.router.analyze_intent("input"))
        provider.text_chat.assert_not_awaited()

    async def test_explicit_model_argument(self):
        cfg = defaults()
        cfg["router_config"]["router_model"] = "offline-router-model"
        plugin, _, provider, _ = fixture(cfg)
        await plugin.router.analyze_intent("input")
        self.assertEqual(provider.text_chat.await_args.kwargs["model"], "offline-router-model")
        self.assertEqual(provider.text_chat.await_args.kwargs["contexts"], [])

    async def test_blank_model_uses_provider_default(self):
        plugin, _, provider, _ = fixture()
        await plugin.router.analyze_intent("input")
        self.assertIsNone(provider.text_chat.await_args.kwargs["model"])


class CacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_analysis_hits_cache(self):
        plugin, _, provider, _ = fixture()
        first = await plugin.router.analyze_intent("same", [{"content": "context"}])
        second = await plugin.router.analyze_intent("same", [{"content": "context"}])
        self.assertEqual(first, second)
        provider.text_chat.assert_awaited_once()

    async def test_ttl_boundary_is_inclusive(self):
        plugin, _, _, _ = fixture()
        with patch.object(routing.time, "time", return_value=1000):
            plugin.router._set_cached("key", response())
        with patch.object(routing.time, "time", return_value=1300):
            self.assertEqual(plugin.router._get_cached("key"), response())

    async def test_ttl_expiration_reissues_call(self):
        plugin, _, provider, _ = fixture()
        with patch.object(routing.time, "time", return_value=1000):
            await plugin.router.analyze_intent("same")
        with patch.object(routing.time, "time", return_value=1300.001):
            await plugin.router.analyze_intent("same")
        self.assertEqual(provider.text_chat.await_count, 2)

    async def test_lru_hit_promotes_entry(self):
        plugin, _, _, _ = fixture()
        router = plugin.router
        router.CACHE_MAX_SIZE = 2
        router._set_cached("a", response())
        router._set_cached("b", response())
        router._get_cached("a")
        router._set_cached("c", response())
        self.assertEqual(list(router._cache), ["a", "c"])
        self.assertIsNone(router._get_cached("b"))

    async def test_default_capacity_100(self):
        plugin, _, _, _ = fixture()
        for index in range(101):
            plugin.router._set_cached(str(index), response())
        self.assertEqual(len(plugin.router._cache), 100)
        self.assertNotIn("0", plugin.router._cache)
        self.assertIn("100", plugin.router._cache)

    async def test_key_changes_with_input_or_last_context(self):
        plugin, _, _, _ = fixture()
        key = plugin.router._get_cache_key
        self.assertNotEqual(key("a", ["x"]), key("b", ["x"]))
        self.assertNotEqual(key("a", ["x"]), key("a", ["y"]))

    async def test_failed_parse_not_cached(self):
        plugin, _, provider, _ = fixture(raw="not JSON")
        await plugin.router.analyze_intent("same")
        provider.text_chat.return_value = types.SimpleNamespace(completion_text=json.dumps(response()))
        self.assertEqual(await plugin.router.analyze_intent("same"), response())
        self.assertEqual(provider.text_chat.await_count, 2)


class TierTests(unittest.TestCase):
    def test_implemented_documented_default_boundaries(self):
        plugin, _, _, _ = fixture()
        # Default thresholds are tier_low.max_score=3, tier_mid.max_score=7.
        # Score 1..3 -> low, 4..7 -> mid (7 is mid, not high), 8..9 -> high.
        for score, tier in [(1, "low"), (3, "low"), (4, "mid"), (6, "mid"), (7, "mid"), (9, "high")]:
            with self.subTest(score=score):
                self.assertEqual(
                    plugin.get_target_config("unknown", score),
                    (f"{tier}-provider", f"{tier}-model", tier),
                )

    def test_category_overrides_global_and_casefolds(self):
        cfg = defaults()
        cfg["tier_low"].update(r1_name="CoDe", r1_provider="specific", r1_model="specific-model")
        plugin, _, _, _ = fixture(cfg)
        self.assertEqual(plugin.get_target_config("code", 2), ("specific", "specific-model", "low"))

    def test_sixth_slot_supported(self):
        cfg = defaults()
        cfg["tier_mid"].update(r6_name="custom", r6_provider="six", r6_model="six-model")
        plugin, _, _, _ = fixture(cfg)
        self.assertEqual(plugin.get_target_config("custom", 5), ("six", "six-model", "mid"))

    def test_first_matching_category_wins(self):
        cfg = defaults()
        cfg["tier_low"].update(r1_name="same", r1_provider="first", r2_name="same", r2_provider="second")
        plugin, _, _, _ = fixture(cfg)
        self.assertEqual(plugin.get_target_config("same", 2)[0], "first")

    def test_unknown_category_uses_tier_global(self):
        plugin, _, _, _ = fixture()
        self.assertEqual(plugin.get_target_config("missing", 5), ("mid-provider", "mid-model", "mid"))

    def test_blank_rule_provider_uses_tier_global(self):
        cfg = defaults()
        cfg["tier_low"].update(r1_name="math", r1_provider="", r1_model="ignored-orphan-model")
        plugin, _, _, _ = fixture(cfg)
        self.assertEqual(plugin.get_target_config("math", 2), ("low-provider", "low-model", "low"))

    def test_missing_tier_returns_empty_selection(self):
        cfg = defaults()
        del cfg["tier_high"]
        plugin, _, _, _ = fixture(cfg)
        self.assertEqual(plugin.get_target_config("code", 8), ("", "", "high"))

    def test_fallback_helper(self):
        plugin, _, _, _ = fixture()
        self.assertEqual(plugin.get_fallback_config("low"), ("low-provider", "low-model"))


class EventTests(unittest.IsolatedAsyncioTestCase):
    async def gate(self, cfg=None, event=None, allowed=False):
        plugin, _, provider, _ = fixture(cfg)
        event = event or Event()
        await plugin.pre_route_message(event)
        if allowed:
            provider.text_chat.assert_awaited_once()
            self.assertEqual(event.get_extra("selected_provider"), "low-provider")
        else:
            provider.text_chat.assert_not_awaited()
            self.assertNotIn("selected_provider", event.extras)
        return plugin, event

    async def test_blacklist_denies(self):
        cfg = defaults()
        cfg["session_control"].update(filter_type="blacklist", blacklist=["test:private:1"])
        await self.gate(cfg)

    async def test_blacklist_allows_other_session(self):
        cfg = defaults()
        cfg["session_control"].update(filter_type="blacklist", blacklist=["other"])
        await self.gate(cfg, allowed=True)

    async def test_whitelist_denies_nonmember(self):
        cfg = defaults()
        cfg["session_control"].update(filter_type="whitelist", whitelist=["other"])
        await self.gate(cfg)

    async def test_whitelist_allows_member(self):
        cfg = defaults()
        cfg["session_control"].update(filter_type="whitelist", whitelist=["test:private:1"])
        await self.gate(cfg, allowed=True)

    async def test_unawakened_skips(self):
        await self.gate(event=Event(awake=False))

    async def test_disabled_plugin_skips(self):
        cfg = defaults()
        cfg["plugin_enabled"] = False
        await self.gate(cfg)

    async def test_matched_command_skips(self):
        await self.gate(event=Event(text="/help", extras={"handlers_parsed_params": {"help": {}}}))

    async def test_empty_text_skips(self):
        await self.gate(event=Event(text=""))

    async def test_real_analysis_sets_selection(self):
        _, event = await self.gate(allowed=True)
        self.assertEqual(event.get_extra("selected_model"), "low-model")

    async def test_blank_target_leaves_existing_default_untouched(self):
        cfg = defaults()
        cfg["tier_low"]["global_provider"] = ""
        plugin, _, _, _ = fixture(cfg)
        event = Event(extras={"selected_provider": "existing-default", "selected_model": "default-model"})
        await plugin.pre_route_message(event)
        self.assertEqual(event.extras, {"selected_provider": "existing-default", "selected_model": "default-model"})

    async def test_missing_target_leaves_selection_unset(self):
        cfg = defaults()
        cfg["tier_low"]["global_provider"] = "nonexistent"
        plugin, _, _, _ = fixture(cfg)
        event = Event()
        await plugin.pre_route_message(event)
        self.assertNotIn("selected_provider", event.extras)

    async def test_model_exception_leaves_selection_unset(self):
        plugin, _, provider, _ = fixture()
        provider.text_chat.side_effect = RuntimeError("simulated outage")
        event = Event()
        await plugin.pre_route_message(event)
        self.assertNotIn("selected_provider", event.extras)

    async def test_empty_router_provider_leaves_selection_unset(self):
        cfg = defaults()
        cfg["router_config"]["router_provider"] = ""
        plugin, _, provider, _ = fixture(cfg)
        event = Event()
        await plugin.pre_route_message(event)
        provider.text_chat.assert_not_awaited()
        self.assertNotIn("selected_provider", event.extras)

    async def test_empty_target_model_does_not_force_model(self):
        cfg = defaults()
        cfg["tier_low"]["global_model"] = ""
        plugin, _, _, _ = fixture(cfg)
        event = Event()
        await plugin.pre_route_message(event)
        self.assertEqual(event.get_extra("selected_provider"), "low-provider")
        self.assertNotIn("selected_model", event.extras)

    async def test_history_failure_does_not_block_routing(self):
        plugin, context, _, _ = fixture()
        context.conversation_manager.get_curr_conversation_id.side_effect = RuntimeError("history unavailable")
        event = Event()
        await plugin.pre_route_message(event)
        self.assertEqual(event.get_extra("selected_provider"), "low-provider")

    async def test_continue_uses_snapshot_score(self):
        plugin, _, _, _ = fixture()
        plugin.task_snapshots["test:private:1"] = {"task_A": {"score": 8, "category": "code", "turn_count": 0}}
        plugin.router.analyze_intent = AsyncMock(return_value=response(2, "code", context_relation="continue", continued_task_id="task_A"))
        event = Event()
        await plugin.pre_route_message(event)
        self.assertEqual(event.get_extra("selected_provider"), "high-provider")
        self.assertEqual(plugin.router.analyze_intent.await_args.kwargs["task_snapshots"][0]["id"], "task_A")

    async def test_downgrade_uses_ai_score(self):
        plugin, _, _, _ = fixture()
        plugin.task_snapshots["test:private:1"] = {"task_A": {"score": 8, "category": "code", "turn_count": 0}}
        plugin.router.analyze_intent = AsyncMock(return_value=response(2, "code", context_relation="downgrade", continued_task_id="task_A"))
        event = Event()
        await plugin.pre_route_message(event)
        self.assertEqual(event.get_extra("selected_provider"), "low-provider")

    async def test_expired_snapshot_removed(self):
        plugin, _, _, _ = fixture()
        plugin.task_snapshots["test:private:1"] = {"old": {"score": 8, "category": "code", "turn_count": 4}}
        event = Event()
        await plugin.pre_route_message(event)
        self.assertNotIn("old", plugin.task_snapshots["test:private:1"])

    async def test_context_turns_one_includes_only_two_messages(self):
        cfg = defaults()
        cfg["router_config"]["context_turns"] = 1
        plugin, _, provider, _ = fixture(cfg)
        await plugin.router.analyze_intent("question", [{"role": "user", "content": f"HISTORY_MARK_{i}"} for i in range(4)])
        prompt = provider.text_chat.await_args.kwargs["prompt"]
        self.assertNotIn("HISTORY_MARK_1", prompt)
        self.assertIn("HISTORY_MARK_2", prompt)
        self.assertIn("HISTORY_MARK_3", prompt)

    async def test_manual_categories_and_snapshots_replaced(self):
        cfg = defaults()
        cfg["router_config"]["router_manual_prompt"] = "Cats: {categories}\nTasks: {task_snapshots}"
        plugin, _, provider, _ = fixture(cfg)
        await plugin.router.analyze_intent("input", task_snapshots=[{"id": "A", "category": "code", "score": 8, "summary": "architecture"}])
        prompt = provider.text_chat.await_args.kwargs["prompt"]
        self.assertNotIn("{categories}", prompt)
        self.assertNotIn("{task_snapshots}", prompt)
        self.assertIn('"code":', prompt)
        self.assertIn("[A] Category=code, Score=8", prompt)
