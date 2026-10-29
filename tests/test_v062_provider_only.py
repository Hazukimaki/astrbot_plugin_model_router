"""v0.6.2 provider-only regression and contract tests.

Verifies:
1. Schema contract:
   - 22 deprecated model fields removed from schema
   - 22 select_provider selectors retained with _special == 'select_provider'
2. Router execution contract:
   - provider.text_chat called without 'model' keyword argument
   - Legacy model overrides ignored and not passed to provider.text_chat
3. Target injection contract:
   - Event receives full selected_provider string (even with vendor slashes)
   - Plugin never injects selected_model
   - Pre-existing selected_model from upstream plugins is not overwritten
   - Pre-existing extras preserved when routing is skipped or fails
4. Complete 3 tiers x 6 slots (18 rules) + 3 global tier fallbacks:
   - All 18 category slots resolve to their configured Chat Provider
   - Unmatched or blank-provider rules resolve to tier's global fallback
   - get_fallback_config returns provider ID string (empty string when unconfigured)
5. UI and debug formatting:
   - _router_debug_data uses router_provider and target_provider
   - Debug message displays full provider IDs
   - /router status displays full router provider ID and v0.6.2
   - _generate_config_table displays real provider IDs, including fallbacks,
     even when no categories are configured
6. Cache granularity:
   - Modifying legacy model keys does NOT invalidate cache
   - Modifying provider selectors, thresholds, or prompts DOES invalidate cache
7. Version consistency across plugin, metadata, and README.
"""
import copy
import json
import unittest
from unittest.mock import AsyncMock, Mock

try:
    from .support import Event, PLUGIN_ROOT, SCHEMA, defaults, fixture, main, response
except ImportError:
    from support import Event, PLUGIN_ROOT, SCHEMA, defaults, fixture, main, response


class SchemaContractTests(unittest.TestCase):
    def test_schema_model_fields_completely_removed(self):
        """All 22 deprecated model fields must be absent from schema."""
        # 1. router_model in router_config
        self.assertNotIn("router_model", SCHEMA.get("router_config", {}).get("items", {}))

        # 2. global_model and r1_model..r6_model in tier_low, tier_mid, tier_high (7 * 3 = 21)
        for tier in ("tier_low", "tier_mid", "tier_high"):
            items = SCHEMA.get(tier, {}).get("items", {})
            self.assertNotIn("global_model", items, f"global_model must not exist in {tier}")
            for i in range(1, 7):
                self.assertNotIn(f"r{i}_model", items, f"r{i}_model must not exist in {tier}")

        # Comprehensive walk: No field in any items dict should end with '_model'
        for section_name, section in SCHEMA.items():
            if section.get("type") == "object":
                for item_name in section.get("items", {}):
                    self.assertFalse(
                        item_name.endswith("_model"),
                        f"Found deprecated model key '{item_name}' in section '{section_name}'",
                    )

    def test_schema_all_22_provider_selectors_retained(self):
        """All 22 select_provider selectors must remain present with _special='select_provider'."""
        router_items = SCHEMA.get("router_config", {}).get("items", {})
        self.assertIn("router_provider", router_items)
        self.assertEqual(router_items["router_provider"].get("_special"), "select_provider")

        selector_count = 1  # router_provider counted
        for tier in ("tier_low", "tier_mid", "tier_high"):
            items = SCHEMA.get(tier, {}).get("items", {})
            self.assertIn("global_provider", items, f"global_provider missing from {tier}")
            self.assertEqual(
                items["global_provider"].get("_special"),
                "select_provider",
                f"global_provider in {tier} must have _special='select_provider'",
            )
            selector_count += 1
            for i in range(1, 7):
                rule_key = f"r{i}_provider"
                self.assertIn(rule_key, items, f"{rule_key} missing from {tier}")
                self.assertEqual(
                    items[rule_key].get("_special"),
                    "select_provider",
                    f"{rule_key} in {tier} must have _special='select_provider'",
                )
                selector_count += 1

        self.assertEqual(selector_count, 22, f"Expected 22 provider selectors, got {selector_count}")


class ProviderOnlyExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_model_overrides_not_passed_to_text_chat(self):
        """router provider.text_chat must be called without the 'model' keyword argument."""
        cfg = defaults()
        cfg["router_config"]["router_model"] = "legacy-override-model"
        plugin, _, provider, _ = fixture(cfg)
        await plugin.router.analyze_intent("analyze this input")
        provider.text_chat.assert_awaited_once()
        self.assertNotIn(
            "model",
            provider.text_chat.await_args.kwargs,
            "provider.text_chat must not receive 'model' kwarg (not even None)",
        )

    async def test_legacy_model_overrides_not_injected_into_event(self):
        """Plugin must set selected_provider but never inject selected_model, even with legacy keys."""
        cfg = defaults()
        cfg["router_config"]["router_model"] = "legacy-router-model"
        cfg["tier_low"]["global_model"] = "legacy-global-model"
        cfg["tier_low"]["r1_model"] = "legacy-r1-model"
        plugin, _, _, _ = fixture(cfg)
        event = Event(text="simple greeting")
        await plugin.pre_route_message(event)

        self.assertEqual(event.get_extra("selected_provider"), "low-provider")
        self.assertNotIn("selected_model", event.extras, "selected_model must not be injected by plugin")

    async def test_existing_upstream_selected_model_not_overwritten_by_plugin(self):
        """Plugin does not promise to clear upstream selected_model, only not to produce model overrides."""
        cfg = defaults()
        plugin, _, _, _ = fixture(cfg)
        event = Event(
            text="hello",
            extras={"selected_model": "upstream-plugin-model", "other_extra": "intact"},
        )
        await plugin.pre_route_message(event)

        self.assertEqual(event.get_extra("selected_provider"), "low-provider")
        # Pre-existing selected_model must not be overwritten or cleared
        self.assertEqual(event.get_extra("selected_model"), "upstream-plugin-model")
        self.assertEqual(event.get_extra("other_extra"), "intact")

    async def test_unrouted_event_preserves_all_existing_extras(self):
        """When routing is skipped or fails, pre-existing extras remain completely intact."""
        cfg = defaults()
        cfg["plugin_enabled"] = False
        plugin, _, _, _ = fixture(cfg)
        initial_extras = {
            "selected_provider": "default-fallback-provider",
            "selected_model": "default-fallback-model",
            "session_user": "user_123",
        }
        event = Event(extras=dict(initial_extras))
        await plugin.pre_route_message(event)
        self.assertEqual(event.extras, initial_extras)


class SlashProviderIdTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_provider_id_with_slashes_and_dashes_preserved_and_injected(self):
        """Full Chat Provider IDs (e.g. 'minimax-token-plan/MiniMax-M3.1-Flash') are looked up and injected verbatim."""
        router_id = "minimax-token-plan/MiniMax-M3.1-Flash"
        target_id = "siliconflow/deepseek-ai/DeepSeek-V3"

        cfg = defaults()
        cfg["router_config"]["router_provider"] = router_id
        cfg["tier_low"]["global_provider"] = target_id

        router_provider = Mock()
        router_provider.text_chat = AsyncMock(
            return_value=Mock(completion_text=json.dumps(response(score=2, category="chat")))
        )
        target_provider = object()

        context_targets = {
            router_id: router_provider,
            target_id: target_provider,
        }

        context = Mock()
        context.get_provider_by_id = Mock(side_effect=context_targets.get)
        context.conversation_manager = Mock()
        context.conversation_manager.get_curr_conversation_id = AsyncMock(return_value=None)

        plugin = main.ModelRouterPlugin(context, cfg)
        event = Event(text="test slash handling")
        await plugin.pre_route_message(event)

        # 1. Exact router provider lookup
        context.get_provider_by_id.assert_any_call(router_id)
        router_provider.text_chat.assert_awaited_once()

        # 2. Exact target provider lookup without splitting '/'
        context.get_provider_by_id.assert_any_call(target_id)
        self.assertEqual(event.get_extra("selected_provider"), target_id)
        self.assertNotIn("selected_model", event.extras)


class AllTiersAndSlotsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = defaults()
        # Configure unique names and provider IDs for all 3 tiers x 6 slots (18 slots)
        for tier in ("low", "mid", "high"):
            tier_key = f"tier_{tier}"
            self.cfg[tier_key]["global_provider"] = f"vendor-{tier}/global-fallback"
            for i in range(1, 7):
                self.cfg[tier_key][f"r{i}_name"] = f"cat_{tier}_{i}"
                self.cfg[tier_key][f"r{i}_provider"] = f"vendor-{tier}/slot-{i}-chat-provider"
        self.plugin, _, _, _ = fixture(self.cfg)

    def test_all_18_rule_slots_route_to_their_chat_providers(self):
        """All 6 slots in low, mid, and high tiers must resolve correctly."""
        tier_scores = {
            "low": 2,
            "mid": 5,
            "high": 8,
        }
        for tier, score in tier_scores.items():
            for i in range(1, 7):
                category = f"cat_{tier}_{i}"
                expected_provider = f"vendor-{tier}/slot-{i}-chat-provider"
                with self.subTest(tier=tier, slot=i, category=category):
                    provider, resolved_tier = self.plugin.get_target_config(category, score)
                    self.assertEqual(resolved_tier, tier)
                    self.assertEqual(provider, expected_provider)

    def test_all_3_tier_fallbacks_route_correctly(self):
        """Unknown category in each tier falls back to that tier's global fallback provider."""
        tier_scores = {
            "low": 1,
            "mid": 6,
            "high": 9,
        }
        for tier, score in tier_scores.items():
            expected_provider = f"vendor-{tier}/global-fallback"
            with self.subTest(tier=tier, score=score):
                provider, resolved_tier = self.plugin.get_target_config("completely_unknown_cat", score)
                self.assertEqual(resolved_tier, tier)
                self.assertEqual(provider, expected_provider)

    def test_blank_slot_provider_falls_back_to_tier_global(self):
        """When a rule's r{i}_provider is blank, it falls back to the tier's global_provider."""
        self.cfg["tier_low"]["r3_provider"] = ""
        plugin, _, _, _ = fixture(self.cfg)
        provider, tier = plugin.get_target_config("cat_low_3", 2)
        self.assertEqual(tier, "low")
        self.assertEqual(provider, "vendor-low/global-fallback")

    def test_get_fallback_config_returns_string(self):
        """get_fallback_config must return provider ID string, or empty string when unconfigured."""
        self.assertEqual(self.plugin.get_fallback_config("low"), "vendor-low/global-fallback")
        self.assertEqual(self.plugin.get_fallback_config("mid"), "vendor-mid/global-fallback")
        self.assertEqual(self.plugin.get_fallback_config("high"), "vendor-high/global-fallback")

        # When global_provider is empty
        self.cfg["tier_high"]["global_provider"] = ""
        plugin, _, _, _ = fixture(self.cfg)
        self.assertEqual(plugin.get_fallback_config("high"), "")
        self.assertEqual(plugin.get_target_config("unknown", 9), ("", "high"))


class UiAndDebugDisplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_debug_data_and_debug_message_show_full_provider_ids(self):
        """_router_debug_data and formatted debug message must display full provider IDs."""
        cfg = defaults()
        cfg["router_config"]["debug_mode"] = True
        cfg["router_config"]["debug_target_sid"] = "test:target:debug_channel"
        cfg["router_config"]["router_provider"] = "router-corp/fast-router-v2"
        cfg["tier_low"]["global_provider"] = "chat-corp/smart-assistant-v1"

        plugin, context, provider, targets = fixture(cfg)
        targets["router-corp/fast-router-v2"] = provider
        targets["chat-corp/smart-assistant-v1"] = object()

        event = Event(text="need quick chat", sid="test:user:session")
        await plugin.pre_route_message(event)

        # Check debug data in event extras
        debug_data = event.get_extra("_router_debug_data")
        self.assertIsNotNone(debug_data)
        self.assertEqual(debug_data["router_provider"], "router-corp/fast-router-v2")
        self.assertEqual(debug_data["target_provider"], "chat-corp/smart-assistant-v1")
        self.assertNotIn("router_model", debug_data)
        self.assertNotIn("model_display", debug_data)

        # Trigger message sent hook to send debug message
        await plugin.on_after_message_sent(event)
        context.send_message.assert_awaited_once()
        call_args = context.send_message.await_args
        target_sid = call_args.args[0] if call_args.args else call_args.kwargs.get("sid")
        message_chain = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("chain")

        self.assertEqual(target_sid, "test:target:debug_channel")
        debug_text = message_chain.chain[0].text
        self.assertIn("🤖 Router: router-corp/fast-router-v2", debug_text)
        self.assertIn("🎯 Target: chat (Score 2 | low) -> chat-corp/smart-assistant-v1", debug_text)

    async def test_status_command_shows_full_provider_id(self):
        """/router status command output displays full router provider ID without model suffix."""
        cfg = defaults()
        cfg["router_config"]["router_provider"] = "provider-team/flagship-model-v3"
        plugin, _, _, _ = fixture(cfg)

        event = Event(text="/router status")
        event.plain_result = lambda text: text
        status_text = await plugin.router_command(event)

        self.assertIn("- Router Chat Provider: provider-team/flagship-model-v3", status_text)
        self.assertIn("- Version: 0.6.2", status_text)
        self.assertNotIn(":router_model", status_text)

    def test_config_table_shows_fallbacks_even_when_no_categories_configured(self):
        """_generate_config_table must show the 3 tier fallbacks even if no category rules exist."""
        cfg = defaults()
        # The shipped schema has default categories; explicitly clear all of
        # them so this really exercises the no-classification-rules case.
        for tier in ("tier_low", "tier_mid", "tier_high"):
            for i in range(1, 7):
                cfg[tier][f"r{i}_name"] = ""
        cfg["tier_low"]["global_provider"] = "vendor-a/model-low"
        cfg["tier_mid"]["global_provider"] = "vendor-b/model-mid"
        cfg["tier_high"]["global_provider"] = "vendor-c/model-high"

        plugin, _, _, _ = fixture(cfg)
        table = plugin._generate_config_table()

        self.assertNotIn("No routing rules configured.", table)
        self.assertNotIn("▸", table)
        self.assertIn("默认目标 (Chat Provider):", table)
        self.assertIn("🟢Low: vendor-a/model-low", table)
        self.assertIn("🟡Mid: vendor-b/model-mid", table)
        self.assertIn("🔴High: vendor-c/model-high", table)

    def test_config_table_shows_real_provider_ids_not_just_global_label(self):
        """_generate_config_table displays real provider IDs for categories rather than literal 'Global'."""
        cfg = defaults()
        cfg["tier_low"]["global_provider"] = "fallback/low-provider"
        cfg["tier_mid"]["global_provider"] = "fallback/mid-provider"
        cfg["tier_high"]["global_provider"] = "fallback/high-provider"

        cfg["tier_low"]["r1_name"] = "coding"
        cfg["tier_low"]["r1_provider"] = "specialized/code-low"

        plugin, _, _, _ = fixture(cfg)
        table = plugin._generate_config_table()

        self.assertIn("▸ coding", table)
        self.assertIn("🟢Low: specialized/code-low", table)
        # Mid and High for category 'coding' fall back to their tier global provider IDs
        self.assertIn("🟡Mid: fallback/mid-provider", table)
        self.assertIn("🔴High: fallback/high-provider", table)
        # The word 'Global' as a standalone replacement label should not be used
        self.assertNotIn(": Global", table)


class CacheGranularityTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_ignores_all_legacy_model_keys(self):
        """Altering retired model fields in router_config or tiers must not change cache key."""
        cfg1 = defaults()
        plugin1, _, provider1, _ = fixture(cfg1)
        base_key = plugin1.router._get_cache_key("test prompt", [])

        # Alter router_model
        cfg_rm = copy.deepcopy(cfg1)
        cfg_rm["router_config"]["router_model"] = "new-legacy-router-model"
        plugin_rm, _, _, _ = fixture(cfg_rm)
        self.assertEqual(
            plugin_rm.router._get_cache_key("test prompt", []),
            base_key,
            "Altering router_config.router_model must not alter cache key",
        )

        # Alter global_model in all tiers
        for tier in ("tier_low", "tier_mid", "tier_high"):
            cfg_gm = copy.deepcopy(cfg1)
            cfg_gm[tier]["global_model"] = f"new-{tier}-global-model"
            plugin_gm, _, _, _ = fixture(cfg_gm)
            self.assertEqual(
                plugin_gm.router._get_cache_key("test prompt", []),
                base_key,
                f"Altering {tier}.global_model must not alter cache key",
            )

        # Alter r1_model..r6_model in tiers
        for tier in ("tier_low", "tier_mid", "tier_high"):
            for i in range(1, 7):
                cfg_rule = copy.deepcopy(cfg1)
                cfg_rule[tier][f"r{i}_model"] = f"new-{tier}-r{i}-model"
                plugin_rule, _, _, _ = fixture(cfg_rule)
                self.assertEqual(
                    plugin_rule.router._get_cache_key("test prompt", []),
                    base_key,
                    f"Altering {tier}.r{i}_model must not alter cache key",
                )

    async def test_cache_invalidates_on_provider_and_threshold_changes(self):
        """Altering active provider selectors, thresholds, or prompts must alter cache key."""
        cfg1 = defaults()
        plugin1, _, _, _ = fixture(cfg1)
        base_key = plugin1.router._get_cache_key("test prompt", [])

        # 1. router_provider
        cfg2 = copy.deepcopy(cfg1)
        cfg2["router_config"]["router_provider"] = "different-router-provider"
        plugin2, _, _, _ = fixture(cfg2)
        self.assertNotEqual(plugin2.router._get_cache_key("test prompt", []), base_key)

        # 2. tier_low global_provider
        cfg3 = copy.deepcopy(cfg1)
        cfg3["tier_low"]["global_provider"] = "different-low-provider"
        plugin3, _, _, _ = fixture(cfg3)
        self.assertNotEqual(plugin3.router._get_cache_key("test prompt", []), base_key)

        # 3. tier_low max_score threshold
        cfg4 = copy.deepcopy(cfg1)
        cfg4["tier_low"]["max_score"] = 5
        plugin4, _, _, _ = fixture(cfg4)
        self.assertNotEqual(plugin4.router._get_cache_key("test prompt", []), base_key)

        # 4. prompt template
        cfg5 = copy.deepcopy(cfg1)
        cfg5["router_config"]["router_manual_prompt"] = "Custom prompt: {categories}"
        plugin5, _, _, _ = fixture(cfg5)
        self.assertNotEqual(plugin5.router._get_cache_key("test prompt", []), base_key)


class VersionConsistencyTests(unittest.TestCase):
    def test_version_062_consistency(self):
        """PLUGIN_VERSION, metadata.yaml, and README.md must all share v0.6.2."""
        self.assertEqual(main.PLUGIN_VERSION, "0.6.2")

        metadata_text = (PLUGIN_ROOT / "metadata.yaml").read_text(encoding="utf-8")
        self.assertIn("version: v0.6.2", metadata_text.splitlines())

        readme_text = (PLUGIN_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("### v0.6.2", readme_text)
