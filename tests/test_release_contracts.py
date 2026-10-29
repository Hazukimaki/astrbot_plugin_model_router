"""Release consistency, prompt safety and live cache invalidation regressions."""
import json
import unittest

try:
    from .support import Event, PLUGIN_ROOT, defaults, fixture, main
except ImportError:
    from support import Event, PLUGIN_ROOT, defaults, fixture, main


class CacheMutationTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_configuration_changes_reissue_analysis(self):
        cfg = defaults()
        plugin, _, provider, targets = fixture(cfg)
        targets["router-secondary"] = provider
        changes = [
            ("router_config", "router_model", "changed-model"),
            ("router_config", "router_provider", "router-secondary"),
            ("router_config", "router_manual_prompt", "JSON: {categories} {valid_cats_json}"),
            ("tier_low", "r1_desc", "Updated category definition"),
            ("tier_mid", "max_score", 8),
        ]
        await plugin.router.analyze_intent("identical input")
        for count, (section, key, value) in enumerate(changes, start=2):
            with self.subTest(section=section, key=key):
                cfg[section][key] = value
                await plugin.router.analyze_intent("identical input")
                self.assertEqual(provider.text_chat.await_count, count)
        self.assertEqual(provider.text_chat.await_args.kwargs["model"], "changed-model")

    async def test_removed_provider_cannot_serve_cached_analysis(self):
        plugin, _, provider, targets = fixture()
        await plugin.router.analyze_intent("same input")
        del targets["router"]
        self.assertIsNone(await plugin.router.analyze_intent("same input"))
        provider.text_chat.assert_awaited_once()

    async def test_identical_effective_history_can_reuse_cache(self):
        cfg = defaults()
        cfg["router_config"].update(context_turns=1, context_max_chars=3)
        plugin, _, provider, _ = fixture(cfg)
        await plugin.router.analyze_intent("q", ["excluded-old", "abc-one", "def-one"])
        await plugin.router.analyze_intent("q", ["excluded-new", "abc-two", "def-two"])
        provider.text_chat.assert_awaited_once()


class PromptContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_literals_and_escaped_category_names_are_preserved(self):
        cfg = defaults()
        name = 'custom"quoted\\分类'
        cfg["tier_low"]["r6_name"] = name
        literal = '{"difficulty_score": 2, "category": "chat"}'
        cfg["router_config"]["router_manual_prompt"] = (
            literal + "\nCATEGORIES={valid_cats_json}\nRULES={categories}\nTASKS={task_snapshots}"
        )
        plugin, _, provider, _ = fixture(cfg)
        await plugin.router.analyze_intent("q")
        prompt = provider.text_chat.await_args.kwargs["prompt"]
        categories_line = prompt.split("CATEGORIES=", 1)[1].splitlines()[0]
        self.assertIn(name, json.loads(categories_line))
        self.assertIn(literal, prompt)
        for marker in ("{categories}", "{valid_cats_json}", "{task_snapshots}"):
            self.assertNotIn(marker, prompt)

    async def test_direct_zero_turn_router_call_ignores_history_and_snapshots(self):
        cfg = defaults()
        cfg["router_config"]["context_turns"] = 0
        plugin, _, provider, _ = fixture(cfg)
        await plugin.router.analyze_intent(
            "q", ["PRIVATE_HISTORY"],
            [{"id": "PRIVATE_TASK", "category": "code", "score": 8, "summary": "PRIVATE_SUMMARY"}],
        )
        prompt = provider.text_chat.await_args.kwargs["prompt"]
        for marker in ("PRIVATE_HISTORY", "PRIVATE_TASK", "PRIVATE_SUMMARY"):
            self.assertNotIn(marker, prompt)


class ReleaseContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_status_metadata_and_readme_share_release_version(self):
        plugin, _, _, _ = fixture()
        event = Event(text="/router status")
        event.plain_result = lambda text: text
        status = await plugin.router_command(event)
        self.assertEqual(main.PLUGIN_VERSION, "0.6.1")
        self.assertIn("Version: " + main.PLUGIN_VERSION, status)
        metadata = (PLUGIN_ROOT / "metadata.yaml").read_text(encoding="utf-8")
        self.assertIn("version: v" + main.PLUGIN_VERSION, metadata.splitlines())
        self.assertIn("repo: https://github.com/Hazukimaki/astrbot_plugin_model_router", metadata.splitlines())
        readme = (PLUGIN_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("### v" + main.PLUGIN_VERSION, readme)
