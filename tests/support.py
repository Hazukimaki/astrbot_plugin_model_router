"""Standard-library-only AstrBot stubs and test fixture helpers.

Import unmodified production files using importlib without installing AstrBot.
All paths are derived relative to tests/__file__.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, Mock

sys.dont_write_bytecode = True

TESTS_DIR = Path(__file__).resolve().parent
PLUGIN_ROOT = TESTS_DIR.parent
SCHEMA_PATH = PLUGIN_ROOT / "_conf_schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

LOGGER = Mock(name="astrbot_logger")


def decorator(*args, **kwargs):
    def wrap(function):
        return function
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return wrap


class Star:
    def __init__(self, context):
        self.context = context


class Context:
    def __init__(self):
        self.conversation_manager = None

    async def send_message(self, sid, chain):
        return True


class Provider:
    pass


class ProviderRequest:
    pass


class Plain:
    def __init__(self, text=""):
        self.text = text


class MessageChain:
    def __init__(self):
        self.chain = []


class Event:
    def __init__(self, text="hello", sid="test:private:1", awake=True, extras=None):
        self.message_str = text
        self.unified_msg_origin = sid
        self.is_at_or_wake_command = awake
        self.extras = dict(extras or {})

    def get_extra(self, name, default=None):
        return self.extras.get(name, default)

    def set_extra(self, name, value):
        self.extras[name] = value


AstrMessageEvent = Event


def install_stubs():
    module_names = [
        "astrbot",
        "astrbot.api",
        "astrbot.api.all",
        "astrbot.api.star",
        "astrbot.api.event",
        "astrbot.api.event.filter",
        "astrbot.core",
        "astrbot.core.provider",
        "astrbot.core.provider.entities",
        "astrbot.core.message",
        "astrbot.core.message.components",
        "astrbot.core.message.message_event_result",
    ]
    for name in module_names:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(sys.modules[parent], child, mod)

    sys.modules["astrbot.api"].logger = LOGGER
    api_all = sys.modules["astrbot.api.all"]
    api_exports = {
        "Star": Star,
        "Context": Context,
        "AstrMessageEvent": AstrMessageEvent,
        "register": decorator,
        "logger": LOGGER,
    }
    api_all.__dict__.update(api_exports)
    api_all.__all__ = list(api_exports.keys())

    sys.modules["astrbot.api.star"].Context = Context
    sys.modules["astrbot.core.provider"].Provider = Provider
    sys.modules["astrbot.core.provider.entities"].ProviderRequest = ProviderRequest
    sys.modules["astrbot.core.message.components"].Plain = Plain
    sys.modules["astrbot.core.message.message_event_result"].MessageChain = MessageChain

    filters = sys.modules["astrbot.api.event.filter"]
    filters.EventMessageType = types.SimpleNamespace(ALL="ALL")
    filters.event_message_type = decorator
    filters.command = decorator
    filters.after_message_sent = decorator


install_stubs()

PKG_NAME = "astrbot_plugin_model_router"
if PKG_NAME not in sys.modules:
    pkg = types.ModuleType(PKG_NAME)
    pkg.__path__ = [str(PLUGIN_ROOT)]
    pkg.__package__ = PKG_NAME
    sys.modules[PKG_NAME] = pkg
else:
    pkg = sys.modules[PKG_NAME]


def load_real(name):
    qualified = f"{PKG_NAME}.{name}"
    spec = importlib.util.spec_from_file_location(
        qualified,
        PLUGIN_ROOT / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = PKG_NAME
    sys.modules[qualified] = module
    setattr(pkg, name, module)
    spec.loader.exec_module(module)
    return module


routing = load_real("routing")
main = load_real("main")


def reload_modules():
    """Reload production modules in case they are modified on disk."""
    global routing, main
    routing = load_real("routing")
    main = load_real("main")
    return routing, main


def defaults():
    def resolve(schema):
        return {
            key: resolve(item["items"]) if item.get("type") == "object"
            else item.get("default")
            for key, item in schema.items()
        }
    cfg = resolve(SCHEMA)
    cfg["router_config"]["router_provider"] = "router"
    for tier in ("low", "mid", "high"):
        cfg[f"tier_{tier}"]["global_provider"] = f"{tier}-provider"
    return cfg


def response(score=2, category="chat", **values):
    return {
        "difficulty_score": score,
        "category": category,
        "context_relation": "unrelated",
        "continued_task_id": None,
        **values,
    }


def fixture(config=None, messages=None, raw=None):
    cfg = defaults() if config is None else config
    provider = Mock(spec=Provider)
    provider.text_chat = AsyncMock(
        return_value=types.SimpleNamespace(
            completion_text=json.dumps(response()) if raw is None else raw
        )
    )
    targets = {
        "router": provider,
        "low-provider": object(),
        "mid-provider": object(),
        "high-provider": object(),
    }
    context = Mock(spec=Context)
    context.get_provider_by_id = Mock(side_effect=targets.get)
    context.send_message = AsyncMock(return_value=True)

    conv_mgr = types.SimpleNamespace(
        get_curr_conversation_id=AsyncMock(
            return_value="conversation" if messages is not None else None
        ),
        get_conversation=AsyncMock(
            return_value=types.SimpleNamespace(messages=messages or [])
        ),
    )
    context.conversation_manager = conv_mgr
    plugin = main.ModelRouterPlugin(context, cfg)
    return plugin, context, provider, targets
