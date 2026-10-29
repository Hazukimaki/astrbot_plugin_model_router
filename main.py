
import json
import time
from astrbot.api.all import *
from astrbot.api.event import filter
from astrbot.api.event.filter import after_message_sent
from astrbot.core.provider.entities import ProviderRequest

from .routing import IntentRouter

PLUGIN_VERSION = "0.6.2"

@register(
    "astrbot_plugin_model_router",
    "Antigravity",
    "基于意图识别的多模型路由插件",
    PLUGIN_VERSION,
    "https://github.com/Hazukimaki/astrbot_plugin_model_router"
)
class ModelRouterPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.router = IntentRouter(context, config)
        self.task_snapshots = {}  # {session_id: {task_id: {score, category, summary, time}}}

    @filter.event_message_type(filter.EventMessageType.ALL, priority=9999)
    async def pre_route_message(self, event: AstrMessageEvent):
        """
        High-priority message handler that runs BEFORE the LLM stage.
        Analyzes intent and sets the complete selected_provider ID on the event.
        This way, when InternalAgentSubStage._select_provider() runs, it will find our selection.
        """
        # Skip if not activated (wake word/@ check)
        if not event.is_at_or_wake_command:
            return
        
        # 0. Plugin Switch
        if not self.config.get("plugin_enabled", True):
            return
        
        # 0.5 Skip if a command handler is matched (e.g., /model, /help, etc.)
        # handlers_parsed_params contains only handlers that matched as commands
        handlers_parsed_params = event.get_extra("handlers_parsed_params", {})
        if handlers_parsed_params:
            logger.debug(f"🔌 Router: Skipping - command matched: {list(handlers_parsed_params.keys())}")
            return
        
        # 1. Session Filtering (Blacklist/Whitelist)
        sid = event.unified_msg_origin
        session_cfg = self.config.get("session_control", {})
        filter_mode = session_cfg.get("filter_type", "blacklist")
        whitelist = session_cfg.get("whitelist", [])
        blacklist = session_cfg.get("blacklist", [])
        
        if filter_mode == "blacklist":
            if sid in blacklist:
                logger.debug(f"🔌 Router: Session {sid} in blacklist, skipping.")
                return
        else:  # whitelist
            if sid not in whitelist:
                logger.debug(f"🔌 Router: Session {sid} NOT in whitelist, skipping.")
                return
        
        # 2. Analyze Intent (using message_str since we don't have ProviderRequest yet)
        try:
            start_time = time.time()
            
            user_text = event.message_str
            if not user_text:
                return
            
            # Zero turns must not even read conversation history. The router
            # applies the same message/character limits to all entry points.
            context_turns = max(0, self.config.get("router_config", {}).get("context_turns", 4))
            contexts = []
            if context_turns > 0:
                try:
                    conv_mgr = self.context.conversation_manager
                    umo = event.unified_msg_origin
                    cid = await conv_mgr.get_curr_conversation_id(umo)
                    if cid:
                        conv = await conv_mgr.get_conversation(umo, cid)
                        if conv and conv.messages:
                            for msg in conv.messages[-context_turns * 2:]:
                                contexts.append({"role": msg.role, "content": str(msg.content)})
                except Exception as e:
                    logger.debug(f"Could not get conversation context: {e}")
            
            # === 获取任务快照 ===
            sid = event.unified_msg_origin
            session_snapshots = self.task_snapshots.get(sid, {})
            
            # 清理过期快照 (基于轮数，而非时间)
            # 每个快照有 turn_count，每次对话后递增
            # 当 turn_count 超过 context_turns 时过期
            valid_snapshots = {}
            for task_id, snap in session_snapshots.items():
                snap["turn_count"] = snap.get("turn_count", 0) + 1
                if snap["turn_count"] <= context_turns:
                    valid_snapshots[task_id] = snap
                else:
                    logger.debug(f"📤 Snapshot expired: {task_id} (turn {snap['turn_count']} > {context_turns})")
            self.task_snapshots[sid] = valid_snapshots
            
            # 构建快照列表供路由器使用
            snapshot_list = []
            for task_id, snap in valid_snapshots.items():
                snapshot_list.append({
                    "id": task_id,
                    "category": snap["category"],
                    "score": snap["score"],
                    "summary": snap.get("summary", "")[:100]
                })

            logger.info(f"🧩 Router analyzing: '{user_text[:30]}...' (Active snapshots: {len(snapshot_list)})")
            analysis = await self.router.analyze_intent(user_text, contexts, task_snapshots=snapshot_list)
            
            end_time = time.time()
            router_time_ms = (end_time - start_time) * 1000
            
            if not analysis:
                debug_on = self.config.get("router_config", {}).get("debug_mode", False)
                if debug_on:
                    logger.warning("⚠️ Router analysis returned None.")
                return
            
            ai_score = analysis.get("difficulty_score", 1)
            category = analysis.get("category", "chat")
            reasoning = analysis.get("reasoning", "")
            context_relation = analysis.get("context_relation", "unrelated")
            continued_task_id = analysis.get("continued_task_id")
            
            # === 根据 context_relation 决定最终分数 ===
            final_score = ai_score
            score_source = "ai"  # 用于 debug
            
            if context_relation == "continue" and continued_task_id:
                # 延续：使用快照分数
                continued_snap = valid_snapshots.get(continued_task_id)
                if continued_snap:
                    final_score = continued_snap["score"]
                    score_source = f"snapshot:{continued_task_id}"
                    logger.info(f"🔄 Context CONTINUE: Using snapshot score {final_score} from {continued_task_id}")
                    
            elif context_relation == "downgrade" and continued_task_id:
                # 降级：使用 AI 评判的分数
                score_source = f"downgrade:{continued_task_id}"
                logger.info(f"🔽 Context DOWNGRADE: AI re-evaluated to {final_score}")
                
            else:  # "unrelated" 或无有效快照
                score_source = "new"
                logger.info(f"🆕 Context UNRELATED: Independent score {final_score}")
            
            # === 更新快照 (仅当 score >= 4 且非纯闲聊) ===
            if context_turns > 0 and final_score >= 4:
                # 生成新的 task_id
                task_id = f"task_{int(time.time()) % 10000}"
                # 从用户输入生成简短摘要
                summary = user_text[:50] + ("..." if len(user_text) > 50 else "")
                
                valid_snapshots[task_id] = {
                    "score": final_score,
                    "category": category,
                    "summary": summary,
                    "turn_count": 0  # 新快照从 0 开始计数
                }
                self.task_snapshots[sid] = valid_snapshots
                logger.debug(f"📸 Snapshot saved: {task_id} (Score {final_score}, Cat: {category})")
            # 注意：低分闲聊不更新快照，保留之前的高难度任务记录
            
            # 3. The selected Chat Provider already identifies a concrete model.
            t_provider_id, t_tier_name = self.get_target_config(category, final_score)
            
            debug_on = self.config.get("router_config", {}).get("debug_mode", False)
            if debug_on:
                logger.info(f"🎯 Routing: {category} (Score {final_score} | {t_tier_name}) -> {t_provider_id}")
            
            if not t_provider_id:
                # No routing configured, let AstrBot use default
                return
            
            # 4. Resolve the exact ID from the selector, without splitting it.
            t_provider = self.context.get_provider_by_id(t_provider_id)
            if not t_provider:
                logger.error(f"❌ Target provider '{t_provider_id}' not found.")
                return
            
            logger.info(f"✅ Router: Pre-selecting Chat Provider={t_provider_id}")
            
            # 5. AstrBot resolves the model from this provider; do not inject a
            # separate model override, even if old config files still contain it.
            event.set_extra("selected_provider", t_provider_id)
            
            # Store debug data for later (will be formatted and sent in on_after_message_sent)
            if debug_on:
                event.set_extra("_router_debug_data", {
                    "time_ms": router_time_ms,
                    "router_provider": self.config.get("router_config", {}).get("router_provider", ""),
                    "category": category,
                    "ai_score": ai_score,
                    "final_score": final_score,
                    "tier_name": t_tier_name,
                    "target_provider": t_provider_id,
                    "context_relation": context_relation,
                    "continued_task_id": continued_task_id,
                    "score_source": score_source,
                    "active_snapshots": len(valid_snapshots),
                    "reasoning": reasoning,
                    "origin_sid": event.unified_msg_origin
                })
            
            # Don't stop event - let AstrBot continue with our selected provider
            
        except Exception as e:
            logger.error(f"Router error in pre_route_message: {e}")
            import traceback
            logger.error(traceback.format_exc())

    @after_message_sent()
    async def on_after_message_sent(self, event: AstrMessageEvent):
        """消息发送后，创建并发送 debug 信息到指定 SID"""
        debug_data = event.get_extra("_router_debug_data")
        if not debug_data:
            return
        
        # 清除标记，避免重复发送
        event.set_extra("_router_debug_data", None)
        
        # 检查是否是原始请求的会话
        if debug_data.get("origin_sid") != event.unified_msg_origin:
            return
        
        # 获取配置的 debug 目标 SID
        debug_target_sid = self.config.get("router_config", {}).get("debug_target_sid", "")
        if not debug_target_sid:
            # 没有配置目标 SID，只记录到日志
            logger.info(f"[Router Debug] {debug_data}")
            return
        
        # 格式化 debug 消息 (新增 context_relation 等字段)
        context_info = f"📋 Context: {debug_data['context_relation']}"
        if debug_data['continued_task_id']:
            context_info += f" -> {debug_data['continued_task_id']}"
        
        score_info = f"Score {debug_data['final_score']}"
        if debug_data['ai_score'] != debug_data['final_score']:
            score_info = f"Score {debug_data['ai_score']}→{debug_data['final_score']}"
        
        debug_msg = (
            f"[🧩 Model Router Debug]\n"
            f"⏱️ Time: {debug_data['time_ms']:.1f}ms\n"
            f"🤖 Router: {debug_data['router_provider']}\n"
            f"{context_info} (Snapshots: {debug_data['active_snapshots']})\n"
            f"🎯 Target: {debug_data['category']} ({score_info} | {debug_data['tier_name']}) -> {debug_data['target_provider']}\n"
            f"💡 Reasoning: {debug_data['reasoning']}"
        )
        
        # 发送到指定 SID
        try:
            from astrbot.core.message.components import Plain
            from astrbot.core.message.message_event_result import MessageChain
            
            message_chain = MessageChain()
            message_chain.chain.append(Plain(debug_msg))
            
            success = await self.context.send_message(debug_target_sid, message_chain)
            if not success:
                logger.warning(f"Failed to find platform for debug SID: {debug_target_sid}")
        except Exception as e:
            logger.error(f"Failed to send debug to {debug_target_sid}: {e}")

        
    def get_target_config(self, category: str, difficulty: int):
        """Return the complete Chat Provider ID and configured difficulty tier."""
        t_low = self.config.get("tier_low", {}).get("max_score", 3)
        t_mid = self.config.get("tier_mid", {}).get("max_score", 7)
        if difficulty <= t_low:
            curr_tier = "low"
        elif difficulty <= t_mid:
            curr_tier = "mid"
        else:
            curr_tier = "high"
        return self._get_tier_provider(category, curr_tier), curr_tier

    def _get_tier_provider(self, category: str, tier_name: str):
        """Prefer the first matching category, then the tier's default provider."""
        tier_config = self.config.get(f"tier_{tier_name}", {})
        for i in range(1, 7):
            name = tier_config.get(f"r{i}_name", "")
            if name and name.lower() == category.lower():
                return tier_config.get(f"r{i}_provider", "") or self.get_fallback_config(tier_name)
        return self.get_fallback_config(tier_name)

    def get_fallback_config(self, tier_name: str):
        """Return the tier's default Chat Provider ID (empty means no override)."""
        return self.config.get(f"tier_{tier_name}", {}).get("global_provider", "") or ""

    def _generate_config_table(self) -> str:
        """Show the provider IDs actually used, never retired model-name fields."""
        tier_icons = {"low": "🟢Low", "mid": "🟡Mid", "high": "🔴High"}
        default_label = "AstrBot 默认（未配置）"
        lines = ["默认目标 (Chat Provider):"]
        categories = set()
        for tier_name, tier_label in tier_icons.items():
            provider_id = self.get_fallback_config(tier_name)
            lines.append(f"  {tier_label}: {provider_id or default_label}")
            tier_config = self.config.get(f"tier_{tier_name}", {})
            for i in range(1, 7):
                name = tier_config.get(f"r{i}_name", "")
                if name:
                    categories.add(name)

        for category in sorted(categories):
            lines.append(f"\n▸ {category}")
            for tier_name, tier_label in tier_icons.items():
                provider_id = self._get_tier_provider(category, tier_name)
                lines.append(f"  {tier_label}: {provider_id or default_label}")
        return "\n".join(lines)

    @filter.command("router")
    async def router_command(self, event: AstrMessageEvent):
        """模型路由器管理命令"""
        args = event.message_str.split()
        
        if len(args) < 2:
            result = event.plain_result(
                "🧩 模型路由器命令:\n"
                "/router config - 显示当前路由配置\n"
                "/router debug - 切换调试模式\n"
                "/router status - 查看路由器状态\n"
                "/router list - 显示黑白名单\n"
                "/router add [sid] - 添加会话到名单\n"
                "/router remove [sid] - 从名单移除会话"
            )
            result.use_t2i(False)
            return result
        
        sub_cmd = args[1].lower()
        
        if sub_cmd == "config":
            table = self._generate_config_table()
            result = event.plain_result(f"📊 模型路由配置:\n{table}")
            result.use_t2i(False)  # 禁止文转图
            return result
        
        elif sub_cmd == "debug":
            if len(args) < 3:
                # 没有参数时，切换当前状态
                if "router_config" not in self.config:
                    self.config["router_config"] = {}
                current = self.config["router_config"].get("debug_mode", False)
                new_state = not current
                self.config["router_config"]["debug_mode"] = new_state
                return event.plain_result(f"🐛 Debug mode: {'ON' if new_state else 'OFF'}")
            
            mode = args[2].lower()
            if mode in ["on", "true", "1"]:
                if "router_config" not in self.config:
                    self.config["router_config"] = {}
                self.config["router_config"]["debug_mode"] = True
                return event.plain_result("🐛 Debug mode enabled")
            elif mode in ["off", "false", "0"]:
                if "router_config" not in self.config:
                    self.config["router_config"] = {}
                self.config["router_config"]["debug_mode"] = False
                return event.plain_result("🐛 Debug mode disabled")
            else:
                return event.plain_result("Usage: /router debug [on|off]")
        
        elif sub_cmd == "status":
            enabled = self.config.get("plugin_enabled", True)
            debug = self.config.get("router_config", {}).get("debug_mode", False)
            router_provider = self.config.get("router_config", {}).get("router_provider", "Not set")
            
            return event.plain_result(
                f"🧩 Model Router Status:\n"
                f"- Enabled: {'Yes' if enabled else 'No'}\n"
                f"- Debug: {'On' if debug else 'Off'}\n"
                f"- Router Chat Provider: {router_provider or 'Not set'}\n"
                f"- Version: {PLUGIN_VERSION}"
            )
        
        elif sub_cmd == "list":
            session_cfg = self.config.get("session_control", {})
            filter_type = session_cfg.get("filter_type", "blacklist")
            whitelist = session_cfg.get("whitelist", [])
            blacklist = session_cfg.get("blacklist", [])
            
            lines = [f"📋 Session Filter Mode: {filter_type.upper()}"]
            
            if filter_type == "whitelist":
                lines.append(f"\n✅ Whitelist ({len(whitelist)} entries):")
                if whitelist:
                    for sid in whitelist[:20]:  # 最多显示20个
                        lines.append(f"  • {sid}")
                    if len(whitelist) > 20:
                        lines.append(f"  ... and {len(whitelist) - 20} more")
                else:
                    lines.append("  (empty)")
            else:
                lines.append(f"\n🚫 Blacklist ({len(blacklist)} entries):")
                if blacklist:
                    for sid in blacklist[:20]:
                        lines.append(f"  • {sid}")
                    if len(blacklist) > 20:
                        lines.append(f"  ... and {len(blacklist) - 20} more")
                else:
                    lines.append("  (empty)")
            
            result = event.plain_result("\n".join(lines))
            result.use_t2i(False)
            return result
        
        elif sub_cmd == "add":
            if len(args) < 3:
                # 使用当前会话的 SID
                sid = event.unified_msg_origin
            else:
                sid = args[2]
            
            if "session_control" not in self.config:
                self.config["session_control"] = {}
            
            filter_type = self.config["session_control"].get("filter_type", "blacklist")
            list_key = "whitelist" if filter_type == "whitelist" else "blacklist"
            
            if list_key not in self.config["session_control"]:
                self.config["session_control"][list_key] = []
            
            if sid not in self.config["session_control"][list_key]:
                self.config["session_control"][list_key].append(sid)
                return event.plain_result(f"✅ Added to {list_key}: {sid}")
            else:
                return event.plain_result(f"⚠️ Already in {list_key}: {sid}")
        
        elif sub_cmd == "remove":
            if len(args) < 3:
                sid = event.unified_msg_origin
            else:
                sid = args[2]
            
            if "session_control" not in self.config:
                return event.plain_result(f"⚠️ Not found: {sid}")
            
            filter_type = self.config["session_control"].get("filter_type", "blacklist")
            list_key = "whitelist" if filter_type == "whitelist" else "blacklist"
            
            if list_key in self.config["session_control"] and sid in self.config["session_control"][list_key]:
                self.config["session_control"][list_key].remove(sid)
                return event.plain_result(f"✅ Removed from {list_key}: {sid}")
            else:
                return event.plain_result(f"⚠️ Not found in {list_key}: {sid}")
        
        else:
            return event.plain_result(f"Unknown subcommand: {sub_cmd}\nUse /router for help.")
