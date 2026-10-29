
import json
import hashlib
import time
from typing import Dict, Any, Optional, List
from collections import OrderedDict

from astrbot.api.star import Context
from astrbot.core.provider import Provider
from astrbot.api import logger  # Use AstrBot's logger from api

class IntentRouter:
    # 缓存配置
    CACHE_MAX_SIZE = 100
    CACHE_TTL_SECONDS = 300  # 5分钟过期
    
    def __init__(self, context: Context, config: Dict[str, Any]):
        self.context = context
        self.config = config
        self._cache: OrderedDict[str, tuple] = OrderedDict()  # key -> (result, timestamp)

    def _prepare_contexts(self, contexts: List) -> List[Dict[str, str]]:
        """Apply configured history limits once, also for direct router calls."""
        router_config = self.config.get("router_config", {})
        turns = max(0, router_config.get("context_turns", 4))
        if turns == 0:
            return []
        max_chars = max(0, router_config.get("context_max_chars", 500))
        prepared = []
        for msg in (contexts or [])[-turns * 2:]:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            elif isinstance(msg, str):
                role, content = "unknown", msg
            else:
                continue
            content = "" if content is None else str(content)
            if max_chars > 0:
                content = content[:max_chars]
            if content:
                prepared.append({"role": role, "content": content})
        return prepared

    def _get_cache_key(self, text: str, contexts: List, task_snapshots: List = None) -> str:
        """Include every input that can change the routing decision."""
        turns = max(0, self.config.get("router_config", {}).get("context_turns", 4))
        # Old saved model-name overrides are intentionally inert, including in
        # the cache key. Only the selected Chat Provider determines the model.
        retired_keys = {"router_model", "global_model"} | {f"r{i}_model" for i in range(1, 7)}
        effective_config = {
            section: {key: value for key, value in self.config.get(section, {}).items()
                      if key not in retired_keys}
            for section in ("router_config", "tier_low", "tier_mid", "tier_high")
        }
        payload = {
            "text": text,
            "contexts": self._prepare_contexts(contexts),
            "task_snapshots": (task_snapshots or []) if turns > 0 else [],
            "config": effective_config,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    
    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if valid (not expired)."""
        if key not in self._cache:
            return None
        result, timestamp = self._cache[key]
        if time.time() - timestamp > self.CACHE_TTL_SECONDS:
            # 过期，删除
            del self._cache[key]
            return None
        # 移动到末尾 (LRU)
        self._cache.move_to_end(key)
        return result
    
    def _set_cached(self, key: str, result: Dict[str, Any]):
        """Store result in cache with LRU eviction."""
        self._cache[key] = (result, time.time())
        self._cache.move_to_end(key)
        # 清理超过上限的缓存
        while len(self._cache) > self.CACHE_MAX_SIZE:
            self._cache.popitem(last=False)

    async def analyze_intent(self, user_text: str, contexts: List[Dict[str, str]] = None, task_snapshots: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Analyze user intent using dynamically built system prompt.
        Now includes task snapshots for multi-task context awareness.
        Includes caching for performance optimization.
        
        Args:
            user_text: Current user input
            contexts: Recent conversation history
            task_snapshots: List of active task snapshots, each with {id, category, score, summary}
        """
        if contexts is None:
            contexts = []
        if task_snapshots is None:
            task_snapshots = []
        
        router_config = self.config.get("router_config", {})
        provider_id = router_config.get("router_provider")
        
        contexts = self._prepare_contexts(contexts)
        if router_config.get("context_turns", 4) <= 0:
            task_snapshots = []
        
        logger.debug(f"Router Config: Chat Provider={provider_id}")

        if not provider_id:
             logger.warning("Router provider ID is empty.")
             return None
        
        provider = self.context.get_provider_by_id(provider_id)
        if not provider:
            logger.error(f"Router provider not found: {provider_id}")
            return None

        # Validate the provider before cache lookup so a removed provider cannot
        # silently reuse a result. Snapshot/config/history changes invalidate it.
        cache_key = self._get_cache_key(user_text, contexts, task_snapshots)
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"Router cache hit for: {user_text[:30]}...")
            return cached

        # --- Build Dynamic System Prompt from Frames (Fixed Slots) ---
        
        categories_map = {} # name -> list of descriptions
        
        # Helper to process fixed slots
        def process_slots(tier_key, slot_count):
            tier_config = self.config.get(tier_key, {})
            if not tier_config:
                return

            for i in range(1, slot_count + 1):
                name = tier_config.get(f"r{i}_name", "").strip()
                desc = tier_config.get(f"r{i}_desc", "").strip()
                
                if name:
                    if name not in categories_map:
                        categories_map[name] = []
                    if desc and desc not in categories_map[name]:
                        categories_map[name].append(desc)

        # Collect from all tiers
        process_slots("tier_low", 6)
        process_slots("tier_mid", 6)
        process_slots("tier_high", 6)

        # Flatten descriptions
        final_cat_map = {}
        for k, v in categories_map.items():
            final_cat_map[k] = " / ".join(v)
            
        # Build prompt text
        cat_lines = []
        for name, desc in final_cat_map.items():
            cat_lines.append(f'- "{name}": {desc}')
        cat_section = "\n".join(cat_lines)
        valid_cats_json = json.dumps(list(final_cat_map), ensure_ascii=False)
        
        # --- Format conversation context ---
        context_section = ""
        if contexts:
            context_lines = [f"[{msg['role']}]: {msg['content']}" for msg in contexts]
            context_section = "\n\nRecent Conversation Context:\n" + "\n".join(context_lines)
        
        # --- Build task snapshots section ---
        snapshots_section = ""
        if task_snapshots:
            snapshot_lines = ["=== ACTIVE TASK SNAPSHOTS ==="]
            for snap in task_snapshots:
                snapshot_lines.append(f"[{snap['id']}] Category={snap['category']}, Score={snap['score']}, Summary=\"{snap['summary']}\"")
            snapshot_lines.append("(Use continued_task_id to reference a task when context_relation is 'continue' or 'downgrade')")
            snapshots_section = "\n".join(snapshot_lines)
        else:
            snapshots_section = "=== ACTIVE TASK SNAPSHOTS ===\n(No active tasks)"
        
        # Check template
        template = router_config.get("router_manual_prompt", "")
        if template and "{categories}" in template:
             system_prompt = template.replace("{categories}", cat_section)
        elif template:
             system_prompt = template
        else:
             system_prompt = f"""You are a Model Router. Analyze user input and output JSON.

Output format:
{{"difficulty_score": 1-9, "category": String, "context_relation": "continue"|"downgrade"|"unrelated", "continued_task_id": String|null, "reasoning": "Brief"}}

Categories: {valid_cats_json}
{cat_section}

{{task_snapshots}}

=== CONTEXT RELATION RULES ===
1. **continue** - Direct continuation of prior task → difficulty_score MUST match that task's score
2. **downgrade** - Related but simpler/closure → Re-evaluate difficulty
3. **unrelated** - New topic or chit-chat → Evaluate based on current input only

=== MULTI-DIMENSIONAL DIFFICULTY SCALE (1-9) ===

[Code/Architecture - code]
- 1-2: Syntax questions, simple scripts, single function
- 3-4: Algorithm implementation, debugging, standard API calls
- 5-6: Multi-file refactoring, basic architecture, standard projects
- 7-8: Distributed systems, microservices, high-concurrency design
- 9: Million-level concurrency, financial-grade systems, TCC/Saga transactions

[Math/Reasoning - math]
- 1-2: Arithmetic, unit conversion, simple formulas
- 3-4: Algebra, geometry proofs, probability
- 5-6: Calculus, linear algebra, statistical analysis
- 7-8: Multi-variable optimization, PDEs, number theory
- 9: Frontier math problems, complex proofs, research-level

[Roleplay - roleplay]
- 1-2: Simple greetings, fixed responses
- 3-4: Basic dialogue, single-scene interaction
- 5-6: Complex plots, multi-character coordination
- 7-8: Deep characterization, emotional nuance, long-term memory
- 9: Professional-level creation, world-building

[General Chat - chat]
- 1-2: Greetings, thanks, simple confirmations
- 3-4: Knowledge Q&A, concept explanations
- 5-6: Deep discussions, opinion analysis, long responses
- 7-8: Cross-domain synthesis, professional consulting
- 9: Complex decision support, multi-dimensional analysis

[Custom Categories]
For user-defined categories, follow general principles:
- 1-3: Simple, single-step, standardized
- 4-6: Medium complexity, requires synthesis
- 7-9: High complexity, cross-domain, frontier problems

=== KEY RULES ===
1. First determine category, then score based on that dimension's standards
2. Keywords like "million-level", "high-concurrency", "distributed" usually mean 7-9
3. When context_relation is "continue", difficulty_score MUST match the continued task's score
4. Pure chit-chat, greetings, thanks should be 1-2
"""

        # Replace only named placeholders, not JSON braces in custom prompts.
        system_prompt = system_prompt.replace("{valid_cats_json}", valid_cats_json)

        # 替换任务快照占位符
        if "{task_snapshots}" in system_prompt:
            system_prompt = system_prompt.replace("{task_snapshots}", snapshots_section)
        else:
            # 兜底：追加快照信息
            system_prompt += f"\n\n{snapshots_section}"

        prompt = f"{context_section}\n\nCurrent User Input: {user_text}\n\nOutput JSON object."
        
        try:
            logger.debug("Sending request to Router Model...")
            response = await provider.text_chat(
                prompt=f"{system_prompt}\n\n{prompt}",
                contexts=[]
            )
            
            raw_text = response.completion_text.strip()
            logger.debug(f"Router Raw Output: {raw_text}")
            
            # Basic cleanup
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"): lines = lines[1:]
                if lines[-1].startswith("```"): lines = lines[:-1]
                raw_text = "\n".join(lines)
            
            data = json.loads(raw_text)
            
            # --- 缓存结果 ---
            self._set_cached(cache_key, data)
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Router JSON Parse Error: {e}. Raw: {raw_text}")
            return None
        except Exception as e:
            logger.error(f"Router Unexpected Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

