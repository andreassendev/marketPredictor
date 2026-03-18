"""
simulationconfig智能生成器
使用LLM根据simulationrequirement、documentcontent、graphinfo自动生成细致的simulationparameter
实现全程自动化，无需人工设置parameter

采用分步生成策略，避免一次性生成过长content导致failed：
1. 生成timeconfig
2. 生成eventconfig
3. 分批生成Agentconfig
4. 生成platformconfig
"""

import json
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('mirofish.simulation_config')

# 中国作息timeconfig（北京time）
CHINA_TIMEZONE_CONFIG = {
    # 深夜时段（几乎无人活动）
    "dead_hours": [0, 1, 2, 3, 4, 5],
    # 早间时段（逐渐醒来）
    "morning_hours": [6, 7, 8],
    # 工作时段
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    # 晚间高峰（最active）
    "peak_hours": [19, 20, 21, 22],
    # 夜间时段（active度下降）
    "night_hours": [23],
    # active度系数
    "activity_multipliers": {
        "dead": 0.05,      # 凌晨几乎无人
        "morning": 0.4,    # 早间逐渐active
        "work": 0.7,       # 工作时段中等
        "peak": 1.5,       # 晚间高峰
        "night": 0.5       # 深夜下降
    }
}


@dataclass
class AgentActivityConfig:
    """单个Agent的活动config"""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str
    
    # active度config (0.0-1.0)
    activity_level: float = 0.5  # 整体active度
    
    # 发言frequency（每小时预期发言次数）
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0
    
    # activetime段（24小时制，0-23）
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))
    
    # response速度（对热点event的反应延迟，单位：simulation分钟）
    response_delay_min: int = 5
    response_delay_max: int = 60
    
    # sentiment倾向 (-1.0到1.0，负面到正面)
    sentiment_bias: float = 0.0
    
    # 立场（对特定topic的态度）
    stance: str = "neutral"  # supportive, opposing, neutral, observer
    
    # influenceweight（决定其发言被其他Agent看到的概率）
    influence_weight: float = 1.0


@dataclass  
class TimeSimulationConfig:
    """timesimulationconfig（基于中国人作息习惯）"""
    # Simulation总时长（simulation小时数）
    total_simulation_hours: int = 72  # Defaultsimulation72小时（3天）
    
    # 每轮代表的time（simulation分钟）- 默认60分钟（1小时），加快time流速
    minutes_per_round: int = 60
    
    # 每小时激活的Agentcount范围
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20
    
    # 高峰时段（晚间19-22点，中国人最active的time）
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5
    
    # 低谷时段（凌晨0-5点，几乎无人活动）
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05  # 凌晨active度极低
    
    # 早间时段
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4
    
    # 工作时段
    work_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """eventconfig"""
    # 初始event（simulationstart时的触发event）
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)
    
    # 定时event（在特定time触发的event）
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # 热点topic关键词
    hot_topics: List[str] = field(default_factory=list)
    
    # 舆论引导方向
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """platform特定config"""
    platform: str  # twitter or reddit
    
    # recommendationalgorithmweight
    recency_weight: float = 0.4  # time新鲜度
    popularity_weight: float = 0.3  # popularity
    relevance_weight: float = 0.3  # 相关性
    
    # 病毒传播threshold（达到多少互动后触发扩散）
    viral_threshold: int = 10
    
    # 回声室效应强度（相似观点聚集程度）
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """完整的simulationparameterconfig"""
    # 基础info
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str
    
    # timeconfig
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)
    
    # Agentconfiglist
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)
    
    # eventconfig
    event_config: EventConfig = field(default_factory=EventConfig)
    
    # platformconfig
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None
    
    # LLMconfig
    llm_model: str = ""
    llm_base_url: str = ""
    
    # Generate元data
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""  # LLM的推理说明
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    simulationconfig智能生成器
    
    使用LLManalysissimulationrequirement、documentcontent、graphentityinfo，
    自动生成最佳的simulationparameterconfig
    
    采用分步生成策略：
    1. 生成timeconfig和eventconfig（轻量级）
    2. 分批生成Agentconfig（每批10-20个）
    3. 生成platformconfig
    """
    
    # 上下文max字符数
    MAX_CONTEXT_LENGTH = 50000
    # 每批生成的Agentcount
    AGENTS_PER_BATCH = 15
    
    # 各步骤的上下文截断长度（字符数）
    TIME_CONFIG_CONTEXT_LENGTH = 10000   # timeconfig
    EVENT_CONFIG_CONTEXT_LENGTH = 8000   # eventconfig
    ENTITY_SUMMARY_LENGTH = 300          # entity摘要
    AGENT_SUMMARY_LENGTH = 300           # Agentconfig中的entity摘要
    ENTITIES_PER_TYPE_DISPLAY = 20       # 每类entity显示count
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未config")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SimulationParameters:
        """
        智能生成完整的simulationconfig（分步生成）
        
        Args:
            simulation_id: simulationID
            project_id: projectID
            graph_id: graphID
            simulation_requirement: simulationrequirement描述
            document_text: 原始documentcontent
            entities: 过滤后的entitylist
            enable_twitter: 是否启用Twitter
            enable_reddit: 是否启用Reddit
            progress_callback: progress回调函数(current_step, total_steps, message)
            
        Returns:
            SimulationParameters: 完整的simulationparameter
        """
        logger.info(f"start智能生成simulationconfig: simulation_id={simulation_id}, entity数={len(entities)}")
        
        # Calculate总步骤数
        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches  # timeconfig + eventconfig + N批Agent + platformconfig
        current_step = 0
        
        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")
        
        # 1. 构建基础上下文info
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities
        )
        
        reasoning_parts = []
        
        # ========== 步骤1: 生成timeconfig ==========
        report_progress(1, "生成timeconfig...")
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities)
        time_config = self._parse_time_config(time_config_result, num_entities)
        reasoning_parts.append(f"timeconfig: {time_config_result.get('reasoning', 'success')}")
        
        # ========== 步骤2: 生成eventconfig ==========
        report_progress(2, "生成eventconfig和热点topic...")
        event_config_result = self._generate_event_config(context, simulation_requirement, entities)
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(f"eventconfig: {event_config_result.get('reasoning', 'success')}")
        
        # ========== 步骤3-N: 分批生成Agentconfig ==========
        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]
            
            report_progress(
                3 + batch_idx,
                f"生成Agentconfig ({start_idx + 1}-{end_idx}/{len(entities)})..."
            )
            
            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement
            )
            all_agent_configs.extend(batch_configs)
        
        reasoning_parts.append(f"Agentconfig: success生成 {len(all_agent_configs)} 个")
        
        # ========== 为初始post分配发布者 Agent ==========
        logger.info("为初始post分配合适的发布者 Agent...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len([p for p in event_config.initial_posts if p.get("poster_agent_id") is not None])
        reasoning_parts.append(f"初始post分配: {assigned_count} 个post已分配发布者")
        
        # ========== 最后一步: 生成platformconfig ==========
        report_progress(total_steps, "生成platformconfig...")
        twitter_config = None
        reddit_config = None
        
        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5
            )
        
        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6
            )
        
        # Build最终parameter
        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts)
        )
        
        logger.info(f"simulationconfig生成complete: {len(params.agent_configs)} 个Agentconfig")
        
        return params
    
    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode]
    ) -> str:
        """BuildLLM上下文，截断到max长度"""
        
        # entity摘要
        entity_summary = self._summarize_entities(entities)
        
        # Build上下文
        context_parts = [
            f"## Simulationrequirement\n{simulation_requirement}",
            f"\n## entityinfo ({len(entities)}个)\n{entity_summary}",
        ]
        
        current_length = sum(len(p) for p in context_parts)
        remaining_length = self.MAX_CONTEXT_LENGTH - current_length - 500  # 留500字符余量
        
        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(document已截断)"
            context_parts.append(f"\n## 原始documentcontent\n{doc_text}")
        
        return "\n".join(context_parts)
    
    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """Generateentity摘要"""
        lines = []
        
        # 按type分组
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)
        
        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)}个)")
            # 使用config的显示count和摘要长度
            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary_preview = (e.summary[:summary_len] + "...") if len(e.summary) > summary_len else e.summary
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ... 还有 {len(type_entities) - display_count} 个")
        
        return "\n".join(lines)
    
    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """带retry的LLM调用，包含JSON修复逻辑"""
        import re
        
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)  # 每次retry降低温度
                    # 不设置max_tokens，让LLM自由发挥
                )
                
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                
                # Check是否被截断
                if finish_reason == 'length':
                    logger.warning(f"LLMoutput被截断 (attempt {attempt+1})")
                    content = self._fix_truncated_json(content)
                
                # 尝试解析JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON解析failed (attempt {attempt+1}): {str(e)[:80]}")
                    
                    # 尝试修复JSON
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed
                    
                    last_error = e
                    
            except Exception as e:
                logger.warning(f"LLM调用failed (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(2 * (attempt + 1))
        
        raise last_error or Exception("LLM调用failed")
    
    def _fix_truncated_json(self, content: str) -> str:
        """修复被截断的JSON"""
        content = content.strip()
        
        # Calculate未闭合的括号
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        # Check是否有未闭合的字符串
        if content and content[-1] not in '",}]':
            content += '"'
        
        # 闭合括号
        content += ']' * open_brackets
        content += '}' * open_braces
        
        return content
    
    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """尝试修复configJSON"""
        import re
        
        # 修复被截断的情况
        content = self._fix_truncated_json(content)
        
        # ExtractJSON部分
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()
            
            # 移除字符串中的换行符
            def fix_string(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s
            
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)
            
            try:
                return json.loads(json_str)
            except:
                # 尝试移除所有控制字符
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass
        
        return None
    
    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:
        """Generatetimeconfig"""
        # 使用config的上下文截断长度
        context_truncated = context[:self.TIME_CONFIG_CONTEXT_LENGTH]
        
        # Calculatemax允许值（80%的agent数）
        max_agents_allowed = max(1, int(num_entities * 0.9))
        
        prompt = f"""基于以下simulationrequirement，生成timesimulationconfig。

{context_truncated}

## task
请生成timeconfigJSON。

### 基本原则（仅供参考，需根据具体event和参与群体灵活调整）：
- user群体为中国人，需符合北京time作息习惯
- 凌晨0-5点几乎无人活动（active度系数0.05）
- 早上6-8点逐渐active（active度系数0.4）
- 工作time9-18点中等active（active度系数0.7）
- 晚间19-22点是高峰期（active度系数1.5）
- 23点后active度下降（active度系数0.5）
- 一般规律：凌晨低active、早间渐增、工作时段中等、晚间高峰
- **重要**：以下示例值仅供参考，你需要根据event性质、参与群体特点来调整具体时段
  - 例如：学生群体高峰可能是21-23点；媒体全天active；官方机构只在工作time
  - 例如：突发热点可能导致深夜也有讨论，off_peak_hours 可适当缩短

### ReturnJSONformat（不要markdown）

示例：
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "针对该event的timeconfig说明"
}}

字段说明：
- total_simulation_hours (int): simulation总时长，24-168小时，突发event短、持续topic长
- minutes_per_round (int): 每轮时长，30-120分钟，建议60分钟
- agents_per_hour_min (int): 每小时最少激活Agent数（取值范围: 1-{max_agents_allowed}）
- agents_per_hour_max (int): 每小时最多激活Agent数（取值范围: 1-{max_agents_allowed}）
- peak_hours (int数组): 高峰时段，根据event参与群体调整
- off_peak_hours (int数组): 低谷时段，通常深夜凌晨
- morning_hours (int数组): 早间时段
- work_hours (int数组): 工作时段
- reasoning (string): 简要说明为什么这样config"""

        system_prompt = "You are a social media simulation expert. Return pure JSON. Time config should reflect typical European/Scandinavian daily patterns."
        
        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"timeconfigLLM生成failed: {e}, 使用默认config")
            return self._get_default_time_config(num_entities)
    
    def _get_default_time_config(self, num_entities: int) -> Dict[str, Any]:
        """Get默认timeconfig（中国人作息）"""
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,  # 每轮1小时，加快time流速
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [19, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "morning_hours": [6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "reasoning": "使用默认中国人作息config（每轮1小时）"
        }
    
    def _parse_time_config(self, result: Dict[str, Any], num_entities: int) -> TimeSimulationConfig:
        """解析timeconfigresult，并验证agents_per_hour值不超过总agent数"""
        # Get原始值
        agents_per_hour_min = result.get("agents_per_hour_min", max(1, num_entities // 15))
        agents_per_hour_max = result.get("agents_per_hour_max", max(5, num_entities // 5))
        
        # Validate并修正：确保不超过总agent数
        if agents_per_hour_min > num_entities:
            logger.warning(f"agents_per_hour_min ({agents_per_hour_min}) 超过总Agent数 ({num_entities})，已修正")
            agents_per_hour_min = max(1, num_entities // 10)
        
        if agents_per_hour_max > num_entities:
            logger.warning(f"agents_per_hour_max ({agents_per_hour_max}) 超过总Agent数 ({num_entities})，已修正")
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)
        
        # 确保 min < max
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(f"agents_per_hour_min >= max，已修正为 {agents_per_hour_min}")
        
        return TimeSimulationConfig(
            total_simulation_hours=result.get("total_simulation_hours", 72),
            minutes_per_round=result.get("minutes_per_round", 60),  # Default每轮1小时
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", [19, 20, 21, 22]),
            off_peak_hours=result.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
            off_peak_activity_multiplier=0.05,  # 凌晨几乎无人
            morning_hours=result.get("morning_hours", [6, 7, 8]),
            morning_activity_multiplier=0.4,
            work_hours=result.get("work_hours", list(range(9, 19))),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5
        )
    
    def _generate_event_config(
        self, 
        context: str, 
        simulation_requirement: str,
        entities: List[EntityNode]
    ) -> Dict[str, Any]:
        """Generateeventconfig"""
        
        # Get可用的entitytypelist，供 LLM 参考
        entity_types_available = list(set(
            e.get_entity_type() or "Unknown" for e in entities
        ))
        
        # 为每种type列出代表性entityname
        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)
        
        type_info = "\n".join([
            f"- {t}: {', '.join(examples)}" 
            for t, examples in type_examples.items()
        ])
        
        # 使用config的上下文截断长度
        context_truncated = context[:self.EVENT_CONFIG_CONTEXT_LENGTH]
        
        prompt = f"""基于以下simulationrequirement，生成eventconfig。

simulationrequirement: {simulation_requirement}

{context_truncated}

## 可用entitytype及示例
{type_info}

## task
请生成eventconfigJSON：
- extract热点topic关键词
- 描述舆论发展方向
- 设计初始postcontent，**每个post必须指定 poster_type（发布者type）**

**重要**: poster_type 必须从上面的"可用entitytype"中选择，这样初始post才能分配给合适的 Agent 发布。
例如：官方声明应由 Official/University type发布，新闻由 MediaOutlet 发布，学生观点由 Student 发布。

返回JSONformat（不要markdown）：
{{
    "hot_topics": ["关键词1", "关键词2", ...],
    "narrative_direction": "<舆论发展方向描述>",
    "initial_posts": [
        {{"content": "postcontent", "poster_type": "entitytype（必须从可用type中选择）"}},
        ...
    ],
    "reasoning": "<简要说明>"
}}"""

        system_prompt = "You are an opinion analysis expert. Return pure JSON. Note: poster_type must exactly match available entity types."
        
        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"eventconfigLLM生成failed: {e}, 使用默认config")
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "使用默认config"
            }
    
    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """解析eventconfigresult"""
        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=[],
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", "")
        )
    
    def _assign_initial_post_agents(
        self,
        event_config: EventConfig,
        agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        """
        为初始post分配合适的发布者 Agent
        
        根据每个post的 poster_type 匹配最合适的 agent_id
        """
        if not event_config.initial_posts:
            return event_config
        
        # 按entitytype建立 agent 索引
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)
        
        # type映射表（处理 LLM 可能output的不同format）
        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "professor": ["professor", "expert", "teacher"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
        }
        
        # 记录每种type已使用的 agent 索引，避免重复使用同一个 agent
        used_indices: Dict[str, int] = {}
        
        updated_posts = []
        for post in event_config.initial_posts:
            poster_type = post.get("poster_type", "").lower()
            content = post.get("content", "")
            
            # 尝试找到匹配的 agent
            matched_agent_id = None
            
            # 1. 直接匹配
            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_agent_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                # 2. 使用别名匹配
                for alias_key, aliases in type_aliases.items():
                    if poster_type in aliases or alias_key == poster_type:
                        for alias in aliases:
                            if alias in agents_by_type:
                                agents = agents_by_type[alias]
                                idx = used_indices.get(alias, 0) % len(agents)
                                matched_agent_id = agents[idx].agent_id
                                used_indices[alias] = idx + 1
                                break
                    if matched_agent_id is not None:
                        break
            
            # 3. 如果仍未找到，使用influence最高的 agent
            if matched_agent_id is None:
                logger.warning(f"未找到type '{poster_type}' 的匹配 Agent，使用influence最高的 Agent")
                if agent_configs:
                    # 按influence排序，选择influence最高的
                    sorted_agents = sorted(agent_configs, key=lambda a: a.influence_weight, reverse=True)
                    matched_agent_id = sorted_agents[0].agent_id
                else:
                    matched_agent_id = 0
            
            updated_posts.append({
                "content": content,
                "poster_type": post.get("poster_type", "Unknown"),
                "poster_agent_id": matched_agent_id
            })
            
            logger.info(f"初始post分配: poster_type='{poster_type}' -> agent_id={matched_agent_id}")
        
        event_config.initial_posts = updated_posts
        return event_config
    
    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str
    ) -> List[AgentActivityConfig]:
        """分批生成Agentconfig"""
        
        # Buildentityinfo（使用config的摘要长度）
        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append({
                "agent_id": start_idx + i,
                "entity_name": e.name,
                "entity_type": e.get_entity_type() or "Unknown",
                "summary": e.summary[:summary_len] if e.summary else ""
            })
        
        prompt = f"""基于以下info，为每个entity生成社交媒体活动config。

simulationrequirement: {simulation_requirement}

## entitylist
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## task
为每个entity生成活动config，注意：
- **time符合中国人作息**：凌晨0-5点几乎不活动，晚间19-22点最active
- **官方机构**（University/GovernmentAgency）：active度低(0.1-0.3)，工作time(9-17)活动，response慢(60-240分钟)，influence高(2.5-3.0)
- **媒体**（MediaOutlet）：active度中(0.4-0.6)，全天活动(8-23)，response快(5-30分钟)，influence高(2.0-2.5)
- **个人**（Student/Person/Alumni）：active度高(0.6-0.9)，主要晚间活动(18-23)，response快(1-15分钟)，influence低(0.8-1.2)
- **公众人物/专家**：active度中(0.4-0.6)，influence中高(1.5-2.0)

返回JSONformat（不要markdown）：
{{
    "agent_configs": [
        {{
            "agent_id": <必须与input一致>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <发帖frequency>,
            "comments_per_hour": <commentfrequency>,
            "active_hours": [<active小时list，考虑中国人作息>],
            "response_delay_min": <minresponse延迟分钟>,
            "response_delay_max": <maxresponse延迟分钟>,
            "sentiment_bias": <-1.0到1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <influenceweight>
        }},
        ...
    ]
}}"""

        system_prompt = "You are a social media behavior analysis expert. Return pure JSON. Config should reflect European/Scandinavian daily patterns."
        
        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])}
        except Exception as e:
            logger.warning(f"Agentconfig批次LLM生成failed: {e}, 使用规则生成")
            llm_configs = {}
        
        # BuildAgentActivityConfig对象
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})
            
            # IfLLM没有生成，使用规则生成
            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity)
            
            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0)
            )
            configs.append(config)
        
        return configs
    
    def _generate_agent_config_by_rule(self, entity: EntityNode) -> Dict[str, Any]:
        """基于规则生成单个Agentconfig（中国人作息）"""
        entity_type = (entity.get_entity_type() or "Unknown").lower()
        
        if entity_type in ["university", "governmentagency", "ngo"]:
            # 官方机构：工作time活动，低frequency，高influence
            return {
                "activity_level": 0.2,
                "posts_per_hour": 0.1,
                "comments_per_hour": 0.05,
                "active_hours": list(range(9, 18)),  # 9:00-17:59
                "response_delay_min": 60,
                "response_delay_max": 240,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 3.0
            }
        elif entity_type in ["mediaoutlet"]:
            # 媒体：全天活动，中等frequency，高influence
            return {
                "activity_level": 0.5,
                "posts_per_hour": 0.8,
                "comments_per_hour": 0.3,
                "active_hours": list(range(7, 24)),  # 7:00-23:59
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.5
            }
        elif entity_type in ["professor", "expert", "official"]:
            # 专家/教授：工作+晚间活动，中等frequency
            return {
                "activity_level": 0.4,
                "posts_per_hour": 0.3,
                "comments_per_hour": 0.5,
                "active_hours": list(range(8, 22)),  # 8:00-21:59
                "response_delay_min": 15,
                "response_delay_max": 90,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 2.0
            }
        elif entity_type in ["student"]:
            # 学生：晚间为主，高frequency
            return {
                "activity_level": 0.8,
                "posts_per_hour": 0.6,
                "comments_per_hour": 1.5,
                "active_hours": [8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  # 上午+晚间
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8
            }
        elif entity_type in ["alumni"]:
            # 校友：晚间为主
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],  # 午休+晚间
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
        else:
            # 普通人：晚间高峰
            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.2,
                "active_hours": [9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  # 白天+晚间
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
    

