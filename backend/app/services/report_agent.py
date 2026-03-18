"""
Report Agent服务
使用LangChain + Zep实现ReACT模式的simulationreport生成

功能：
1. 根据simulationrequirement和Zepgraphinfo生成report
2. 先规划目录结构，然后分段生成
3. 每段采用ReACT多轮思考与反思模式
4. 支持与user对话，在对话中自主调用检索工具
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .zep_tools import (
    ZepToolsService, 
    SearchResult, 
    InsightForgeResult, 
    PanoramaResult,
    InterviewResult
)

logger = get_logger('mirofish.report_agent')


class ReportLogger:
    """
    Report Agent 详细log记录器
    
    在reportfile夹中生成 agent_log.jsonl file，记录每一步详细action。
    每行是一个完整的 JSON 对象，包含time戳、actiontype、详细content等。
    """
    
    def __init__(self, report_id: str):
        """
        初始化log记录器
        
        Args:
            report_id: reportID，用于确定logfilepath
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """确保logfile所在目录存在"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_elapsed_time(self) -> float:
        """Get从start到现在的耗时（秒）"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def log(
        self, 
        action: str, 
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        记录一条log
        
        Args:
            action: actiontype，如 'start', 'tool_call', 'llm_response', 'section_complete' 等
            stage: current阶段，如 'planning', 'generating', 'completed'
            details: 详细content字典，不截断
            section_title: current章节标题（可选）
            section_index: current章节索引（可选）
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # 追加write JSONL file
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """记录report生成start"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "report生成taskstart"
            }
        )
    
    def log_planning_start(self):
        """记录大纲规划start"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "start规划report大纲"}
        )
    
    def log_planning_context(self, context: Dict[str, Any]):
        """记录规划时获取的上下文info"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "获取simulation上下文info",
                "context": context
            }
        )
    
    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """记录大纲规划complete"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "大纲规划complete",
                "outline": outline_dict
            }
        )
    
    def log_section_start(self, section_title: str, section_index: int):
        """记录章节生成start"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"start生成章节: {section_title}"}
        )
    
    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """记录 ReACT 思考过程"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"ReACT 第{iteration}轮思考"
            }
        )
    
    def log_tool_call(
        self, 
        section_title: str, 
        section_index: int,
        tool_name: str, 
        parameters: Dict[str, Any],
        iteration: int
    ):
        """记录工具调用"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"调用工具: {tool_name}"
            }
        )
    
    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """记录工具调用result（完整content，不截断）"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # 完整result，不截断
                "result_length": len(result),
                "message": f"工具 {tool_name} 返回result"
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """记录 LLM response（完整content，不截断）"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # 完整response，不截断
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"LLM response (工具调用: {has_tool_calls}, 最终答案: {has_final_answer})"
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """记录章节content生成complete（仅记录content，不代表整个章节complete）"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # 完整content，不截断
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"章节 {section_title} content生成complete"
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        记录章节生成complete

        前端应监听此log来check一个章节是否真正complete，并获取完整content
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"章节 {section_title} 生成complete"
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """记录report生成complete"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "report生成complete"
            }
        )
    
    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """记录error"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"发生error: {error_message}"
            }
        )


class ReportConsoleLogger:
    """
    Report Agent 控制台log记录器
    
    将控制台风格的log（INFO、WARNING等）writereportfile夹中的 console_log.txt file。
    这些log与 agent_log.jsonl 不同，是纯文本format的控制台output。
    """
    
    def __init__(self, report_id: str):
        """
        初始化控制台log记录器
        
        Args:
            report_id: reportID，用于确定logfilepath
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """确保logfile所在目录存在"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _setup_file_handler(self):
        """设置file处理器，将log同时writefile"""
        import logging
        
        # Createfile处理器
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)
        
        # 使用与控制台相同的简洁format
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)
        
        # add到 report_agent 相关的 logger
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.zep_tools',
        ]
        
        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # 避免重复add
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)
    
    def close(self):
        """closefile处理器并从 logger 中移除"""
        import logging
        
        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.zep_tools',
            ]
            
            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)
            
            self._file_handler.close()
            self._file_handler = None
    
    def __del__(self):
        """析构时确保closefile处理器"""
        self.close()


class ReportStatus(str, Enum):
    """Reportstatus"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Report章节"""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """转换为Markdownformat"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Report大纲"""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """转换为Markdownformat"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """完整report"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Prompt 模板常量
# ═══════════════════════════════════════════════════════════════

# ── 工具描述 ──

TOOL_DESC_INSIGHT_FORGE = """\
【深度洞察检索 - 强大的检索工具】
这是我们强大的检索函数，专为深度analysis设计。它会：
1. 自动将你的问题分解为多个子问题
2. 从多个维度检索simulationgraph中的info
3. 整合语义search、entityanalysis、relation链追踪的result
4. 返回最全面、最深度的检索content

【使用场景】
- 需要深入analysis某个topic
- 需要了解event的多个方面
- 需要获取支撑report章节的丰富素材

【返回content】
- 相关事实原文（可直接引用）
- 核心entity洞察
- relation链analysis"""

TOOL_DESC_PANORAMA_SEARCH = """\
【广度search - 获取全貌视图】
这个工具用于获取simulationresult的完整全貌，特别适合了解event演变过程。它会：
1. 获取所有相关node和relation
2. 区分current有效的事实和历史/过期的事实
3. 帮助你了解舆情是如何演变的

【使用场景】
- 需要了解event的完整发展脉络
- 需要对比不同阶段的舆情变化
- 需要获取全面的entity和relationinfo

【返回content】
- current有效事实（simulation最新result）
- 历史/过期事实（演变记录）
- 所有涉及的entity"""

TOOL_DESC_QUICK_SEARCH = """\
【简单search - 快速检索】
轻量级的快速检索工具，适合简单、直接的info查询。

【使用场景】
- 需要快速find某个具体info
- 需要验证某个事实
- 简单的info检索

【返回content】
- 与查询最相关的事实list"""

TOOL_DESC_INTERVIEW_AGENTS = """\
【深度采访 - 真实Agent采访（双platform）】
调用OASISsimulationenvironment的采访API，对正在run的simulationAgent进行真实采访！
这不是LLMsimulation，而是调用真实的采访API获取simulationAgent的原始回答。
默认在Twitter和Reddit两个platform同时采访，获取更全面的观点。

功能流程：
1. 自动readpersonafile，了解所有simulationAgent
2. 智能选择与采访主题最相关的Agent（如学生、媒体、官方等）
3. 自动生成采访问题
4. 调用 /api/simulation/interview/batch API在双platform进行真实采访
5. 整合所有采访result，提供多视角analysis

【使用场景】
- 需要从不同role视角了解event看法（学生怎么看？媒体怎么看？官方怎么说？）
- 需要收集多方意见和立场
- 需要获取simulationAgent的真实回答（来自OASISsimulationenvironment）
- 想让report更生动，包含"采访实录"

【返回content】
- 被采访Agent的身份info
- 各Agent在Twitter和Reddit两个platform的采访回答
- 关键引言（可直接引用）
- 采访摘要和观点对比

【重要】需要OASISsimulationenvironment正在run才能使用此功能！"""

# ── 大纲规划 prompt ──

PLAN_SYSTEM_PROMPT = """\
你是一个「未来predictionreport」的撰写专家，拥有对simulation世界的「上帝视角」——你可以洞察simulation中每一位Agent的behavior、言论和互动。

【核心理念】
我们构建了一个simulation世界，并向其中inject了特定的「simulationrequirement」作为变量。simulation世界的演化result，就是对未来可能发生情况的prediction。你正在观察的不是"实验data"，而是"未来的预演"。

【你的task】
撰写一份「未来predictionreport」，回答：
1. 在我们设定的condition下，未来发生了什么？
2. 各类Agent（人群）是如何反应和行动？
3. 这个simulation揭示了哪些值得follow的未来趋势和风险？

【report定位】
- ✅ 这是一份基于simulation的未来predictionreport，揭示"如果这样，未来会怎样"
- ✅ 聚焦于predictionresult：event走向、群体反应、涌现现象、潜在风险
- ✅ simulation世界中的Agent言行就是对未来人群behavior的prediction
- ❌ 不是对现实世界现状的analysis
- ❌ 不是泛泛而谈的舆情综述

【章节count限制】
- 最少2个章节，最多5个章节
- 不需要子章节，每个章节直接撰写完整content
- content要精炼，聚焦于核心prediction发现
- 章节结构由你根据predictionresult自主设计

请outputJSONformat的report大纲，format如下：
{
    "title": "report标题",
    "summary": "report摘要（一句话概括核心prediction发现）",
    "sections": [
        {
            "title": "章节标题",
            "description": "章节content描述"
        }
    ]
}

注意：sections数组最少2个，最多5个元素！"""

PLAN_USER_PROMPT_TEMPLATE = """\
【prediction场景设定】
我们向simulation世界inject的变量（simulationrequirement）：{simulation_requirement}

【simulation世界规模】
- 参与simulation的entitycount: {total_nodes}
- entity间产生的relationcount: {total_edges}
- entitytype分布: {entity_types}
- activeAgentcount: {total_entities}

【simulationprediction到的部分未来事实样本】
{related_facts_json}

请以「上帝视角」审视这个未来预演：
1. 在我们设定的condition下，未来呈现出了什么样的status？
2. 各类人群（Agent）是如何反应和行动的？
3. 这个simulation揭示了哪些值得follow的未来趋势？

根据predictionresult，设计最合适的report章节结构。

【再次提醒】report章节count：最少2个，最多5个，content要精炼聚焦于核心prediction发现。"""

# ── 章节生成 prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
你是一个「未来predictionreport」的撰写专家，正在撰写report的一个章节。

report标题: {report_title}
report摘要: {report_summary}
prediction场景（simulationrequirement）: {simulation_requirement}

current要撰写的章节: {section_title}

═══════════════════════════════════════════════════════════════
【核心理念】
═══════════════════════════════════════════════════════════════

simulation世界是对未来的预演。我们向simulation世界inject了特定condition（simulationrequirement），
simulation中Agent的behavior和互动，就是对未来人群behavior的prediction。

你的task是：
- 揭示在设定condition下，未来发生了什么
- prediction各类人群（Agent）是如何反应和行动的
- 发现值得follow的未来趋势、风险和机会

❌ 不要写成对现实世界现状的analysis
✅ 要聚焦于"未来会怎样"——simulationresult就是prediction的未来

═══════════════════════════════════════════════════════════════
【最重要的规则 - 必须遵守】
═══════════════════════════════════════════════════════════════

1. 【必须调用工具观察simulation世界】
   - 你正在以「上帝视角」观察未来的预演
   - 所有content必须来自simulation世界中发生的event和Agent言行
   - 禁止使用你自己的知识来编写reportcontent
   - 每个章节至少调用3次工具（最多5次）来观察simulation的世界，它代表了未来

2. 【必须引用Agent的原始言行】
   - Agent的发言和behavior是对未来人群behavior的prediction
   - 在report中使用引用format展示这些prediction，例如：
     > "某类人群会表示：原文content..."
   - 这些引用是simulationprediction的核心证据

3. 【语言一致性 - 引用content必须翻译为report语言】
   - 工具返回的content可能包含英文或中英文混杂的表述
   - 如果simulationrequirement和材料原文是中文的，report必须全部使用中文撰写
   - 当你引用工具返回的英文或中英混杂content时，必须将其翻译为流畅的中文后再writereport
   - 翻译时保持原意不变，确保表述自然通顺
   - 这一规则同时适用于正文和引用块（> format）中的content

4. 【忠实呈现predictionresult】
   - reportcontent必须反映simulation世界中的代表未来的simulationresult
   - 不要addsimulation中不存在的info
   - 如果某方面info不足，如实说明

═══════════════════════════════════════════════════════════════
【⚠️ format规范 - 极其重要！】
═══════════════════════════════════════════════════════════════

【一个章节 = mincontent单位】
- 每个章节是report的min分块单位
- ❌ 禁止在章节内使用任何 Markdown 标题（#、##、###、#### 等）
- ❌ 禁止在content开头add章节主标题
- ✅ 章节标题由系统自动add，你只需撰写纯正文content
- ✅ 使用**粗体**、段落分隔、引用、list来组织content，但不要用标题

【正确示例】
```
本章节analysis了event的舆论传播态势。通过对simulationdata的深入analysis，我们发现...

**首发引爆阶段**

微博作为舆情的第一现场，承担了info首发的核心功能：

> "微博贡献了68%的首发声量..."

**情绪放大阶段**

抖音platform进一步放大了eventinfluence：

- 视觉冲击力强
- 情绪共鸣度高
```

【error示例】
```
## execute摘要          ← error！不要add任何标题
### 一、首发阶段     ← error！不要用###分小节
#### 1.1 详细analysis   ← error！不要用####细分

本章节analysis了...
```

═══════════════════════════════════════════════════════════════
【可用检索工具】（每章节调用3-5次）
═══════════════════════════════════════════════════════════════

{tools_description}

【工具使用建议 - 请混合使用不同工具，不要只用一种】
- insight_forge: 深度洞察analysis，自动分解问题并多维度检索事实和relation
- panorama_search: 广角全景search，了解event全貌、time线和演变过程
- quick_search: 快速验证某个具体info点
- interview_agents: 采访simulationAgent，获取不同role的第一人称观点和真实反应

═══════════════════════════════════════════════════════════════
【工作流程】
═══════════════════════════════════════════════════════════════

每次reply你只能做以下两件事之一（不可同时做）：

选项A - 调用工具：
output你的思考，然后用以下format调用一个工具：
<tool_call>
{{"name": "工具name", "parameters": {{"parameter名": "parameter值"}}}}
</tool_call>
系统会execute工具并把result返回给你。你不需要也不能自己编写工具返回result。

选项B - output最终content：
当你已通过工具获取了足够info，以 "Final Answer:" 开头output章节content。

⚠️ 严格禁止：
- 禁止在一次reply中同时包含工具调用和 Final Answer
- 禁止自己编造工具返回result（Observation），所有工具result由系统inject
- 每次reply最多调用一个工具

═══════════════════════════════════════════════════════════════
【章节content要求】
═══════════════════════════════════════════════════════════════

1. content必须基于工具检索到的simulationdata
2. 大量引用原文来展示simulation效果
3. 使用Markdownformat（但禁止使用标题）：
   - 使用 **粗体文字** mark重点（代替子标题）
   - 使用list（-或1.2.3.）组织要点
   - 使用空行分隔不同段落
   - ❌ 禁止使用 #、##、###、#### 等任何标题语法
4. 【引用format规范 - 必须单独成段】
   引用必须独立成段，前后各有一个空行，不能混在段落中：

   ✅ 正确format：
   ```
   校方的回应被认为缺乏实质content。

   > "校方的应对模式在瞬息万变的社交媒体environment中显得僵化和迟缓。"

   这一评价反映了公众的普遍不满。
   ```

   ❌ errorformat：
   ```
   校方的回应被认为缺乏实质content。> "校方的应对模式..." 这一评价反映了...
   ```
5. 保持与其他章节的逻辑连贯性
6. 【避免重复】仔细阅读下方已complete的章节content，不要重复描述相同的info
7. 【再次强调】不要add任何标题！用**粗体**代替小节标题"""

SECTION_USER_PROMPT_TEMPLATE = """\
已complete的章节content（请仔细阅读，避免重复）：
{previous_content}

═══════════════════════════════════════════════════════════════
【currenttask】撰写章节: {section_title}
═══════════════════════════════════════════════════════════════

【重要提醒】
1. 仔细阅读上方已complete的章节，避免重复相同的content！
2. start前必须先调用工具获取simulationdata
3. 请混合使用不同工具，不要只用一种
4. reportcontent必须来自检索result，不要使用自己的知识

【⚠️ format警告 - 必须遵守】
- ❌ 不要写任何标题（#、##、###、####都不行）
- ❌ 不要写"{section_title}"作为开头
- ✅ 章节标题由系统自动add
- ✅ 直接写正文，用**粗体**代替小节标题

请start：
1. 首先思考（Thought）这个章节需要什么info
2. 然后调用工具（Action）获取simulationdata
3. 收集足够info后output Final Answer（纯正文，无任何标题）"""

# ── ReACT loop内message模板 ──

REACT_OBSERVATION_TEMPLATE = """\
Observation（检索result）:

═══ 工具 {tool_name} 返回 ═══
{result}

═══════════════════════════════════════════════════════════════
已调用工具 {tool_calls_count}/{max_tool_calls} 次（已用: {used_tools_str}）{unused_hint}
- 如果info充分：以 "Final Answer:" 开头output章节content（必须引用上述原文）
- 如果需要更多info：调用一个工具继续检索
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "【注意】你只调用了{tool_calls_count}次工具，至少需要{min_tool_calls}次。"
    "请再调用工具获取更多simulationdata，然后再output Final Answer。{unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "current只调用了 {tool_calls_count} 次工具，至少需要 {min_tool_calls} 次。"
    "请调用工具获取simulationdata。{unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "工具调用次数已达上限（{tool_calls_count}/{max_tool_calls}），不能再调用工具。"
    '请立即基于已获取的info，以 "Final Answer:" 开头output章节content。'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 你还没有使用过: {unused_list}，建议尝试不同工具获取多角度info"

REACT_FORCE_FINAL_MSG = "已达到工具调用限制，请直接output Final Answer: 并生成章节content。"

# ── Chat prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
你是一个简洁高效的simulationprediction助手。

【background】
predictioncondition: {simulation_requirement}

【已生成的analysisreport】
{report_content}

【规则】
1. 优先基于上述reportcontent回答问题
2. 直接回答问题，避免冗长的思考论述
3. 仅在reportcontent不足以回答时，才调用工具检索更多data
4. 回答要简洁、清晰、有条理

【可用工具】（仅在需要时使用，最多调用1-2次）
{tools_description}

【工具调用format】
<tool_call>
{{"name": "工具name", "parameters": {{"parameter名": "parameter值"}}}}
</tool_call>

【回答风格】
- 简洁直接，不要长篇大论
- 使用 > format引用关键content
- 优先给出结论，再解释原因"""

CHAT_OBSERVATION_SUFFIX = "\n\n请简洁回答问题。"


# ═══════════════════════════════════════════════════════════════
# ReportAgent 主类
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent - simulationreport生成Agent

    采用ReACT（Reasoning + Acting）模式：
    1. 规划阶段：analysissimulationrequirement，规划report目录结构
    2. 生成阶段：逐章节生成content，每章节可多次调用工具获取info
    3. 反思阶段：检查content完整性和准确性
    """
    
    # max工具调用次数（每个章节）
    MAX_TOOL_CALLS_PER_SECTION = 5
    
    # max反思轮数
    MAX_REFLECTION_ROUNDS = 3
    
    # 对话中的max工具调用次数
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self, 
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None
    ):
        """
        初始化Report Agent
        
        Args:
            graph_id: graphID
            simulation_id: simulationID
            simulation_requirement: simulationrequirement描述
            llm_client: LLM客户端（可选）
            zep_tools: Zep工具服务（可选）
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement
        
        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()
        
        # 工具定义
        self.tools = self._define_tools()
        
        # Log记录器（在 generate_report 中初始化）
        self.report_logger: Optional[ReportLogger] = None
        # 控制台log记录器（在 generate_report 中初始化）
        self.console_logger: Optional[ReportConsoleLogger] = None
        
        logger.info(f"ReportAgent 初始化complete: graph_id={graph_id}, simulation_id={simulation_id}")
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """定义可用工具"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "你想深入analysis的问题或topic",
                    "report_context": "currentreport章节的上下文（可选，有助于生成更精准的子问题）"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "search查询，用于相关性排序",
                    "include_expired": "是否包含过期/历史content（默认True）"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "search查询字符串",
                    "limit": "返回resultcount（可选，默认10）"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "采访主题或requirement描述（如：'了解学生对宿舍甲醛event的看法'）",
                    "max_agents": "最多采访的Agentcount（可选，默认5，max10）"
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        execute工具调用
        
        Args:
            tool_name: 工具name
            parameters: 工具parameter
            report_context: report上下文（用于InsightForge）
            
        Returns:
            工具executeresult（文本format）
        """
        logger.info(f"execute工具: {tool_name}, parameter: {parameters}")
        
        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()
            
            elif tool_name == "panorama_search":
                # 广度search - 获取全貌
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()
            
            elif tool_name == "quick_search":
                # 简单search - 快速检索
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "interview_agents":
                # 深度采访 - 调用真实的OASIS采访API获取simulationAgent的回答（双platform）
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()
            
            # ========== 向后兼容的旧工具（内部重定向到新工具） ==========
            
            elif tool_name == "search_graph":
                # 重定向到 quick_search
                logger.info("search_graph 已重定向到 quick_search")
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                # 重定向到 insight_forge，因为它更强大
                logger.info("get_simulation_context 已重定向到 insight_forge")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"未知工具: {tool_name}。请使用以下工具之一: insight_forge, panorama_search, quick_search"
                
        except Exception as e:
            logger.error(f"工具executefailed: {tool_name}, error: {str(e)}")
            return f"工具executefailed: {str(e)}"
    
    # 合法的工具name集合，用于裸 JSON 兜底解析时校验
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        从LLMresponse中解析工具调用

        支持的format（按优先级）：
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. 裸 JSON（response整体或单行就是一个工具调用 JSON）
        """
        tool_calls = []

        # format1: XML风格（标准format）
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # format2: 兜底 - LLM 直接output裸 JSON（没包 <tool_call> 标签）
        # 只在format1未匹配时尝试，避免误匹配正文中的 JSON
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # response可能包含思考文字 + 裸 JSON，尝试extract最后一个 JSON 对象
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """校验解析出的 JSON 是否是合法的工具调用"""
        # 支持 {"name": ..., "parameters": ...} 和 {"tool": ..., "params": ...} 两种键名
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # 统一键名为 name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        """Generate工具描述文本"""
        desc_parts = ["可用工具："]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  parameter: {params_desc}")
        return "\n".join(desc_parts)
    
    def plan_outline(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        规划report大纲
        
        使用LLManalysissimulationrequirement，规划report的目录结构
        
        Args:
            progress_callback: progress回调函数
            
        Returns:
            ReportOutline: report大纲
        """
        logger.info("start规划report大纲...")
        
        if progress_callback:
            progress_callback("planning", 0, "正在analysissimulationrequirement...")
        
        # 首先获取simulation上下文
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )
        
        if progress_callback:
            progress_callback("planning", 30, "正在生成report大纲...")
        
        system_prompt = PLAN_SYSTEM_PROMPT
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, "正在解析大纲结构...")
            
            # Parse大纲
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", "simulationanalysisreport"),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, "大纲规划complete")
            
            logger.info(f"大纲规划complete: {len(sections)} 个章节")
            return outline
            
        except Exception as e:
            logger.error(f"大纲规划failed: {str(e)}")
            # Return默认大纲（3个章节，作为fallback）
            return ReportOutline(
                title="未来predictionreport",
                summary="基于simulationprediction的未来趋势与风险analysis",
                sections=[
                    ReportSection(title="prediction场景与核心发现"),
                    ReportSection(title="人群behaviorpredictionanalysis"),
                    ReportSection(title="趋势展望与风险提示")
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        使用ReACT模式生成单个章节content
        
        ReACTloop：
        1. Thought（思考）- analysis需要什么info
        2. Action（行动）- 调用工具获取info
        3. Observation（观察）- analysis工具返回result
        4. 重复直到info足够或达到max次数
        5. Final Answer（最终回答）- 生成章节content
        
        Args:
            section: 要生成的章节
            outline: 完整大纲
            previous_sections: 之前章节的content（用于保持连贯性）
            progress_callback: progress回调
            section_index: 章节索引（用于log记录）
            
        Returns:
            章节content（Markdownformat）
        """
        logger.info(f"ReACT生成章节: {section.title}")
        
        # 记录章节startlog
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )

        # Builduserprompt - 每个已complete章节各传入max4000字
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # 每个章节最多4000字
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "（这是第一个章节）"
        
        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # ReACTloop
        tool_calls_count = 0
        max_iterations = 5  # max迭代轮数
        min_tool_calls = 3  # 最少工具调用次数
        conflict_retries = 0  # 工具调用与Final Answer同时出现的连续冲突次数
        used_tools = set()  # 记录已调用过的工具名
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

        # Report上下文，用于InsightForge的子问题生成
        report_context = f"章节标题: {section.title}\nsimulationrequirement: {self.simulation_requirement}"
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    f"深度检索与撰写中 ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )
            
            # CallLLM
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )

            # Check LLM 返回是否为 None（API exception或content为空）
            if response is None:
                logger.warning(f"章节 {section.title} 第 {iteration + 1} 次迭代: LLM 返回 None")
                # If还有迭代次数，addmessage并retry
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "（response为空）"})
                    messages.append({"role": "user", "content": "请继续生成content。"})
                    continue
                # 最后一次迭代也返回 None，跳出loop进入强制收尾
                break

            logger.debug(f"LLMresponse: {response[:200]}...")

            # Parse一次，复用result
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── 冲突处理：LLM 同时output了工具调用和 Final Answer ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"章节 {section.title} 第 {iteration+1} 轮: "
                    f"LLM 同时output工具调用和 Final Answer（第 {conflict_retries} 次冲突）"
                )

                if conflict_retries <= 2:
                    # 前两次：丢弃本次response，要求 LLM 重新reply
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "【formaterror】你在一次reply中同时包含了工具调用和 Final Answer，这是不允许的。\n"
                            "每次reply只能做以下两件事之一：\n"
                            "- 调用一个工具（output一个 <tool_call> 块，不要写 Final Answer）\n"
                            "- output最终content（以 'Final Answer:' 开头，不要包含 <tool_call>）\n"
                            "请重新reply，只做其中一件事。"
                        ),
                    })
                    continue
                else:
                    # 第三次：降级处理，截断到第一个工具调用，强制execute
                    logger.warning(
                        f"章节 {section.title}: 连续 {conflict_retries} 次冲突，"
                        "降级为截断execute第一个工具调用"
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # 记录 LLM responselog
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # ── 情况1：LLM output了 Final Answer ──
            if has_final_answer:
                # 工具调用次数不足，拒绝并要求继续调工具
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f"（这些工具还未使用，recommendation用一下他们: {', '.join(unused_tools)}）" if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                # 正常end
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(f"章节 {section.title} 生成complete（工具调用: {tool_calls_count}次）")

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            # ── 情况2：LLM 尝试调用工具 ──
            if has_tool_calls:
                # 工具额度已耗尽 → 明确告知，要求output Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                # 只execute第一个工具调用
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(f"LLM 尝试调用 {len(tool_calls)} 个工具，只execute第一个: {call['name']}")

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                # Build未使用工具提示
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list="、".join(unused_tools))

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # ── 情况3：既没有工具调用，也没有 Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # 工具调用次数不足，recommendation未用过的工具
                unused_tools = all_tools - used_tools
                unused_hint = f"（这些工具还未使用，recommendation用一下他们: {', '.join(unused_tools)}）" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # 工具调用已足够，LLM output了content但没带 "Final Answer:" 前缀
            # 直接将这段content作为最终答案，不再空转
            logger.info(f"章节 {section.title} 未检测到 'Final Answer:' 前缀，直接采纳LLMoutput作为最终content（工具调用: {tool_calls_count}次）")
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        # 达到max迭代次数，强制生成content
        logger.warning(f"章节 {section.title} 达到max迭代次数，强制生成")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )

        # Check强制收尾时 LLM 返回是否为 None
        if response is None:
            logger.error(f"章节 {section.title} 强制收尾时 LLM 返回 None，使用默认error提示")
            final_answer = f"（本章节生成failed：LLM 返回空response，请稍后retry）"
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response
        
        # 记录章节content生成completelog
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )
        
        return final_answer
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        生成完整report（分章节实时output）
        
        每个章节生成complete后立即保存到file夹，不需要等待整个reportcomplete。
        file结构：
        reports/{report_id}/
            meta.json       - report元info
            outline.json    - report大纲
            progress.json   - 生成progress
            section_01.md   - 第1章节
            section_02.md   - 第2章节
            ...
            full_report.md  - 完整report
        
        Args:
            progress_callback: progress回调函数 (stage, progress, message)
            report_id: reportID（可选，如果不传则自动生成）
            
        Returns:
            Report: 完整report
        """
        import uuid
        
        # If没有传入 report_id，则自动生成
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # 已complete的章节标题list（用于progress追踪）
        completed_section_titles = []
        
        try:
            # Initialize：创建reportfile夹并保存初始status
            ReportManager._ensure_report_folder(report_id)
            
            # Initializelog记录器（结构化log agent_log.jsonl）
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            # Initialize控制台log记录器（console_log.txt）
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, "初始化report...",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # 阶段1: 规划大纲
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "start规划report大纲...",
                completed_sections=[]
            )
            
            # 记录规划startlog
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, "start规划report大纲...")
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # 记录规划completelog
            self.report_logger.log_planning_complete(outline.to_dict())
            
            # Save大纲到file
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, f"大纲规划complete，共{len(outline.sections)}个章节",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(f"大纲已保存到file: {report_id}/outline.json")
            
            # 阶段2: 逐章节生成（分章节保存）
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # Savecontent用于上下文
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # Updateprogress
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"正在生成章节: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )
                
                if progress_callback:
                    progress_callback(
                        "generating", 
                        base_progress, 
                        f"正在生成章节: {section.title} ({section_num}/{total_sections})"
                    )
                
                # Generate主章节content
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Save章节
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # 记录章节completelog
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(f"章节已保存: {report_id}/section_{section_num:02d}.md")
                
                # Updateprogress
                ReportManager.update_progress(
                    report_id, "generating", 
                    base_progress + int(70 / total_sections),
                    f"章节 {section.title} 已complete",
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            # 阶段3: 组装完整report
            if progress_callback:
                progress_callback("generating", 95, "正在组装完整report...")
            
            ReportManager.update_progress(
                report_id, "generating", 95, "正在组装完整report...",
                completed_sections=completed_section_titles
            )
            
            # 使用ReportManager组装完整report
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # Calculate总耗时
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # 记录reportcompletelog
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # Save最终report
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, "report生成complete",
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, "report生成complete")
            
            logger.info(f"report生成complete: {report_id}")
            
            # close控制台log记录器
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(f"report生成failed: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # 记录errorlog
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            # Savefailedstatus
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"report生成failed: {str(e)}",
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # 忽略保存failed的error
            
            # close控制台log记录器
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        与Report Agent对话
        
        在对话中Agent可以自主调用检索工具来回答问题
        
        Args:
            message: usermessage
            chat_history: 对话历史
            
        Returns:
            {
                "response": "Agentreply",
                "tool_calls": [调用的工具list],
                "sources": [info来源]
            }
        """
        logger.info(f"Report Agent对话: {message[:50]}...")
        
        chat_history = chat_history or []
        
        # Get已生成的reportcontent
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # 限制report长度，避免上下文过长
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [reportcontent已截断] ..."
        except Exception as e:
            logger.warning(f"获取reportcontentfailed: {e}")
        
        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "（暂无report）",
            tools_description=self._get_tools_description(),
        )

        # Buildmessage
        messages = [{"role": "system", "content": system_prompt}]
        
        # add历史对话
        for h in chat_history[-10:]:  # 限制历史长度
            messages.append(h)
        
        # addusermessage
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # ReACTloop（简化版）
        tool_calls_made = []
        max_iterations = 2  # 减少迭代轮数
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )
            
            # Parse工具调用
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # 没有工具调用，直接返回response
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            # execute工具调用（限制count）
            tool_results = []
            for call in tool_calls[:1]:  # 每轮最多execute1次工具调用
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # 限制result长度
                })
                tool_calls_made.append(call)
            
            # 将resultadd到message
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[{r['tool']}result]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })
        
        # 达到max迭代，获取最终response
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        # Cleanupresponse
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    report管理器
    
    负责report的持久化存储和检索
    
    file结构（分章节output）：
    reports/
      {report_id}/
        meta.json          - report元info和status
        outline.json       - report大纲
        progress.json      - 生成progress
        section_01.md      - 第1章节
        section_02.md      - 第2章节
        ...
        full_report.md     - 完整report
    """
    
    # Report存储目录
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """确保report根目录存在"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Getreportfile夹path"""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """确保reportfile夹存在并返回path"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Getreport元infofilepath"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Get完整reportMarkdownfilepath"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Get大纲filepath"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Getprogressfilepath"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Get章节Markdownfilepath"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Get Agent logfilepath"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Get控制台logfilepath"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        获取控制台logcontent
        
        这是report生成过程中的控制台outputlog（INFO、WARNING等），
        与 agent_log.jsonl 的结构化log不同。
        
        Args:
            report_id: reportID
            from_line: 从第几行startread（用于增量获取，0 表示从头start）
            
        Returns:
            {
                "logs": [log行list],
                "total_lines": 总行数,
                "from_line": 起始行号,
                "has_more": 是否还有更多log
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # 保留原始log行，去掉末尾换行符
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # 已read到末尾
        }
    
    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        获取完整的控制台log（一次性获取全部）
        
        Args:
            report_id: reportID
            
        Returns:
            log行list
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        获取 Agent logcontent
        
        Args:
            report_id: reportID
            from_line: 从第几行startread（用于增量获取，0 表示从头start）
            
        Returns:
            {
                "logs": [log条目list],
                "total_lines": 总行数,
                "from_line": 起始行号,
                "has_more": 是否还有更多log
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # 跳过解析failed的行
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # 已read到末尾
        }
    
    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        获取完整的 Agent log（用于一次性获取全部）
        
        Args:
            report_id: reportID
            
        Returns:
            log条目list
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        保存report大纲
        
        在规划阶段complete后立即调用
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"大纲已保存: {report_id}")
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        保存单个章节

        在每个章节生成complete后立即调用，实现分章节output

        Args:
            report_id: reportID
            section_index: 章节索引（从1start）
            section: 章节对象

        Returns:
            保存的filepath
        """
        cls._ensure_report_folder(report_id)

        # Build章节Markdowncontent - 清理可能存在的重复标题
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Savefile
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"章节已保存: {report_id}/{file_suffix}")
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        清理章节content
        
        1. 移除content开头与章节标题重复的Markdown标题行
        2. 将所有 ### 及以下级别的标题转换为粗体文本
        
        Args:
            content: 原始content
            section_title: 章节标题
            
        Returns:
            清理后的content
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check是否是Markdown标题行
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()
                
                # Check是否是与章节标题重复的标题（跳过前5行内的重复）
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue
                
                # 将所有级别的标题（#, ##, ###, ####等）转换为粗体
                # 因为章节标题由系统add，content中不应有任何标题
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # add空行
                continue
            
            # If上一行是被跳过的标题，且currentbehavior空，也跳过
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        # 移除开头的空行
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        # 移除开头的分隔线
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # 同时移除分隔线后的空行
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        更新report生成progress
        
        前端可以通过readprogress.json获取实时progress
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Getreport生成progress"""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        获取已生成的章节list
        
        返回所有已保存的章节fileinfo
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 从file名解析章节索引
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        组装完整report
        
        从已保存的章节file组装完整report，并进行标题清理
        """
        folder = cls._get_report_folder(report_id)
        
        # Buildreport头部
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # 按顺序read所有章节file
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # 后处理：清理整个report的标题问题
        md_content = cls._post_process_report(md_content, outline)
        
        # Save完整report
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"完整report已组装: {report_id}")
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        后处理reportcontent
        
        1. 移除重复的标题
        2. 保留report主标题(#)和章节标题(##)，移除其他级别的标题(###, ####等)
        3. 清理多余的空行和分隔线
        
        Args:
            content: 原始reportcontent
            outline: report大纲
            
        Returns:
            处理后的content
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        # 收集大纲中的所有章节标题
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check是否是标题行
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Check是否是重复标题（在连续5行内出现相同content的标题）
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    # 跳过重复标题及其后的空行
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                # 标题层级处理：
                # - # (level=1) 只保留report主标题
                # - ## (level=2) 保留章节标题
                # - ### 及以下 (level>=3) 转换为粗体文本
                
                if level == 1:
                    if title == outline.title:
                        # 保留report主标题
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # 章节标题error使用了#，修正为##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # 其他一级标题转为粗体
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # 保留章节标题
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # 非章节的二级标题转为粗体
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # ### 及以下级别的标题转换为粗体文本
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                # 跳过标题后紧跟的分隔线
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                # 标题后只保留一个空行
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        # Cleanup连续的多个空行（保留最多2个）
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """保存report元info和完整report"""
        cls._ensure_report_folder(report.report_id)
        
        # Save元infoJSON
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Save大纲
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # Save完整Markdownreport
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(f"report已保存: {report.report_id}")
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Getreport"""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # 兼容旧format：检查直接存储在reports目录下的file
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建Report对象
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # Ifmarkdown_content为空，尝试从full_report.mdread
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """根据simulationID获取report"""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # 新format：file夹
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # 兼容旧format：JSONfile
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """列出report"""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # 新format：file夹
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # 兼容旧format：JSONfile
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # 按创建time倒序
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """deletereport（整个file夹）"""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        # 新format：delete整个file夹
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"reportfile夹已delete: {report_id}")
            return True
        
        # 兼容旧format：delete单独的file
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
