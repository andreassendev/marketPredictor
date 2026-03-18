"""
Zep检索工具服务
封装graphsearch、noderead、边查询等工具，供Report Agent使用

核心检索工具（优化后）：
1. InsightForge（深度洞察检索）- 最强大的混合检索，自动生成子问题并多维度检索
2. PanoramaSearch（广度search）- 获取全貌，包括过期content
3. QuickSearch（简单search）- 快速检索
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger('mirofish.zep_tools')


@dataclass
class SearchResult:
    """searchresult"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }
    
    def to_text(self) -> str:
        """转换为文本format，供LLM理解"""
        text_parts = [f"search查询: {self.query}", f"找到 {self.total_count} 条相关info"]
        
        if self.facts:
            text_parts.append("\n### 相关事实:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")
        
        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """nodeinfo"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }
    
    def to_text(self) -> str:
        """转换为文本format"""
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "未知type")
        return f"entity: {self.name} (type: {entity_type})\n摘要: {self.summary}"


@dataclass
class EdgeInfo:
    """边info"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # timeinfo
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }
    
    def to_text(self, include_temporal: bool = False) -> str:
        """转换为文本format"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"relation: {source} --[{self.name}]--> {target}\n事实: {self.fact}"
        
        if include_temporal:
            valid_at = self.valid_at or "未知"
            invalid_at = self.invalid_at or "至今"
            base_text += f"\n时效: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (已过期: {self.expired_at})"
        
        return base_text
    
    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        return self.expired_at is not None
    
    @property
    def is_invalid(self) -> bool:
        """是否已失效"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    深度洞察检索result (InsightForge)
    包含多个子问题的检索result，以及综合analysis
    """
    query: str
    simulation_requirement: str
    sub_queries: List[str]
    
    # 各维度检索result
    semantic_facts: List[str] = field(default_factory=list)  # 语义searchresult
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)  # entity洞察
    relationship_chains: List[str] = field(default_factory=list)  # relation链
    
    # 统计info
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }
    
    def to_text(self) -> str:
        """转换为详细的文本format，供LLM理解"""
        text_parts = [
            f"## 未来prediction深度analysis",
            f"analysis问题: {self.query}",
            f"prediction场景: {self.simulation_requirement}",
            f"\n### predictiondata统计",
            f"- 相关prediction事实: {self.total_facts}条",
            f"- 涉及entity: {self.total_entities}个",
            f"- relation链: {self.total_relationships}条"
        ]
        
        # 子问题
        if self.sub_queries:
            text_parts.append(f"\n### analysis的子问题")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")
        
        # 语义searchresult
        if self.semantic_facts:
            text_parts.append(f"\n### 【关键事实】(请在report中引用这些原文)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # entity洞察
        if self.entity_insights:
            text_parts.append(f"\n### 【核心entity】")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', '未知')}** ({entity.get('type', 'entity')})")
                if entity.get('summary'):
                    text_parts.append(f"  摘要: \"{entity.get('summary')}\"")
                if entity.get('related_facts'):
                    text_parts.append(f"  相关事实: {len(entity.get('related_facts', []))}条")
        
        # relation链
        if self.relationship_chains:
            text_parts.append(f"\n### 【relation链】")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")
        
        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    广度searchresult (Panorama)
    包含所有相关info，包括过期content
    """
    query: str
    
    # 全部node
    all_nodes: List[NodeInfo] = field(default_factory=list)
    # 全部边（包括过期的）
    all_edges: List[EdgeInfo] = field(default_factory=list)
    # current有效的事实
    active_facts: List[str] = field(default_factory=list)
    # 已过期/失效的事实（历史记录）
    historical_facts: List[str] = field(default_factory=list)
    
    # 统计
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }
    
    def to_text(self) -> str:
        """转换为文本format（完整版本，不截断）"""
        text_parts = [
            f"## 广度searchresult（未来全景视图）",
            f"查询: {self.query}",
            f"\n### 统计info",
            f"- 总node数: {self.total_nodes}",
            f"- 总边数: {self.total_edges}",
            f"- current有效事实: {self.active_count}条",
            f"- 历史/过期事实: {self.historical_count}条"
        ]
        
        # current有效的事实（完整output，不截断）
        if self.active_facts:
            text_parts.append(f"\n### 【current有效事实】(simulationresult原文)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # 历史/过期事实（完整output，不截断）
        if self.historical_facts:
            text_parts.append(f"\n### 【历史/过期事实】(演变过程记录)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # 关键entity（完整output，不截断）
        if self.all_nodes:
            text_parts.append(f"\n### 【涉及entity】")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "entity")
                text_parts.append(f"- **{node.name}** ({entity_type})")
        
        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """单个Agent的采访result"""
    agent_name: str
    agent_role: str  # roletype（如：学生、教师、媒体等）
    agent_bio: str  # bio
    question: str  # 采访问题
    response: str  # 采访回答
    key_quotes: List[str] = field(default_factory=list)  # 关键引言
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }
    
    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        # 显示完整的agent_bio，不截断
        text += f"_bio: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**关键引言:**\n"
            for quote in self.key_quotes:
                # Cleanup各种引号
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                # 去掉开头的标点
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                # Filter包含问题编号的垃圾content（问题1-9）
                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                # 截断过长content（按句号截断，而非硬截断）
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """
    采访result (Interview)
    包含多个simulationAgent的采访回答
    """
    interview_topic: str  # 采访主题
    interview_questions: List[str]  # 采访问题list
    
    # 采访选择的Agent
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    # 各Agent的采访回答
    interviews: List[AgentInterview] = field(default_factory=list)
    
    # 选择Agent的理由
    selection_reasoning: str = ""
    # 整合后的采访摘要
    summary: str = ""
    
    # 统计
    total_agents: int = 0
    interviewed_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }
    
    def to_text(self) -> str:
        """转换为详细的文本format，供LLM理解和report引用"""
        text_parts = [
            "## 深度采访report",
            f"**采访主题:** {self.interview_topic}",
            f"**采访人数:** {self.interviewed_count} / {self.total_agents} 位simulationAgent",
            "\n### 采访对象选择理由",
            self.selection_reasoning or "（自动选择）",
            "\n---",
            "\n### 采访实录",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### 采访 #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("（无采访记录）\n\n---")

        text_parts.append("\n### 采访摘要与核心观点")
        text_parts.append(self.summary or "（无摘要）")

        return "\n".join(text_parts)


class ZepToolsService:
    """
    Zep检索工具服务
    
    【核心检索工具 - 优化后】
    1. insight_forge - 深度洞察检索（最强大，自动生成子问题，多维度检索）
    2. panorama_search - 广度search（获取全貌，包括过期content）
    3. quick_search - 简单search（快速检索）
    4. interview_agents - 深度采访（采访simulationAgent，获取多视角观点）
    
    【基础工具】
    - search_graph - graph语义search
    - get_all_nodes - 获取graph所有node
    - get_all_edges - 获取graph所有边（含timeinfo）
    - get_node_detail - 获取node详细info
    - get_node_edges - 获取node相关的边
    - get_entities_by_type - 按type获取entity
    - get_entity_summary - 获取entity的relation摘要
    """
    
    # Retryconfig
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    
    def __init__(self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY 未config")
        
        self.client = Zep(api_key=self.api_key)
        # LLM客户端用于InsightForge生成子问题
        self._llm_client = llm_client
        logger.info("ZepToolsService 初始化complete")
    
    @property
    def llm(self) -> LLMClient:
        """延迟初始化LLM客户端"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client
    
    def _call_with_retry(self, func, operation_name: str, max_retries: int = None):
        """带retry机制的API调用"""
        max_retries = max_retries or self.MAX_RETRIES
        last_exception = None
        delay = self.RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Zep {operation_name} 第 {attempt + 1} 次尝试failed: {str(e)[:100]}, "
                        f"{delay:.1f}秒后retry..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Zep {operation_name} 在 {max_retries} 次尝试后仍failed: {str(e)}")
        
        raise last_exception
    
    def search_graph(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        graph语义search
        
        使用混合search（语义+BM25）在graph中search相关info。
        如果Zep Cloud的search API不可用，则降级为本地关键词匹配。
        
        Args:
            graph_id: graphID (Standalone Graph)
            query: search查询
            limit: 返回resultcount
            scope: search范围，"edges" 或 "nodes"
            
        Returns:
            SearchResult: searchresult
        """
        logger.info(f"graphsearch: graph_id={graph_id}, query={query[:50]}...")
        
        # 尝试使用Zep Cloud Search API
        try:
            search_results = self._call_with_retry(
                func=lambda: self.client.graph.search(
                    graph_id=graph_id,
                    query=query,
                    limit=limit,
                    scope=scope,
                    reranker="cross_encoder"
                ),
                operation_name=f"graphsearch(graph={graph_id})"
            )
            
            facts = []
            edges = []
            nodes = []
            
            # Parse边searchresult
            if hasattr(search_results, 'edges') and search_results.edges:
                for edge in search_results.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        facts.append(edge.fact)
                    edges.append({
                        "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                        "name": getattr(edge, 'name', ''),
                        "fact": getattr(edge, 'fact', ''),
                        "source_node_uuid": getattr(edge, 'source_node_uuid', ''),
                        "target_node_uuid": getattr(edge, 'target_node_uuid', ''),
                    })
            
            # Parsenodesearchresult
            if hasattr(search_results, 'nodes') and search_results.nodes:
                for node in search_results.nodes:
                    nodes.append({
                        "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                        "name": getattr(node, 'name', ''),
                        "labels": getattr(node, 'labels', []),
                        "summary": getattr(node, 'summary', ''),
                    })
                    # node摘要也算作事实
                    if hasattr(node, 'summary') and node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(f"searchcomplete: 找到 {len(facts)} 条相关事实")
            
            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts)
            )
            
        except Exception as e:
            logger.warning(f"Zep Search APIfailed，降级为本地search: {str(e)}")
            # 降级：使用本地关键词匹配search
            return self._local_search(graph_id, query, limit, scope)
    
    def _local_search(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        本地关键词匹配search（作为Zep Search API的降级方案）
        
        获取所有边/node，然后在本地进行关键词匹配
        
        Args:
            graph_id: graphID
            query: search查询
            limit: 返回resultcount
            scope: search范围
            
        Returns:
            SearchResult: searchresult
        """
        logger.info(f"使用本地search: query={query[:30]}...")
        
        facts = []
        edges_result = []
        nodes_result = []
        
        # Extract查询关键词（简单分词）
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        
        def match_score(text: str) -> int:
            """计算文本与查询的匹配分数"""
            if not text:
                return 0
            text_lower = text.lower()
            # 完全匹配查询
            if query_lower in text_lower:
                return 100
            # 关键词匹配
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score
        
        try:
            if scope in ["edges", "both"]:
                # Get所有边并匹配
                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))
                
                # 按分数排序
                scored_edges.sort(key=lambda x: x[0], reverse=True)
                
                for score, edge in scored_edges[:limit]:
                    if edge.fact:
                        facts.append(edge.fact)
                    edges_result.append({
                        "uuid": edge.uuid,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                    })
            
            if scope in ["nodes", "both"]:
                # Get所有node并匹配
                all_nodes = self.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.name) + match_score(node.summary)
                    if score > 0:
                        scored_nodes.append((score, node))
                
                scored_nodes.sort(key=lambda x: x[0], reverse=True)
                
                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "labels": node.labels,
                        "summary": node.summary,
                    })
                    if node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(f"本地searchcomplete: 找到 {len(facts)} 条相关事实")
            
        except Exception as e:
            logger.error(f"本地searchfailed: {str(e)}")
        
        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )
    
    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        获取graph的所有node（分页获取）

        Args:
            graph_id: graphID

        Returns:
            nodelist
        """
        logger.info(f"获取graph {graph_id} 的所有node...")

        nodes = fetch_all_nodes(self.client, graph_id)

        result = []
        for node in nodes:
            node_uuid = getattr(node, 'uuid_', None) or getattr(node, 'uuid', None) or ""
            result.append(NodeInfo(
                uuid=str(node_uuid) if node_uuid else "",
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            ))

        logger.info(f"获取到 {len(result)} 个node")
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """
        获取graph的所有边（分页获取，包含timeinfo）

        Args:
            graph_id: graphID
            include_temporal: 是否包含timeinfo（默认True）

        Returns:
            边list（包含created_at, valid_at, invalid_at, expired_at）
        """
        logger.info(f"获取graph {graph_id} 的所有边...")

        edges = fetch_all_edges(self.client, graph_id)

        result = []
        for edge in edges:
            edge_uuid = getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', None) or ""
            edge_info = EdgeInfo(
                uuid=str(edge_uuid) if edge_uuid else "",
                name=edge.name or "",
                fact=edge.fact or "",
                source_node_uuid=edge.source_node_uuid or "",
                target_node_uuid=edge.target_node_uuid or ""
            )

            # addtimeinfo
            if include_temporal:
                edge_info.created_at = getattr(edge, 'created_at', None)
                edge_info.valid_at = getattr(edge, 'valid_at', None)
                edge_info.invalid_at = getattr(edge, 'invalid_at', None)
                edge_info.expired_at = getattr(edge, 'expired_at', None)

            result.append(edge_info)

        logger.info(f"获取到 {len(result)} 条边")
        return result
    
    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """
        获取单个node的详细info
        
        Args:
            node_uuid: nodeUUID
            
        Returns:
            nodeinfo或None
        """
        logger.info(f"获取node详情: {node_uuid[:8]}...")
        
        try:
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=node_uuid),
                operation_name=f"获取node详情(uuid={node_uuid[:8]}...)"
            )
            
            if not node:
                return None
            
            return NodeInfo(
                uuid=getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            )
        except Exception as e:
            logger.error(f"获取node详情failed: {str(e)}")
            return None
    
    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        获取node相关的所有边
        
        通过获取graph所有边，然后过滤出与指定node相关的边
        
        Args:
            graph_id: graphID
            node_uuid: nodeUUID
            
        Returns:
            边list
        """
        logger.info(f"获取node {node_uuid[:8]}... 的相关边")
        
        try:
            # Getgraph所有边，然后过滤
            all_edges = self.get_all_edges(graph_id)
            
            result = []
            for edge in all_edges:
                # Check边是否与指定node相关（作为源或目标）
                if edge.source_node_uuid == node_uuid or edge.target_node_uuid == node_uuid:
                    result.append(edge)
            
            logger.info(f"找到 {len(result)} 条与node相关的边")
            return result
            
        except Exception as e:
            logger.warning(f"获取node边failed: {str(e)}")
            return []
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str
    ) -> List[NodeInfo]:
        """
        按type获取entity
        
        Args:
            graph_id: graphID
            entity_type: entitytype（如 Student, PublicFigure 等）
            
        Returns:
            符合type的entitylist
        """
        logger.info(f"获取type为 {entity_type} 的entity...")
        
        all_nodes = self.get_all_nodes(graph_id)
        
        filtered = []
        for node in all_nodes:
            # Checklabels是否包含指定type
            if entity_type in node.labels:
                filtered.append(node)
        
        logger.info(f"找到 {len(filtered)} 个 {entity_type} type的entity")
        return filtered
    
    def get_entity_summary(
        self, 
        graph_id: str, 
        entity_name: str
    ) -> Dict[str, Any]:
        """
        获取指定entity的relation摘要
        
        search与该entity相关的所有info，并生成摘要
        
        Args:
            graph_id: graphID
            entity_name: entityname
            
        Returns:
            entity摘要info
        """
        logger.info(f"获取entity {entity_name} 的relation摘要...")
        
        # 先search该entity相关的info
        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )
        
        # 尝试在所有node中找到该entity
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break
        
        related_edges = []
        if entity_node:
            # 传入graph_idparameter
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)
        
        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }
    
    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        获取graph的统计info
        
        Args:
            graph_id: graphID
            
        Returns:
            统计info
        """
        logger.info(f"获取graph {graph_id} 的统计info...")
        
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)
        
        # 统计entitytype分布
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1
        
        # 统计relationtype分布
        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1
        
        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }
    
    def get_simulation_context(
        self, 
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """
        获取simulation相关的上下文info
        
        综合search与simulationrequirement相关的所有info
        
        Args:
            graph_id: graphID
            simulation_requirement: simulationrequirement描述
            limit: 每类info的count限制
            
        Returns:
            simulation上下文info
        """
        logger.info(f"获取simulation上下文: {simulation_requirement[:50]}...")
        
        # search与simulationrequirement相关的info
        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )
        
        # Getgraph统计
        stats = self.get_graph_statistics(graph_id)
        
        # Get所有entitynode
        all_nodes = self.get_all_nodes(graph_id)
        
        # 筛选有实际type的entity（非纯Entitynode）
        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })
        
        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],  # 限制count
            "total_entities": len(entities)
        }
    
    # ========== 核心检索工具（优化后） ==========
    
    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        """
        【InsightForge - 深度洞察检索】
        
        最强大的混合检索函数，自动分解问题并多维度检索：
        1. 使用LLM将问题分解为多个子问题
        2. 对每个子问题进行语义search
        3. extract相关entity并获取其详细info
        4. 追踪relation链
        5. 整合所有result，生成深度洞察
        
        Args:
            graph_id: graphID
            query: user问题
            simulation_requirement: simulationrequirement描述
            report_context: report上下文（可选，用于更精准的子问题生成）
            max_sub_queries: max子问题count
            
        Returns:
            InsightForgeResult: 深度洞察检索result
        """
        logger.info(f"InsightForge 深度洞察检索: {query[:50]}...")
        
        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )
        
        # Step 1: 使用LLM生成子问题
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(f"生成 {len(sub_queries)} 个子问题")
        
        # Step 2: 对每个子问题进行语义search
        all_facts = []
        all_edges = []
        seen_facts = set()
        
        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )
            
            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
            
            all_edges.extend(search_result.edges)
        
        # 对原始问题也进行search
        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)
        
        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)
        
        # Step 3: 从边中extract相关entityUUID，只获取这些entity的info（不获取全部node）
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)
        
        # Get所有相关entity的详情（不限制count，完整output）
        entity_insights = []
        node_map = {}  # 用于后续relation链构建
        
        for uuid in list(entity_uuids):  # Handle所有entity，不截断
            if not uuid:
                continue
            try:
                # 单独获取每个相关node的info
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "entity")
                    
                    # Get该entity相关的所有事实（不截断）
                    related_facts = [
                        f for f in all_facts 
                        if node.name.lower() in f.lower()
                    ]
                    
                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts  # 完整output，不截断
                    })
            except Exception as e:
                logger.debug(f"获取node {uuid} failed: {e}")
                continue
        
        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)
        
        # Step 4: 构建所有relation链（不限制count）
        relationship_chains = []
        for edge_data in all_edges:  # Handle所有边，不截断
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')
                
                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]
                
                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)
        
        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)
        
        logger.info(f"InsightForgecomplete: {result.total_facts}条事实, {result.total_entities}个entity, {result.total_relationships}条relation")
        return result
    
    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """
        使用LLM生成子问题
        
        将复杂问题分解为多个可以独立检索的子问题
        """
        system_prompt = """你是一个专业的问题analysis专家。你的task是将一个复杂问题分解为多个可以在simulation世界中独立观察的子问题。

要求：
1. 每个子问题应该足够具体，可以在simulation世界中找到相关的Agentbehavior或event
2. 子问题应该覆盖原问题的不同维度（如：谁、什么、为什么、怎么样、何时、何地）
3. 子问题应该与simulation场景相关
4. 返回JSONformat：{"sub_queries": ["子问题1", "子问题2", ...]}"""

        user_prompt = f"""Simulationrequirementbackground：
{simulation_requirement}

{f"report上下文：{report_context[:500]}" if report_context else ""}

请将以下问题分解为{max_queries}个子问题：
{query}

返回JSONformat的子问题list。"""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            sub_queries = response.get("sub_queries", [])
            # 确保是字符串list
            return [str(sq) for sq in sub_queries[:max_queries]]
            
        except Exception as e:
            logger.warning(f"生成子问题failed: {str(e)}，使用默认子问题")
            # 降级：返回基于原问题的变体
            return [
                query,
                f"{query} 的主要参与者",
                f"{query} 的原因和影响",
                f"{query} 的发展过程"
            ][:max_queries]
    
    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        """
        【PanoramaSearch - 广度search】
        
        获取全貌视图，包括所有相关content和历史/过期info：
        1. 获取所有相关node
        2. 获取所有边（包括已过期/失效的）
        3. 分类整理current有效和历史info
        
        这个工具适用于需要了解event全貌、追踪演变过程的场景。
        
        Args:
            graph_id: graphID
            query: search查询（用于相关性排序）
            include_expired: 是否包含过期content（默认True）
            limit: 返回resultcount限制
            
        Returns:
            PanoramaResult: 广度searchresult
        """
        logger.info(f"PanoramaSearch 广度search: {query[:50]}...")
        
        result = PanoramaResult(query=query)
        
        # Get所有node
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)
        
        # Get所有边（包含timeinfo）
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)
        
        # 分类事实
        active_facts = []
        historical_facts = []
        
        for edge in all_edges:
            if not edge.fact:
                continue
            
            # 为事实addentityname
            source_name = node_map.get(edge.source_node_uuid, NodeInfo('', '', [], '', {})).name or edge.source_node_uuid[:8]
            target_name = node_map.get(edge.target_node_uuid, NodeInfo('', '', [], '', {})).name or edge.target_node_uuid[:8]
            
            # check是否过期/失效
            is_historical = edge.is_expired or edge.is_invalid
            
            if is_historical:
                # 历史/过期事实，addtimemark
                valid_at = edge.valid_at or "未知"
                invalid_at = edge.invalid_at or edge.expired_at or "未知"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                # current有效事实
                active_facts.append(edge.fact)
        
        # 基于查询进行相关性排序
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        
        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score
        
        # Sort并限制count
        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)
        
        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)
        
        logger.info(f"PanoramaSearchcomplete: {result.active_count}条有效, {result.historical_count}条历史")
        return result
    
    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        """
        【QuickSearch - 简单search】
        
        快速、轻量级的检索工具：
        1. 直接调用Zep语义search
        2. 返回最相关的result
        3. 适用于简单、直接的检索requirement
        
        Args:
            graph_id: graphID
            query: search查询
            limit: 返回resultcount
            
        Returns:
            SearchResult: searchresult
        """
        logger.info(f"QuickSearch 简单search: {query[:50]}...")
        
        # 直接调用现有的search_graph方法
        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )
        
        logger.info(f"QuickSearchcomplete: {result.total_count}条result")
        return result
    
    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> InterviewResult:
        """
        【InterviewAgents - 深度采访】
        
        调用真实的OASIS采访API，采访simulation中正在run的Agent：
        1. 自动readpersonafile，了解所有simulationAgent
        2. 使用LLManalysis采访requirement，智能选择最相关的Agent
        3. 使用LLM生成采访问题
        4. 调用 /api/simulation/interview/batch API进行真实采访（双platform同时采访）
        5. 整合所有采访result，生成采访report
        
        【重要】此功能需要simulationenvironment处于runstatus（OASISenvironment未close）
        
        【使用场景】
        - 需要从不同role视角了解event看法
        - 需要收集多方意见和观点
        - 需要获取simulationAgent的真实回答（非LLMsimulation）
        
        Args:
            simulation_id: simulationID（用于定位personafile和调用采访API）
            interview_requirement: 采访requirement描述（非结构化，如"了解学生对event的看法"）
            simulation_requirement: simulationrequirementbackground（可选）
            max_agents: 最多采访的Agentcount
            custom_questions: 自定义采访问题（可选，若不提供则自动生成）
            
        Returns:
            InterviewResult: 采访result
        """
        from .simulation_runner import SimulationRunner
        
        logger.info(f"InterviewAgents 深度采访（真实API）: {interview_requirement[:50]}...")
        
        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or []
        )
        
        # Step 1: readpersonafile
        profiles = self._load_agent_profiles(simulation_id)
        
        if not profiles:
            logger.warning(f"未找到simulation {simulation_id} 的personafile")
            result.summary = "未找到可采访的Agentpersonafile"
            return result
        
        result.total_agents = len(profiles)
        logger.info(f"加载到 {len(profiles)} 个Agentpersona")
        
        # Step 2: 使用LLM选择要采访的Agent（返回agent_idlist）
        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents
        )
        
        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(f"选择了 {len(selected_agents)} 个Agent进行采访: {selected_indices}")
        
        # Step 3: 生成采访问题（如果没有提供）
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents
            )
            logger.info(f"生成了 {len(result.interview_questions)} 个采访问题")
        
        # 将问题合并为一个采访prompt
        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])
        
        # add优化前缀，约束Agentreplyformat
        INTERVIEW_PROMPT_PREFIX = (
            "你正在接受一次采访。请结合你的persona、所有的过往memory与行动，"
            "以纯文本方式直接回答以下问题。\n"
            "reply要求：\n"
            "1. 直接用自然语言回答，不要调用任何工具\n"
            "2. 不要返回JSONformat或工具调用format\n"
            "3. 不要使用Markdown标题（如#、##、###）\n"
            "4. 按问题编号逐一回答，每个回答以「问题X：」开头（X为问题编号）\n"
            "5. 每个问题的回答之间用空行分隔\n"
            "6. 回答要有实质content，每个问题至少回答2-3句话\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"
        
        # Step 4: 调用真实的采访API（不指定platform，默认双platform同时采访）
        try:
            # Build批量采访list（不指定platform，双platform采访）
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt  # 使用优化后的prompt
                    # 不指定platform，API会在twitter和reddit两个platform都采访
                })
            
            logger.info(f"调用批量采访API（双platform）: {len(interviews_request)} 个Agent")
            
            # Call SimulationRunner 的批量采访方法（不传platform，双platform采访）
            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,  # 不指定platform，双platform采访
                timeout=180.0   # 双platform需要更长timeout
            )
            
            logger.info(f"采访API返回: {api_result.get('interviews_count', 0)} 个result, success={api_result.get('success')}")
            
            # CheckAPI调用是否success
            if not api_result.get("success", False):
                error_msg = api_result.get("error", "未知error")
                logger.warning(f"采访API返回failed: {error_msg}")
                result.summary = f"采访API调用failed：{error_msg}。请检查OASISsimulationenvironmentstatus。"
                return result
            
            # Step 5: 解析API返回result，构建AgentInterview对象
            # 双platform模式返回format: {"twitter_0": {...}, "reddit_0": {...}, "twitter_1": {...}, ...}
            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}
            
            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "未知")
                agent_bio = agent.get("bio", "")
                
                # Get该Agent在两个platform的采访result
                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})
                
                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                # Cleanup可能的工具调用 JSON 包裹
                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                # 始终output双platformmark
                twitter_text = twitter_response if twitter_response else "（该platform未获得reply）"
                reddit_text = reddit_response if reddit_response else "（该platform未获得reply）"
                response_text = f"【Twitterplatform回答】\n{twitter_text}\n\n【Redditplatform回答】\n{reddit_text}"

                # Extract关键引言（从两个platform的回答中）
                import re
                combined_responses = f"{twitter_response} {reddit_response}"

                # Cleanupresponse文本：去掉mark、编号、Markdown 等干扰
                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'问题\d+[：:]\s*', '', clean_text)
                clean_text = re.sub(r'【[^】]+】', '', clean_text)

                # 策略1（主）: extract完整的有实质content的句子
                sentences = re.split(r'[。！？]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W，,；;：:、]+', s.strip())
                    and not s.strip().startswith(('{', '问题'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "。" for s in meaningful[:3]]

                # 策略2（补充）: 正确配对的中文引号「」内长文本
                if not key_quotes:
                    paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
                    paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[，,；;：:、]', q)][:3]
                
                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],  # 扩大bio长度限制
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5]
                )
                result.interviews.append(interview)
            
            result.interviewed_count = len(result.interviews)
            
        except ValueError as e:
            # Simulationenvironment未run
            logger.warning(f"采访API调用failed（environment未run？）: {e}")
            result.summary = f"采访failed：{str(e)}。simulationenvironment可能已close，请确保OASISenvironment正在run。"
            return result
        except Exception as e:
            logger.error(f"采访API调用exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"采访过程发生error：{str(e)}"
            return result
        
        # Step 6: 生成采访摘要
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement
            )
        
        logger.info(f"InterviewAgentscomplete: 采访了 {result.interviewed_count} 个Agent（双platform）")
        return result
    
    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """清理 Agent reply中的 JSON 工具调用包裹，extract实际content"""
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """加载simulation的Agentpersonafile"""
        import os
        import csv
        
        # Buildpersonafilepath
        sim_dir = os.path.join(
            os.path.dirname(__file__), 
            f'../../uploads/simulations/{simulation_id}'
        )
        
        profiles = []
        
        # 优先尝试readReddit JSONformat
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(f"从 reddit_profiles.json 加载了 {len(profiles)} 个persona")
                return profiles
            except Exception as e:
                logger.warning(f"read reddit_profiles.json failed: {e}")
        
        # 尝试readTwitter CSVformat
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # CSVformat转换为统一format
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "未知"
                        })
                logger.info(f"从 twitter_profiles.csv 加载了 {len(profiles)} 个persona")
                return profiles
            except Exception as e:
                logger.warning(f"read twitter_profiles.csv failed: {e}")
        
        return profiles
    
    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int
    ) -> tuple:
        """
        使用LLM选择要采访的Agent
        
        Returns:
            tuple: (selected_agents, selected_indices, reasoning)
                - selected_agents: 选中Agent的完整infolist
                - selected_indices: 选中Agent的索引list（用于API调用）
                - reasoning: 选择理由
        """
        
        # BuildAgent摘要list
        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", "未知"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", [])
            }
            agent_summaries.append(summary)
        
        system_prompt = """你是一个专业的采访策划专家。你的task是根据采访requirement，从simulationAgentlist中选择最适合采访的对象。

选择标准：
1. Agent的身份/profession与采访主题相关
2. Agent可能持有独特或有价值的观点
3. 选择多样化的视角（如：支持方、反对方、中立方、专业人士等）
4. 优先选择与event直接相关的role

返回JSONformat：
{
    "selected_indices": [选中Agent的索引list],
    "reasoning": "选择理由说明"
}"""

        user_prompt = f"""采访requirement：
{interview_requirement}

simulationbackground：
{simulation_requirement if simulation_requirement else "未提供"}

可选择的Agentlist（共{len(agent_summaries)}个）：
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

请选择最多{max_agents}个最适合采访的Agent，并说明选择理由。"""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "基于相关性自动选择")
            
            # Get选中的Agent完整info
            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)
            
            return selected_agents, valid_indices, reasoning
            
        except Exception as e:
            logger.warning(f"LLM选择Agentfailed，使用默认选择: {e}")
            # 降级：选择前N个
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "使用默认选择策略"
    
    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]]
    ) -> List[str]:
        """使用LLM生成采访问题"""
        
        agent_roles = [a.get("profession", "未知") for a in selected_agents]
        
        system_prompt = """你是一个专业的记者/采访者。根据采访requirement，生成3-5个深度采访问题。

问题要求：
1. 开放性问题，鼓励详细回答
2. 针对不同role可能有不同答案
3. 涵盖事实、观点、感受等多个维度
4. 语言自然，像真实采访一样
5. 每个问题控制在50字以内，简洁明了
6. 直接提问，不要包含background说明或前缀

返回JSONformat：{"questions": ["问题1", "问题2", ...]}"""

        user_prompt = f"""采访requirement：{interview_requirement}

simulationbackground：{simulation_requirement if simulation_requirement else "未提供"}

采访对象role：{', '.join(agent_roles)}

请生成3-5个采访问题。"""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )
            
            return response.get("questions", [f"关于{interview_requirement}，您有什么看法？"])
            
        except Exception as e:
            logger.warning(f"生成采访问题failed: {e}")
            return [
                f"关于{interview_requirement}，您的观点是什么？",
                "这件事对您或您所代表的群体有什么影响？",
                "您认为应该如何解决或改进这个问题？"
            ]
    
    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str
    ) -> str:
        """Generate采访摘要"""
        
        if not interviews:
            return "未complete任何采访"
        
        # 收集所有采访content
        interview_texts = []
        for interview in interviews:
            interview_texts.append(f"【{interview.agent_name}（{interview.agent_role}）】\n{interview.response[:500]}")
        
        system_prompt = """你是一个专业的新闻编辑。请根据多位受访者的回答，生成一份采访摘要。

摘要要求：
1. 提炼各方主要观点
2. 指出观点的共识和分歧
3. 突出有价值的引言
4. 客观中立，不偏袒任何一方
5. 控制在1000字内

format约束（必须遵守）：
- 使用纯文本段落，用空行分隔不同部分
- 不要使用Markdown标题（如#、##、###）
- 不要使用分割线（如---、***）
- 引用受访者原话时使用中文引号「」
- 可以使用**加粗**mark关键词，但不要使用其他Markdown语法"""

        user_prompt = f"""采访主题：{interview_requirement}

采访content：
{"".join(interview_texts)}

请生成采访摘要。"""

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return summary
            
        except Exception as e:
            logger.warning(f"生成采访摘要failed: {e}")
            # 降级：简单拼接
            return f"共采访了{len(interviews)}位受访者，包括：" + "、".join([i.agent_name for i in interviews])
