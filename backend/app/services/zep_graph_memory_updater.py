"""
Zepgraphmemory更新服务
将simulation中的Agent活动动态更新到Zepgraph中
"""

import os
import time
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.zep_graph_memory_updater')


@dataclass
class AgentActivity:
    """Agent活动记录"""
    platform: str           # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str        # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any]
    round_num: int
    timestamp: str
    
    def to_episode_text(self) -> str:
        """
        将活动转换为可以发送给Zep的文本描述
        
        采用自然语言描述format，让Zep能够从中extractentity和relation
        不addsimulation相关的前缀，避免误导graph更新
        """
        # 根据不同的actiontype生成不同的描述
        action_descriptions = {
            "CREATE_POST": self._describe_create_post,
            "LIKE_POST": self._describe_like_post,
            "DISLIKE_POST": self._describe_dislike_post,
            "REPOST": self._describe_repost,
            "QUOTE_POST": self._describe_quote_post,
            "FOLLOW": self._describe_follow,
            "CREATE_COMMENT": self._describe_create_comment,
            "LIKE_COMMENT": self._describe_like_comment,
            "DISLIKE_COMMENT": self._describe_dislike_comment,
            "SEARCH_POSTS": self._describe_search,
            "SEARCH_USER": self._describe_search_user,
            "MUTE": self._describe_mute,
        }
        
        describe_func = action_descriptions.get(self.action_type, self._describe_generic)
        description = describe_func()
        
        # 直接返回 "agentname: 活动描述" format，不addsimulation前缀
        return f"{self.agent_name}: {description}"
    
    def _describe_create_post(self) -> str:
        content = self.action_args.get("content", "")
        if content:
            return f"发布了一条post：「{content}」"
        return "发布了一条post"
    
    def _describe_like_post(self) -> str:
        """likepost - 包含post原文和作者info"""
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        
        if post_content and post_author:
            return f"like了{post_author}的post：「{post_content}」"
        elif post_content:
            return f"like了一条post：「{post_content}」"
        elif post_author:
            return f"like了{post_author}的一条post"
        return "like了一条post"
    
    def _describe_dislike_post(self) -> str:
        """踩post - 包含post原文和作者info"""
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        
        if post_content and post_author:
            return f"踩了{post_author}的post：「{post_content}」"
        elif post_content:
            return f"踩了一条post：「{post_content}」"
        elif post_author:
            return f"踩了{post_author}的一条post"
        return "踩了一条post"
    
    def _describe_repost(self) -> str:
        """repostpost - 包含原帖content和作者info"""
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        
        if original_content and original_author:
            return f"repost了{original_author}的post：「{original_content}」"
        elif original_content:
            return f"repost了一条post：「{original_content}」"
        elif original_author:
            return f"repost了{original_author}的一条post"
        return "repost了一条post"
    
    def _describe_quote_post(self) -> str:
        """引用post - 包含原帖content、作者info和引用comment"""
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        quote_content = self.action_args.get("quote_content", "") or self.action_args.get("content", "")
        
        base = ""
        if original_content and original_author:
            base = f"引用了{original_author}的post「{original_content}」"
        elif original_content:
            base = f"引用了一条post「{original_content}」"
        elif original_author:
            base = f"引用了{original_author}的一条post"
        else:
            base = "引用了一条post"
        
        if quote_content:
            base += f"，并comment道：「{quote_content}」"
        return base
    
    def _describe_follow(self) -> str:
        """followuser - 包含被followuser的name"""
        target_user_name = self.action_args.get("target_user_name", "")
        
        if target_user_name:
            return f"follow了user「{target_user_name}」"
        return "follow了一个user"
    
    def _describe_create_comment(self) -> str:
        """发表comment - 包含commentcontent和所comment的postinfo"""
        content = self.action_args.get("content", "")
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        
        if content:
            if post_content and post_author:
                return f"在{post_author}的post「{post_content}」下comment道：「{content}」"
            elif post_content:
                return f"在post「{post_content}」下comment道：「{content}」"
            elif post_author:
                return f"在{post_author}的post下comment道：「{content}」"
            return f"comment道：「{content}」"
        return "发表了comment"
    
    def _describe_like_comment(self) -> str:
        """likecomment - 包含commentcontent和作者info"""
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")
        
        if comment_content and comment_author:
            return f"like了{comment_author}的comment：「{comment_content}」"
        elif comment_content:
            return f"like了一条comment：「{comment_content}」"
        elif comment_author:
            return f"like了{comment_author}的一条comment"
        return "like了一条comment"
    
    def _describe_dislike_comment(self) -> str:
        """踩comment - 包含commentcontent和作者info"""
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")
        
        if comment_content and comment_author:
            return f"踩了{comment_author}的comment：「{comment_content}」"
        elif comment_content:
            return f"踩了一条comment：「{comment_content}」"
        elif comment_author:
            return f"踩了{comment_author}的一条comment"
        return "踩了一条comment"
    
    def _describe_search(self) -> str:
        """searchpost - 包含search关键词"""
        query = self.action_args.get("query", "") or self.action_args.get("keyword", "")
        return f"search了「{query}」" if query else "进行了search"
    
    def _describe_search_user(self) -> str:
        """searchuser - 包含search关键词"""
        query = self.action_args.get("query", "") or self.action_args.get("username", "")
        return f"search了user「{query}」" if query else "search了user"
    
    def _describe_mute(self) -> str:
        """屏蔽user - 包含被屏蔽user的name"""
        target_user_name = self.action_args.get("target_user_name", "")
        
        if target_user_name:
            return f"屏蔽了user「{target_user_name}」"
        return "屏蔽了一个user"
    
    def _describe_generic(self) -> str:
        # 对于未知的actiontype，生成通用描述
        return f"execute了{self.action_type}操作"


class ZepGraphMemoryUpdater:
    """
    Zepgraphmemory更新器
    
    监控simulation的actionslogfile，将新的agent活动实时更新到Zepgraph中。
    按platform分组，每累积BATCH_SIZE条活动后批量发送到Zep。
    
    所有有意义的behavior都会被更新到Zep，action_args中会包含完整的上下文info：
    - like/踩的post原文
    - repost/引用的post原文
    - follow/屏蔽的user名
    - like/踩的comment原文
    """
    
    # 批量发送大小（每个platform累积多少条后发送）
    BATCH_SIZE = 5
    
    # platformname映射（用于控制台显示）
    PLATFORM_DISPLAY_NAMES = {
        'twitter': '世界1',
        'reddit': '世界2',
    }
    
    # Send间隔（秒），避免request过快
    SEND_INTERVAL = 0.5
    
    # Retryconfig
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # 秒
    
    def __init__(self, graph_id: str, api_key: Optional[str] = None):
        """
        初始化更新器
        
        Args:
            graph_id: ZepgraphID
            api_key: Zep API Key（可选，默认从configread）
        """
        self.graph_id = graph_id
        self.api_key = api_key or Config.ZEP_API_KEY
        
        if not self.api_key:
            raise ValueError("ZEP_API_KEY未config")
        
        self.client = Zep(api_key=self.api_key)
        
        # 活动队列
        self._activity_queue: Queue = Queue()
        
        # 按platform分组的活动缓冲区（每个platform各自累积到BATCH_SIZE后批量发送）
        self._platform_buffers: Dict[str, List[AgentActivity]] = {
            'twitter': [],
            'reddit': [],
        }
        self._buffer_lock = threading.Lock()
        
        # 控制标志
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # 统计
        self._total_activities = 0  # 实际add到队列的活动数
        self._total_sent = 0        # success发送到Zep的批次数
        self._total_items_sent = 0  # success发送到Zep的活动条数
        self._failed_count = 0      # Sendfailed的批次数
        self._skipped_count = 0     # 被过滤跳过的活动数（DO_NOTHING）
        
        logger.info(f"ZepGraphMemoryUpdater 初始化complete: graph_id={graph_id}, batch_size={self.BATCH_SIZE}")
    
    def _get_platform_display_name(self, platform: str) -> str:
        """Getplatform的显示name"""
        return self.PLATFORM_DISPLAY_NAMES.get(platform.lower(), platform)
    
    def start(self):
        """Start后台工作thread"""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"ZepMemoryUpdater-{self.graph_id[:8]}"
        )
        self._worker_thread.start()
        logger.info(f"ZepGraphMemoryUpdater 已启动: graph_id={self.graph_id}")
    
    def stop(self):
        """Stop后台工作thread"""
        self._running = False
        
        # Send剩余的活动
        self._flush_remaining()
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)
        
        logger.info(f"ZepGraphMemoryUpdater 已停止: graph_id={self.graph_id}, "
                   f"total_activities={self._total_activities}, "
                   f"batches_sent={self._total_sent}, "
                   f"items_sent={self._total_items_sent}, "
                   f"failed={self._failed_count}, "
                   f"skipped={self._skipped_count}")
    
    def add_activity(self, activity: AgentActivity):
        """
        add一个agent活动到队列
        
        所有有意义的behavior都会被add到队列，包括：
        - CREATE_POST（发帖）
        - CREATE_COMMENT（comment）
        - QUOTE_POST（引用post）
        - SEARCH_POSTS（searchpost）
        - SEARCH_USER（searchuser）
        - LIKE_POST/DISLIKE_POST（like/踩post）
        - REPOST（repost）
        - FOLLOW（follow）
        - MUTE（屏蔽）
        - LIKE_COMMENT/DISLIKE_COMMENT（like/踩comment）
        
        action_args中会包含完整的上下文info（如post原文、user名等）。
        
        Args:
            activity: Agent活动记录
        """
        # 跳过DO_NOTHINGtype的活动
        if activity.action_type == "DO_NOTHING":
            self._skipped_count += 1
            return
        
        self._activity_queue.put(activity)
        self._total_activities += 1
        logger.debug(f"add活动到Zep队列: {activity.agent_name} - {activity.action_type}")
    
    def add_activity_from_dict(self, data: Dict[str, Any], platform: str):
        """
        从字典dataadd活动
        
        Args:
            data: 从actions.jsonl解析的字典data
            platform: platformname (twitter/reddit)
        """
        # 跳过eventtype的条目
        if "event_type" in data:
            return
        
        activity = AgentActivity(
            platform=platform,
            agent_id=data.get("agent_id", 0),
            agent_name=data.get("agent_name", ""),
            action_type=data.get("action_type", ""),
            action_args=data.get("action_args", {}),
            round_num=data.get("round", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )
        
        self.add_activity(activity)
    
    def _worker_loop(self):
        """后台工作loop - 按platform批量发送活动到Zep"""
        while self._running or not self._activity_queue.empty():
            try:
                # 尝试从队列获取活动（timeout1秒）
                try:
                    activity = self._activity_queue.get(timeout=1)
                    
                    # 将活动add到对应platform的缓冲区
                    platform = activity.platform.lower()
                    with self._buffer_lock:
                        if platform not in self._platform_buffers:
                            self._platform_buffers[platform] = []
                        self._platform_buffers[platform].append(activity)
                        
                        # Check该platform是否达到批量大小
                        if len(self._platform_buffers[platform]) >= self.BATCH_SIZE:
                            batch = self._platform_buffers[platform][:self.BATCH_SIZE]
                            self._platform_buffers[platform] = self._platform_buffers[platform][self.BATCH_SIZE:]
                            # 释放锁后再发送
                            self._send_batch_activities(batch, platform)
                            # Send间隔，避免request过快
                            time.sleep(self.SEND_INTERVAL)
                    
                except Empty:
                    pass
                    
            except Exception as e:
                logger.error(f"工作loopexception: {e}")
                time.sleep(1)
    
    def _send_batch_activities(self, activities: List[AgentActivity], platform: str):
        """
        批量发送活动到Zepgraph（合并为一条文本）
        
        Args:
            activities: Agent活动list
            platform: platformname
        """
        if not activities:
            return
        
        # 将多条活动合并为一条文本，用换行分隔
        episode_texts = [activity.to_episode_text() for activity in activities]
        combined_text = "\n".join(episode_texts)
        
        # 带retry的发送
        for attempt in range(self.MAX_RETRIES):
            try:
                self.client.graph.add(
                    graph_id=self.graph_id,
                    type="text",
                    data=combined_text
                )
                
                self._total_sent += 1
                self._total_items_sent += len(activities)
                display_name = self._get_platform_display_name(platform)
                logger.info(f"success批量发送 {len(activities)} 条{display_name}活动到graph {self.graph_id}")
                logger.debug(f"批量content预览: {combined_text[:200]}...")
                return
                
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"批量发送到Zepfailed (尝试 {attempt + 1}/{self.MAX_RETRIES}): {e}")
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"批量发送到Zepfailed，已retry{self.MAX_RETRIES}次: {e}")
                    self._failed_count += 1
    
    def _flush_remaining(self):
        """发送队列和缓冲区中剩余的活动"""
        # 首先处理队列中剩余的活动，add到缓冲区
        while not self._activity_queue.empty():
            try:
                activity = self._activity_queue.get_nowait()
                platform = activity.platform.lower()
                with self._buffer_lock:
                    if platform not in self._platform_buffers:
                        self._platform_buffers[platform] = []
                    self._platform_buffers[platform].append(activity)
            except Empty:
                break
        
        # 然后发送各platform缓冲区中剩余的活动（即使不足BATCH_SIZE条）
        with self._buffer_lock:
            for platform, buffer in self._platform_buffers.items():
                if buffer:
                    display_name = self._get_platform_display_name(platform)
                    logger.info(f"发送{display_name}platform剩余的 {len(buffer)} 条活动")
                    self._send_batch_activities(buffer, platform)
            # 清空所有缓冲区
            for platform in self._platform_buffers:
                self._platform_buffers[platform] = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get统计info"""
        with self._buffer_lock:
            buffer_sizes = {p: len(b) for p, b in self._platform_buffers.items()}
        
        return {
            "graph_id": self.graph_id,
            "batch_size": self.BATCH_SIZE,
            "total_activities": self._total_activities,  # add到队列的活动total
            "batches_sent": self._total_sent,            # success发送的批次数
            "items_sent": self._total_items_sent,        # success发送的活动条数
            "failed_count": self._failed_count,          # Sendfailed的批次数
            "skipped_count": self._skipped_count,        # 被过滤跳过的活动数（DO_NOTHING）
            "queue_size": self._activity_queue.qsize(),
            "buffer_sizes": buffer_sizes,                # 各platform缓冲区大小
            "running": self._running,
        }


class ZepGraphMemoryManager:
    """
    管理多个simulation的Zepgraphmemory更新器
    
    每个simulation可以有自己的更新器实例
    """
    
    _updaters: Dict[str, ZepGraphMemoryUpdater] = {}
    _lock = threading.Lock()
    
    @classmethod
    def create_updater(cls, simulation_id: str, graph_id: str) -> ZepGraphMemoryUpdater:
        """
        为simulation创建graphmemory更新器
        
        Args:
            simulation_id: simulationID
            graph_id: ZepgraphID
            
        Returns:
            ZepGraphMemoryUpdater实例
        """
        with cls._lock:
            # If已存在，先停止旧的
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
            
            updater = ZepGraphMemoryUpdater(graph_id)
            updater.start()
            cls._updaters[simulation_id] = updater
            
            logger.info(f"创建graphmemory更新器: simulation_id={simulation_id}, graph_id={graph_id}")
            return updater
    
    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[ZepGraphMemoryUpdater]:
        """Getsimulation的更新器"""
        return cls._updaters.get(simulation_id)
    
    @classmethod
    def stop_updater(cls, simulation_id: str):
        """Stop并移除simulation的更新器"""
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
                del cls._updaters[simulation_id]
                logger.info(f"已停止graphmemory更新器: simulation_id={simulation_id}")
    
    # 防止 stop_all 重复调用的标志
    _stop_all_done = False
    
    @classmethod
    def stop_all(cls):
        """Stop所有更新器"""
        # 防止重复调用
        if cls._stop_all_done:
            return
        cls._stop_all_done = True
        
        with cls._lock:
            if cls._updaters:
                for simulation_id, updater in list(cls._updaters.items()):
                    try:
                        updater.stop()
                    except Exception as e:
                        logger.error(f"停止更新器failed: simulation_id={simulation_id}, error={e}")
                cls._updaters.clear()
            logger.info("已停止所有graphmemory更新器")
    
    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Get所有更新器的统计info"""
        return {
            sim_id: updater.get_stats() 
            for sim_id, updater in cls._updaters.items()
        }
