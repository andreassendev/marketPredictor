"""
ontology生成服务
API1：analysis文本content，生成适合社会simulation的entity和relationtype定义
"""

import json
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient


# ontology生成的系统提示词
ONTOLOGY_SYSTEM_PROMPT = """你是一个专业的知识graphontology设计专家。你的task是analysis给定的文本content和simulationrequirement，设计适合**社交媒体舆论simulation**的entitytype和relationtype。

**重要：你必须output有效的JSONformatdata，不要output任何其他content。**

## 核心taskbackground

我们正在构建一个**社交媒体舆论simulation系统**。在这个系统中：
- 每个entity都是一个可以在社交媒体上发声、互动、传播info的"账号"或"主体"
- entity之间会相互影响、repost、comment、回应
- 我们需要simulation舆论event中各方的反应和info传播path

因此，**entity必须是现实中真实存在的、可以在社媒上发声和互动的主体**：

**可以是**：
- 具体的个人（公众人物、当事人、意见领袖、专家学者、普通人）
- 公司、企业（包括其官方账号）
- 组织机构（大学、协会、NGO、工会等）
- 政府部门、监管机构
- 媒体机构（报纸、电视台、自媒体、网站）
- 社交媒体platform本身
- 特定群体代表（如校友会、粉丝团、维权群体等）

**不可以是**：
- 抽象概念（如"舆论"、"情绪"、"趋势"）
- 主题/topic（如"学术诚信"、"教育改革"）
- 观点/态度（如"支持方"、"反对方"）

## outputformat

请outputJSONformat，包含以下结构：

```json
{
    "entity_types": [
        {
            "name": "entitytypename（英文，PascalCase）",
            "description": "简短描述（英文，不超过100字符）",
            "attributes": [
                {
                    "name": "属性名（英文，snake_case）",
                    "type": "text",
                    "description": "属性描述"
                }
            ],
            "examples": ["示例entity1", "示例entity2"]
        }
    ],
    "edge_types": [
        {
            "name": "relationtypename（英文，UPPER_SNAKE_CASE）",
            "description": "简短描述（英文，不超过100字符）",
            "source_targets": [
                {"source": "源entitytype", "target": "目标entitytype"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "对文本content的简要analysis说明（中文）"
}
```

## 设计指南（极其重要！）

### 1. entitytype设计 - 必须严格遵守

**count要求：必须正好10个entitytype**

**层次结构要求（必须同时包含具体type和兜底type）**：

你的10个entitytype必须包含以下层次：

A. **兜底type（必须包含，放在list最后2个）**：
   - `Person`: 任何自然人个体的兜底type。当一个人不属于其他更具体的人物type时，归入此类。
   - `Organization`: 任何组织机构的兜底type。当一个组织不属于其他更具体的组织type时，归入此类。

B. **具体type（8个，根据文本content设计）**：
   - 针对文本中出现的主要role，设计更具体的type
   - 例如：如果文本涉及学术event，可以有 `Student`, `Professor`, `University`
   - 例如：如果文本涉及商业event，可以有 `Company`, `CEO`, `Employee`

**为什么需要兜底type**：
- 文本中会出现各种人物，如"中小学教师"、"路人甲"、"某位网友"
- 如果没有专门的type匹配，他们应该被归入 `Person`
- 同理，小型组织、临时团体等应该归入 `Organization`

**具体type的设计原则**：
- 从文本中识别出高频出现或关键的roletype
- 每个具体type应该有明确的边界，避免重叠
- description 必须清晰说明这个type和兜底type的区别

### 2. relationtype设计

- count：6-10个
- relation应该反映社媒互动中的真实联系
- 确保relation的 source_targets 涵盖你定义的entitytype

### 3. 属性设计

- 每个entitytype1-3个关键属性
- **注意**：属性名不能使用 `name`、`uuid`、`group_id`、`created_at`、`summary`（这些是系统保留字）
- recommendation使用：`full_name`, `title`, `role`, `position`, `location`, `description` 等

## entitytype参考

**个人类（具体）**：
- Student: 学生
- Professor: 教授/学者
- Journalist: 记者
- Celebrity: 明星/网红
- Executive: 高管
- Official: 政府官员
- Lawyer: 律师
- Doctor: 医生

**个人类（兜底）**：
- Person: 任何自然人（不属于上述具体type时使用）

**组织类（具体）**：
- University: 高校
- Company: 公司企业
- GovernmentAgency: 政府机构
- MediaOutlet: 媒体机构
- Hospital: 医院
- School: 中小学
- NGO: 非政府组织

**组织类（兜底）**：
- Organization: 任何组织机构（不属于上述具体type时使用）

## relationtype参考

- WORKS_FOR: 工作于
- STUDIES_AT: 就读于
- AFFILIATED_WITH: 隶属于
- REPRESENTS: 代表
- REGULATES: 监管
- REPORTS_ON: 报道
- COMMENTS_ON: comment
- RESPONDS_TO: 回应
- SUPPORTS: 支持
- OPPOSES: 反对
- COLLABORATES_WITH: 合作
- COMPETES_WITH: 竞争
"""


class OntologyGenerator:
    """
    ontology生成器
    analysis文本content，生成entity和relationtype定义
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
    
    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成ontology定义
        
        Args:
            document_texts: document文本list
            simulation_requirement: simulationrequirement描述
            additional_context: 额外上下文
            
        Returns:
            ontology定义（entity_types, edge_types等）
        """
        # Buildusermessage
        user_message = self._build_user_message(
            document_texts, 
            simulation_requirement,
            additional_context
        )
        
        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        # CallLLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )
        
        # Validate和后处理
        result = self._validate_and_process(result)
        
        return result
    
    # 传给 LLM 的文本max长度（5万字）
    MAX_TEXT_LENGTH_FOR_LLM = 50000
    
    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Buildusermessage"""
        
        # Merge文本
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)
        
        # If文本超过5万字，截断（仅影响传给LLM的content，不影响graph构建）
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(原文共{original_length}字，已截取前{self.MAX_TEXT_LENGTH_FOR_LLM}字用于ontologyanalysis)..."
        
        message = f"""## Simulationrequirement

{simulation_requirement}

## documentcontent

{combined_text}
"""
        
        if additional_context:
            message += f"""
## 额外说明

{additional_context}
"""
        
        message += """
请根据以上content，设计适合社会舆论simulation的entitytype和relationtype。

**必须遵守的规则**：
1. 必须正好output10个entitytype
2. 最后2个必须是兜底type：Person（个人兜底）和 Organization（组织兜底）
3. 前8个是根据文本content设计的具体type
4. 所有entitytype必须是现实中可以发声的主体，不能是抽象概念
5. 属性名不能使用 name、uuid、group_id 等保留字，用 full_name、org_name 等替代
"""
        
        return message
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证和后处理result"""
        
        # 确保必要字段存在
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        
        # Validateentitytype
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # 确保description不超过100字符
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
        
        # Validaterelationtype
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."
        
        # Zep API 限制：最多 10 个自定义entitytype，最多 10 个自定义边type
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10
        
        # 兜底type定义
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }
        
        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }
        
        # Check是否已有兜底type
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names
        
        # 需要add的兜底type
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)
        
        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)
            
            # Ifadd后会超过 10 个，需要移除一些现有type
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # Calculate需要移除多少个
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # 从末尾移除（保留前面更重要的具体type）
                result["entity_types"] = result["entity_types"][:-to_remove]
            
            # add兜底type
            result["entity_types"].extend(fallbacks_to_add)
        
        # 最终确保不超过限制（防御性编程）
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]
        
        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]
        
        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        将ontology定义转换为Python代码（类似ontology.py）
        
        Args:
            ontology: ontology定义
            
        Returns:
            Python代码字符串
        """
        code_lines = [
            '"""',
            '自定义entitytype定义',
            '由MiroFish自动生成，用于社会舆论simulation',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== entitytype定义 ==============',
            '',
        ]
        
        # Generateentitytype
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== relationtype定义 ==============')
        code_lines.append('')
        
        # Generaterelationtype
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # Convert为PascalCase类名
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        # Generatetype字典
        code_lines.append('# ============== typeconfig ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        
        # Generate边的source_targets映射
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)

