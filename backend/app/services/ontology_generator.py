"""
ontology生成服务
API1：analysis文本content，生成适合社会simulation的entity和relationtype定义
"""

import json
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient


# ontology生成的系统提示词
ONTOLOGY_SYSTEM_PROMPT = """You are a professional knowledge graph ontology design expert. Your task is to analyze the given text content and simulation requirements, and design entity types and relation types suitable for **social media opinion simulation**.

**Important: You must output valid JSON format data only, no other content.**

## Core Task Background

We are building a **social media opinion simulation system**. In this system:
- Each entity is an "account" or "actor" that can post, interact, and spread information on social media
- Entities influence each other through reposts, comments, and responses
- We need to simulate reactions and information propagation paths in opinion events

Therefore, **entities must be real-world actors that can post and interact on social media**:

**Allowed**:
- Specific individuals (public figures, stakeholders, opinion leaders, experts, ordinary people)
- Companies and businesses (including their official accounts)
- Organizations (universities, associations, NGOs, unions, etc.)
- Government departments, regulatory bodies
- Media organizations (newspapers, TV stations, independent media, websites)
- Social media platforms themselves
- Specific group representatives (alumni associations, fan groups, advocacy groups, etc.)

**Not allowed**:
- Abstract concepts (e.g. "public opinion", "sentiment", "trends")
- Topics (e.g. "academic integrity", "education reform")
- Opinions/attitudes (e.g. "supporters", "opponents")

## Output Format

Output JSON with the following structure:

```json
{
    "entity_types": [
        {
            "name": "EntityTypeName (English, PascalCase)",
            "description": "Brief description (English, max 100 chars)",
            "attributes": [
                {
                    "name": "attribute_name (English, snake_case)",
                    "type": "text",
                    "description": "Attribute description"
                }
            ],
            "examples": ["Example entity 1", "Example entity 2"]
        }
    ],
    "edge_types": [
        {
            "name": "RELATION_TYPE_NAME (English, UPPER_SNAKE_CASE)",
            "description": "Brief description (English, max 100 chars)",
            "source_targets": [
                {"source": "SourceEntityType", "target": "TargetEntityType"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Brief analysis of the text content (in Norwegian)"
}
```

## Design Guidelines (Critical!)

### 1. Entity Type Design - Must Follow Strictly

**Count requirement: Exactly 10 entity types**

**Hierarchy requirement (must include both specific and fallback types)**:

Your 10 entity types must include:

A. **Fallback types (required, place last 2 in list)**:
   - `Person`: Fallback for any individual. Used when a person doesn't fit a more specific type.
   - `Organization`: Fallback for any organization. Used when an org doesn't fit a more specific type.

B. **Specific types (8, designed based on text content)**:
   - Design specific types for the main roles appearing in the text
   - Example: For political events: `Politician`, `PoliticalParty`, `Voter`, `MediaOutlet`
   - Example: For sports events: `Athlete`, `Team`, `Coach`, `SportsFederation`

**Why fallback types are needed**:
- Text will contain various people who don't fit specific categories
- Without a matching type, they should fall into `Person`
- Similarly, small organizations should fall into `Organization`

**Specific type design principles**:
- Identify high-frequency or key role types from the text
- Each type should have clear boundaries, no overlap
- Description must clearly explain how this type differs from the fallback

### 2. Relation Type Design

- Count: 6-10 relation types
- Relations should reflect real connections in social media interactions
- Ensure relation source_targets cover your defined entity types

### 3. Attribute Design

- 1-3 key attributes per entity type
- **Note**: Attribute names cannot use `name`, `uuid`, `group_id`, `created_at`, `summary` (reserved)
- Recommended: `full_name`, `title`, `role`, `position`, `location`, `description`

## Entity Type Reference

**Individual (specific)**: Student, Professor, Journalist, Celebrity, Executive, Official, Politician, Lawyer, Doctor
**Individual (fallback)**: Person
**Organization (specific)**: University, Company, GovernmentAgency, MediaOutlet, PoliticalParty, Hospital, NGO
**Organization (fallback)**: Organization

## Relation Type Reference

- WORKS_FOR, STUDIES_AT, AFFILIATED_WITH, REPRESENTS, REGULATES
- REPORTS_ON, COMMENTS_ON, RESPONDS_TO, SUPPORTS, OPPOSES
- COLLABORATES_WITH, COMPETES_WITH
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

