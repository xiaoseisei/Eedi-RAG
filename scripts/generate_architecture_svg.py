import os

svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 980" width="100%" height="100%">
  <defs>
    <!-- Filter for soft shadow -->
    <filter id="card-shadow" x="-4%" y="-4%" width="108%" height="108%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#1B3A57" flood-opacity="0.08"/>
    </filter>
    <filter id="chip-shadow" x="-4%" y="-6%" width="108%" height="114%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#1B3A57" flood-opacity="0.05"/>
    </filter>

    <!-- Arrow markers -->
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#2B5B84"/>
    </marker>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#1B8755"/>
    </marker>
    <marker id="arrow-purple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#7C3AED"/>
    </marker>
    <marker id="arrow-orange" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#EA580C"/>
    </marker>

    <!-- Gradients -->
    <linearGradient id="bg-canvas" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#E8F1F8"/>
      <stop offset="100%" stop-color="#DEE9F3"/>
    </linearGradient>
    <linearGradient id="pill-yellow" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FEF3C7"/>
      <stop offset="100%" stop-color="#FDE68A"/>
    </linearGradient>
    <linearGradient id="pill-blue" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#E0F2FE"/>
      <stop offset="100%" stop-color="#BAE6FD"/>
    </linearGradient>
    <linearGradient id="pill-green" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#DCFCE7"/>
      <stop offset="100%" stop-color="#BBF7D0"/>
    </linearGradient>
    <linearGradient id="pill-purple" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#F3E8FF"/>
      <stop offset="100%" stop-color="#E9D5FF"/>
    </linearGradient>
    <linearGradient id="pill-orange" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FFEDD5"/>
      <stop offset="100%" stop-color="#FED7AA"/>
    </linearGradient>

    <!-- Chip Gradient -->
    <linearGradient id="chip-blue" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#F0F7FF"/>
      <stop offset="100%" stop-color="#E1EFFF"/>
    </linearGradient>
    <linearGradient id="chip-cyan" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#F0FDFA"/>
      <stop offset="100%" stop-color="#CCFBF1"/>
    </linearGradient>
    <linearGradient id="chip-purple" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FAF5FF"/>
      <stop offset="100%" stop-color="#F3E8FF"/>
    </linearGradient>
    <linearGradient id="chip-amber" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FFFBEB"/>
      <stop offset="100%" stop-color="#FEF3C7"/>
    </linearGradient>
    <linearGradient id="chip-green" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#F0FDF4"/>
      <stop offset="100%" stop-color="#DCFCE7"/>
    </linearGradient>
  </defs>

  <style>
    text {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    }
    .card-container {
      fill: #FFFFFF;
      stroke: #2E5B82;
      stroke-width: 1.8;
      stroke-dasharray: 6,4;
      rx: 14;
      ry: 14;
      filter: url(#card-shadow);
    }
    .card-container-accent {
      fill: #FFFFFF;
      stroke: #0284C7;
      stroke-width: 2.2;
      stroke-dasharray: 6,4;
      rx: 14;
      ry: 14;
      filter: url(#card-shadow);
    }
    .pill-header {
      rx: 14;
      ry: 14;
      stroke-width: 1.2;
    }
    .pill-title {
      font-size: 13px;
      font-weight: 700;
      text-anchor: middle;
      dominant-baseline: central;
    }
    .chip {
      rx: 8;
      ry: 8;
      stroke-width: 1.2;
      filter: url(#chip-shadow);
    }
    .chip-title {
      font-size: 11.5px;
      font-weight: 600;
      text-anchor: middle;
      dominant-baseline: central;
    }
    .chip-sub {
      font-size: 9.5px;
      fill: #4B647A;
      text-anchor: middle;
      dominant-baseline: central;
    }
    .flow-line {
      fill: none;
      stroke: #2B5B84;
      stroke-width: 2.2;
      stroke-linecap: round;
      stroke-linejoin: round;
      marker-end: url(#arrow);
    }
    .flow-line-dashed {
      fill: none;
      stroke: #0284C7;
      stroke-width: 2;
      stroke-dasharray: 5,4;
      marker-end: url(#arrow);
    }
    .flow-label {
      font-size: 10px;
      font-weight: 700;
      fill: #1E3A8A;
      text-anchor: middle;
      dominant-baseline: central;
    }
    .flow-label-bg {
      fill: #FFFFFF;
      rx: 4;
      ry: 4;
      stroke: #93C5FD;
      stroke-width: 1;
    }
    .canvas-title {
      font-size: 22px;
      font-weight: 800;
      fill: #0F2942;
      letter-spacing: 0.5px;
    }
    .canvas-sub {
      font-size: 12px;
      font-weight: 500;
      fill: #47637E;
    }
  </style>

  <!-- Canvas Background -->
  <rect x="0" y="0" width="1280" height="980" fill="url(#bg-canvas)" rx="16" ry="16"/>

  <!-- Canvas Header -->
  <g transform="translate(40, 24)">
    <text x="0" y="24" class="canvas-title">Eedi-RAG 系统全景架构与端到端程序数据流图</text>
    <text x="0" y="44" class="canvas-sub">基于 1,576 场真实初等数学课堂对白 | 离线双轨知识抽取 ➔ 多路混合检索 ➔ 9.7 贪心预算装配 ➔ 契约流式生成 ➔ 对白内联闭环审计</text>
  </g>

  <!-- ========================================================================================== -->
  <!-- 1. TOP ROW: Client Interaction / Query Gateway / Evaluation Plane -->
  <!-- ========================================================================================== -->

  <!-- Card 1: 用户交互与接入层 -->
  <g id="box-clients" transform="translate(40, 85)">
    <rect width="320" height="150" class="card-container"/>
    <!-- Pill Header -->
    <rect x="60" y="-14" width="200" height="28" fill="url(#pill-yellow)" stroke="#D97706" class="pill-header"/>
    <text x="160" y="0" fill="#92400E" class="pill-title">🖥️ 用户交互与接入终端</text>
    <!-- Chips -->
    <g transform="translate(18, 26)">
      <rect width="136" height="46" fill="url(#chip-blue)" stroke="#93C5FD" class="chip"/>
      <text x="68" y="16" fill="#1E3A8A" class="chip-title">教研人员 Web 终端</text>
      <text x="68" y="32" class="chip-sub">多轮问答 / 学情看板</text>
    </g>
    <g transform="translate(166, 26)">
      <rect width="136" height="46" fill="url(#chip-blue)" stroke="#93C5FD" class="chip"/>
      <text x="68" y="16" fill="#1E3A8A" class="chip-title">交互式 CLI 调试器</text>
      <text x="68" y="32" class="chip-sub">interactive_cli.py</text>
    </g>
    <g transform="translate(18, 84)">
      <rect width="284" height="48" fill="url(#chip-cyan)" stroke="#5EEAD4" class="chip"/>
      <text x="142" y="18" fill="#134E4A" class="chip-title">RESTful API &amp; SSE 流式输出端点</text>
      <text x="142" y="34" class="chip-sub">支持 Server-Sent Events / 首字延迟 (TTFT ~ 0.8s) 实时度量</text>
    </g>
  </g>

  <!-- Card 2: 意图识别与查询治理 -->
  <g id="box-gateway" transform="translate(400, 85)">
    <rect width="460" height="150" class="card-container"/>
    <!-- Pill Header -->
    <rect x="130" y="-14" width="200" height="28" fill="url(#pill-blue)" stroke="#0284C7" class="pill-header"/>
    <text x="230" y="0" fill="#0369A1" class="pill-title">🔍 意图识别与查询治理网关</text>
    <!-- Chips -->
    <g transform="translate(20, 26)">
      <rect width="200" height="46" fill="url(#chip-amber)" stroke="#FCD34D" class="chip"/>
      <text x="100" y="16" fill="#78350F" class="chip-title">💬 自然语言教研提问</text>
      <text x="100" y="32" class="chip-sub">输入例: 5.45 四舍五入学生常犯何错?</text>
    </g>
    <g transform="translate(240, 26)">
      <rect width="200" height="46" fill="url(#chip-purple)" stroke="#D8B4FE" class="chip"/>
      <text x="100" y="16" fill="#581C87" class="chip-title">🎯 实体与考点识别分流</text>
      <text x="100" y="32" class="chip-sub">错因诊断 / 策略引导 / 综合问答</text>
    </g>
    <g transform="translate(20, 84)">
      <rect width="420" height="48" fill="url(#chip-blue)" stroke="#93C5FD" class="chip"/>
      <text x="210" y="18" fill="#1E3A8A" class="chip-title">🛡️ 保护性查询改写 (Protected Token Preservation)</text>
      <text x="210" y="34" class="chip-sub">强制保护数学公式、题号、选项等核心实体 (100% 免受改写损毁)</text>
    </g>
  </g>

  <!-- Card 3: 自动化评测与监控控制面 -->
  <g id="box-eval" transform="translate(900, 85)">
    <rect width="340" height="150" class="card-container"/>
    <!-- Pill Header -->
    <rect x="70" y="-14" width="200" height="28" fill="url(#pill-purple)" stroke="#9333EA" class="pill-header"/>
    <text x="170" y="0" fill="#6B21A8" class="pill-title">📊 评测控制台与指标监控</text>
    <!-- Chips -->
    <g transform="translate(18, 26)">
      <rect width="144" height="46" fill="url(#chip-green)" stroke="#86EFAC" class="chip"/>
      <text x="72" y="16" fill="#14532D" class="chip-title">L1 检索诊断门禁</text>
      <text x="72" y="32" class="chip-sub">Turn Recall 81.11%</text>
    </g>
    <g transform="translate(178, 26)">
      <rect width="144" height="46" fill="url(#chip-green)" stroke="#86EFAC" class="chip"/>
      <text x="72" y="16" fill="#14532D" class="chip-title">L2 DeepEval 裁判</text>
      <text x="72" y="32" class="chip-sub">Contextual Recall 98.3%</text>
    </g>
    <g transform="translate(18, 84)">
      <rect width="304" height="48" fill="url(#chip-amber)" stroke="#FCD34D" class="chip"/>
      <text x="152" y="18" fill="#78350F" class="chip-title">⚡ 全链路耗时审计 (Performance Tracing)</text>
      <text x="152" y="34" class="chip-sub">装配 P95: 7.25ms | 首字 TTFT: 0.82s | 全套 282 测试通过</text>
    </g>
  </g>

  <!-- ========================================================================================== -->
  <!-- 2. MIDDLE ROW: Offline Pipeline / Core Search & Assembler / Generation & Audit -->
  <!-- ========================================================================================== -->

  <!-- Card 4: 离线知识工程流水线 (Left) -->
  <g id="box-offline" transform="translate(40, 275)">
    <rect width="320" height="400" class="card-container"/>
    <!-- Pill Header -->
    <rect x="50" y="-14" width="220" height="28" fill="url(#pill-orange)" stroke="#EA580C" class="pill-header"/>
    <text x="160" y="0" fill="#9A3412" class="pill-title">📁 离线知识工程构建流水线</text>
    <!-- Chips (Stacked vertically) -->
    <g transform="translate(18, 24)">
      <rect width="284" height="46" fill="url(#chip-blue)" stroke="#93C5FD" class="chip"/>
      <text x="142" y="16" fill="#1E3A8A" class="chip-title">📁 Eedi 官方课堂对白源数据</text>
      <text x="142" y="32" class="chip-sub">1,576 场会话 / 30,000+ 原声轮次 / 题干选项</text>
    </g>
    <g transform="translate(18, 80)">
      <rect width="284" height="46" fill="url(#chip-blue)" stroke="#93C5FD" class="chip"/>
      <text x="142" y="16" fill="#1E3A8A" class="chip-title">🧹 会话清洗与发言角色对齐</text>
      <text x="142" y="32" class="chip-sub">分离 Student / Tutor，规范化题干元数据</text>
    </g>
    <g transform="translate(18, 136)">
      <rect width="136" height="52" fill="url(#chip-purple)" stroke="#D8B4FE" class="chip"/>
      <text x="68" y="18" fill="#581C87" class="chip-title">🧠 错因机理卡抽取</text>
      <text x="68" y="36" class="chip-sub">学情障碍/错误选项绑定</text>
    </g>
    <g transform="translate(166, 136)">
      <rect width="136" height="52" fill="url(#chip-purple)" stroke="#D8B4FE" class="chip"/>
      <text x="68" y="18" fill="#581C87" class="chip-title">💡 名师策略卡抽取</text>
      <text x="68" y="36" class="chip-sub">苏格拉底/破局一问</text>
    </g>
    <g transform="translate(18, 200)">
      <rect width="284" height="52" fill="url(#chip-amber)" stroke="#F59E0B" class="chip"/>
      <text x="142" y="18" fill="#78350F" class="chip-title">🛡️ 对白指针逐字强校验 (Fail-Fast)</text>
      <text x="142" y="36" class="chip-sub">验证 Turn ID 权威映射，逐字保真度 100% 门禁</text>
    </g>
    <g transform="translate(18, 264)">
      <rect width="284" height="46" fill="url(#chip-cyan)" stroke="#5EEAD4" class="chip"/>
      <text x="142" y="16" fill="#134E4A" class="chip-title">🪟 冷备滑动窗口切片生成</text>
      <text x="142" y="32" class="chip-sub">Window=7, Stride=3 时序对白完整切分</text>
    </g>
    <g transform="translate(18, 322)">
      <rect width="284" height="54" fill="url(#chip-green)" stroke="#86EFAC" class="chip"/>
      <text x="142" y="18" fill="#14532D" class="chip-title">🔄 双引擎原子同步写入</text>
      <text x="142" y="36" class="chip-sub">DuckDB 关系底表 + ChromaDB 多物理隔离向量库</text>
    </g>
  </g>

  <!-- Card 5: 核心检索与 9.7 贪心装配引擎 (Center) -->
  <g id="box-core" transform="translate(400, 275)">
    <rect width="460" height="400" class="card-container-accent"/>
    <!-- Pill Header -->
    <rect x="100" y="-14" width="260" height="28" fill="url(#pill-yellow)" stroke="#D97706" class="pill-header"/>
    <text x="230" y="0" fill="#92400E" class="pill-title">⚡ 核心多路检索与 9.7 极速装配引擎</text>
    
    <!-- Row 1: Dual Retrieval -->
    <g transform="translate(20, 24)">
      <rect width="200" height="50" fill="url(#chip-cyan)" stroke="#5EEAD4" class="chip"/>
      <text x="100" y="18" fill="#134E4A" class="chip-title">⚡ BM25 稀疏检索</text>
      <text x="100" y="34" class="chip-sub">精确命中题干数字、方程式、专业术语</text>
    </g>
    <g transform="translate(240, 24)">
      <rect width="200" height="50" fill="url(#chip-cyan)" stroke="#5EEAD4" class="chip"/>
      <text x="100" y="18" fill="#134E4A" class="chip-title">🎯 Qwen3-Embedding (1024d)</text>
      <text x="100" y="34" class="chip-sub">稠密语义检索，理解深层困惑意图</text>
    </g>

    <!-- Row 2: RRF & Reranker -->
    <g transform="translate(20, 86)">
      <rect width="420" height="48" fill="url(#chip-blue)" stroke="#93C5FD" class="chip"/>
      <text x="210" y="18" fill="#1E3A8A" class="chip-title">🔀 倒数秩融合 (Reciprocal Rank Fusion, RRF)</text>
      <text x="210" y="34" class="chip-sub">多路融合生成 Top-20 混合初筛候选池 (黄金证据覆盖率 100%)</text>
    </g>
    <g transform="translate(20, 146)">
      <rect width="420" height="52" fill="url(#chip-purple)" stroke="#D8B4FE" class="chip"/>
      <text x="210" y="18" fill="#581C87" class="chip-title">🧠 Qwen3-Reranker-0.6B Cross-Encoder 深度重排序</text>
      <text x="210" y="36" class="chip-sub">Top-15 最佳候选池精排，前 5 位相关切片精准率大幅提升 +24%</text>
    </g>

    <!-- Row 3: 9.7 Breakthrough Architecture -->
    <g transform="translate(20, 210)">
      <rect width="200" height="58" fill="url(#chip-green)" stroke="#86EFAC" class="chip"/>
      <text x="100" y="20" fill="#14532D" class="chip-title">📌 Parent-5 核心卡片锚定</text>
      <text x="100" y="40" class="chip-sub">锁定 Top-5 核心错因与策略卡<br/>以其 source_turns 作为对白锚点</text>
    </g>
    <g transform="translate(240, 210)">
      <rect width="200" height="58" fill="url(#chip-green)" stroke="#86EFAC" class="chip"/>
      <text x="100" y="20" fill="#14532D" class="chip-title">🪟 Window-7/Stride-3 展开</text>
      <text x="100" y="40" class="chip-sub">前后双向扩展各 3 轮交互<br/>覆盖连续 7 轮真实师生对白</text>
    </g>

    <g transform="translate(20, 280)">
      <rect width="420" height="56" fill="url(#chip-amber)" stroke="#F59E0B" class="chip"/>
      <text x="210" y="20" fill="#78350F" class="chip-title">⚡ greedy_budget 确定性贪心装配引擎 (9.7 核心突破)</text>
      <text x="210" y="40" class="chip-sub">彻底消除动态规划超时 | 装配延迟降至 7.25ms (提速 1,128 倍) | 4000 Token 预算</text>
    </g>

    <g transform="translate(20, 348)">
      <rect width="420" height="36" fill="url(#chip-blue)" stroke="#60A5FA" class="chip"/>
      <text x="210" y="18" fill="#1E3A8A" class="chip-title">📄 结构化上下文生成 (题干 + 考点 + 7 组连贯逻辑窗口 + 编号证据)</text>
    </g>
  </g>

  <!-- Card 6: 契约生成与对白权威审计 (Right) -->
  <g id="box-gen" transform="translate(900, 275)">
    <rect width="340" height="400" class="card-container"/>
    <!-- Pill Header -->
    <rect x="60" y="-14" width="220" height="28" fill="url(#pill-green)" stroke="#16A34A" class="pill-header"/>
    <text x="170" y="0" fill="#15803D" class="pill-title">🤖 契约生成与对白双向审计</text>
    <!-- Chips -->
    <g transform="translate(18, 24)">
      <rect width="304" height="50" fill="url(#chip-blue)" stroke="#93C5FD" class="chip"/>
      <text x="152" y="18" fill="#1E3A8A" class="chip-title">🤖 LLM 纯文本流式推理 (Mimo / Qwen)</text>
      <text x="152" y="34" class="chip-sub">首字延迟 TTFT P50: 0.82s | 输出包含 [E*] 引用</text>
    </g>
    <g transform="translate(18, 86)">
      <rect width="304" height="54" fill="url(#chip-purple)" stroke="#D8B4FE" class="chip"/>
      <text x="152" y="18" fill="#581C87" class="chip-title">📋 Pydantic 强类型契约解析校验</text>
      <text x="152" y="36" class="chip-sub">思维链 + 学情障碍机理 + 破局一问 + 证据标号声明</text>
    </g>
    <g transform="translate(18, 152)">
      <rect width="304" height="60" fill="url(#chip-amber)" stroke="#F59E0B" class="chip"/>
      <text x="152" y="20" fill="#78350F" class="chip-title">🛡️ DuckDB 对白内联双向审计门禁</text>
      <text x="152" y="42" class="chip-sub">100% 核实正文引用在 DuckDB 中真实存在<br/>说话人角色、会话 ID、逐字原声严格对齐</text>
    </g>
    <g transform="translate(18, 224)">
      <rect width="304" height="50" fill="#FEE2E2" stroke="#EF4444" class="chip"/>
      <text x="152" y="18" fill="#991B1B" class="chip-title">🚫 零幻觉拦截 (CitationAuditError)</text>
      <text x="152" y="34" fill="#991B1B" class="chip-sub">凭空捏造、张冠李戴立即拦截，绝不伪造输出</text>
    </g>
    <g transform="translate(18, 286)">
      <rect width="304" height="96" fill="url(#chip-green)" stroke="#22C55E" class="chip"/>
      <text x="152" y="22" fill="#14532D" class="chip-title" font-size="13px">🎉 合规高质量教研答复渲染</text>
      <text x="152" y="44" class="chip-sub">✅ 诊断: 错误选项背后认知根因剖析</text>
      <text x="152" y="62" class="chip-sub">✅ 引导: 名师苏格拉底式破局一问与动作</text>
      <text x="152" y="80" class="chip-sub">✅ 出处: 权威标注 [E01 Session#10 Turn 4]</text>
    </g>
  </g>

  <!-- ========================================================================================== -->
  <!-- 3. BOTTOM ROW: Dual Engine Storage Base -->
  <!-- ========================================================================================== -->

  <g id="box-storage" transform="translate(40, 715)">
    <rect width="1200" height="230" class="card-container"/>
    <!-- Pill Header -->
    <rect x="460" y="-16" width="280" height="32" fill="url(#pill-blue)" stroke="#0284C7" class="pill-header"/>
    <text x="600" y="0" fill="#0369A1" class="pill-title" font-size="14px">🗄️ 双引擎分层数据存储底座 (Dual Engine Storage Base)</text>

    <!-- 6 Columns of Storage Subsystems -->
    <!-- Col 1 -->
    <g transform="translate(24, 30)">
      <rect width="176" height="175" fill="url(#chip-blue)" stroke="#93C5FD" class="chip"/>
      <text x="88" y="24" fill="#1E3A8A" class="chip-title">🗄️ DuckDB 会话事实表</text>
      <text x="88" y="48" class="chip-sub">tutoring_sessions</text>
      <line x1="16" y1="62" x2="160" y2="62" stroke="#BFDBFE" stroke-width="1"/>
      <text x="88" y="84" class="chip-sub">1,576 场完整辅导</text>
      <text x="88" y="104" class="chip-sub">原题题干 / 选项</text>
      <text x="88" y="124" class="chip-sub">正确答案 / 错误项</text>
      <text x="88" y="144" class="chip-sub">毫秒级主键检索</text>
    </g>

    <!-- Col 2 -->
    <g transform="translate(220, 30)">
      <rect width="176" height="175" fill="url(#chip-blue)" stroke="#93C5FD" class="chip"/>
      <text x="88" y="24" fill="#1E3A8A" class="chip-title">🗄️ DuckDB 对白实录表</text>
      <text x="88" y="48" class="chip-sub">session_dialogue_turns</text>
      <line x1="16" y1="62" x2="160" y2="62" stroke="#BFDBFE" stroke-width="1"/>
      <text x="88" y="84" class="chip-sub">30,000+ 轮时序对话</text>
      <text x="88" y="104" class="chip-sub">Student / Tutor 角色</text>
      <text x="88" y="124" class="chip-sub">不可篡改权威文本</text>
      <text x="88" y="144" class="chip-sub">权威引用溯源基石</text>
    </g>

    <!-- Col 3 -->
    <g transform="translate(416, 30)">
      <rect width="176" height="175" fill="url(#chip-blue)" stroke="#93C5FD" class="chip"/>
      <text x="88" y="24" fill="#1E3A8A" class="chip-title">📊 DuckDB 知识卡片表</text>
      <text x="88" y="48" class="chip-sub">misconception / strategy</text>
      <line x1="16" y1="62" x2="160" y2="62" stroke="#BFDBFE" stroke-width="1"/>
      <text x="88" y="84" class="chip-sub">错因机理结构化数据</text>
      <text x="88" y="104" class="chip-sub">破局一问 / 教学动作</text>
      <text x="88" y="124" class="chip-sub">强指针: source_turns</text>
      <text x="88" y="144" class="chip-sub">双轨事实与向量映射</text>
    </g>

    <!-- Col 4 -->
    <g transform="translate(612, 30)">
      <rect width="176" height="175" fill="url(#chip-cyan)" stroke="#5EEAD4" class="chip"/>
      <text x="88" y="24" fill="#134E4A" class="chip-title">🧠 ChromaDB 错因向量库</text>
      <text x="88" y="48" class="chip-sub">student_misconceptions</text>
      <line x1="16" y1="62" x2="160" y2="62" stroke="#99F6E4" stroke-width="1"/>
      <text x="88" y="84" class="chip-sub">Qwen3-0.6B (1024d)</text>
      <text x="88" y="104" class="chip-sub">深层学生思维误区向量</text>
      <text x="88" y="124" class="chip-sub">元数据过滤: QID/考点</text>
      <text x="88" y="144" class="chip-sub">物理隔离独立索引</text>
    </g>

    <!-- Col 5 -->
    <g transform="translate(808, 30)">
      <rect width="176" height="175" fill="url(#chip-cyan)" stroke="#5EEAD4" class="chip"/>
      <text x="88" y="24" fill="#134E4A" class="chip-title">💡 ChromaDB 策略向量库</text>
      <text x="88" y="48" class="chip-sub">tutor_strategies</text>
      <line x1="16" y1="62" x2="160" y2="62" stroke="#99F6E4" stroke-width="1"/>
      <text x="88" y="84" class="chip-sub">Qwen3-0.6B (1024d)</text>
      <text x="88" y="104" class="chip-sub">名师启发提问与脚手架</text>
      <text x="88" y="124" class="chip-sub">教学动作与策略分类</text>
      <text x="88" y="144" class="chip-sub">物理隔离独立索引</text>
    </g>

    <!-- Col 6 -->
    <g transform="translate(1004, 30)">
      <rect width="176" height="175" fill="url(#chip-cyan)" stroke="#5EEAD4" class="chip"/>
      <text x="88" y="24" fill="#134E4A" class="chip-title">⚡ ChromaDB 冷备滑动窗口</text>
      <text x="88" y="48" class="chip-sub">fallback_windows</text>
      <line x1="16" y1="62" x2="160" y2="62" stroke="#99F6E4" stroke-width="1"/>
      <text x="88" y="84" class="chip-sub">Window=7, Stride=3</text>
      <text x="88" y="104" class="chip-sub">零信任纯原文冷备</text>
      <text x="88" y="124" class="chip-sub">当卡片失焦时兜底直取</text>
      <text x="88" y="144" class="chip-sub">覆盖全量连续对白切块</text>
    </g>
  </g>

  <!-- ========================================================================================== -->
  <!-- 4. CONNECTING FLOW LINES & ANNOTATIONS -->
  <!-- ========================================================================================== -->

  <!-- Top: Client -> Gateway -->
  <path d="M 360 160 L 400 160" class="flow-line"/>

  <!-- Top: Gateway -> Core -->
  <path d="M 630 235 L 630 275" class="flow-line"/>
  <rect x="585" y="245" width="90" height="20" class="flow-label-bg"/>
  <text x="630" y="255" class="flow-label">保护性改写请求</text>

  <!-- Left: Offline -> Bottom Storage -->
  <path d="M 200 675 L 200 715" class="flow-line" stroke="#EA580C" marker-end="url(#arrow-orange)"/>
  <rect x="155" y="685" width="90" height="20" class="flow-label-bg" stroke="#FDBA74"/>
  <text x="200" y="695" class="flow-label" fill="#9A3412">双轨原子写入</text>

  <!-- Bottom Storage -> Center Core (Retrieval) -->
  <path d="M 630 715 L 630 675" class="flow-line" stroke="#0D9488" marker-end="url(#arrow-green)"/>
  <rect x="575" y="685" width="110" height="20" class="flow-label-bg" stroke="#5EEAD4"/>
  <text x="630" y="695" class="flow-label" fill="#0F766E">混合检索 / 向量匹配</text>

  <!-- Center Core -> Right Generation -->
  <path d="M 860 475 L 900 475" class="flow-line" stroke="#7C3AED" marker-end="url(#arrow-purple)"/>
  <rect x="850" y="445" width="60" height="20" class="flow-label-bg" stroke="#C4B5FD"/>
  <text x="880" y="455" class="flow-label" fill="#6D28D9">装配上下文</text>

  <!-- Right Generation -> Bottom Storage (Citation Audit lookup) -->
  <path d="M 1070 675 L 1070 715" class="flow-line" stroke="#16A34A" marker-end="url(#arrow-green)"/>
  <rect x="1015" y="685" width="110" height="20" class="flow-label-bg" stroke="#86EFAC"/>
  <text x="1070" y="695" class="flow-label" fill="#15803D">Turn ID 权威逐字核查</text>

  <!-- Right Generation -> Top Client (Feedback Loop / Final Output) -->
  <path d="M 1070 275 L 1070 215 Q 1070 190 1000 190 L 860 190" class="flow-line-dashed"/>
  <path d="M 400 190 L 360 190" class="flow-line-dashed"/>
  <rect x="880" y="200" width="100" height="20" class="flow-label-bg" stroke="#BAE6FD"/>
  <text x="930" y="210" class="flow-label" fill="#0369A1">流式答复传输 (SSE)</text>

</svg>
"""

assets_dir = r"E:\PIAgent\10-projects\AgentLearn\Eedi-RAG\docs\assets"
os.makedirs(assets_dir, exist_ok=True)
svg_path = os.path.join(assets_dir, "architecture.svg")

with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg_content.strip())

print(f"Generated architecture SVG at: {svg_path} ({len(svg_content)} bytes)")
