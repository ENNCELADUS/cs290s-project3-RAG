type Lang = "zh" | "en";

const labels = {
  zh: {
    title: "上科大信息学院 RAG 问答",
    subtitle: "基于官方数据的检索增强问答系统",
    placeholder: "输入关于上海科技大学或信息学院的问题...",
    send: "发送",
    sources: "参考来源",
    expandSources: "展开来源",
    collapseSources: "收起来源",
    settings: "设置",
    retrievalMode: "检索模式",
    topK: "返回数量",
    retrievalOnly: "仅检索模式",
    retrievalOnlyDesc: "只显示检索结果，不生成回答",
    noAnswer: "证据不足，无法回答此问题。",
    loading: "正在检索中...",
    generating: "正在生成回答...",
    emptyState: "输入问题开始提问，或点击下方示例",
    sampleTitle: "试试这些问题",
    statusFullRag: "完整 RAG 模式",
    statusRetrievalOnly: "仅检索模式",
    statusUnavailable: "服务不可用",
    statusConnecting: "连接中...",
    errorConnection: "无法连接到后端服务",
    errorGeneral: "请求出错，请重试",
    score: "分数",
    snippet: "摘要",
    modeHybrid: "混合",
    modeDense: "语义",
    modeBm25: "关键词",
    elapsed: "耗时",
  },
  en: {
    title: "ShanghaiTech SIST RAG Q&A",
    subtitle: "Retrieval-augmented QA over official university sources",
    placeholder: "Ask about ShanghaiTech University or SIST...",
    send: "Send",
    sources: "Sources",
    expandSources: "Expand sources",
    collapseSources: "Collapse sources",
    settings: "Settings",
    retrievalMode: "Retrieval Mode",
    topK: "Top-K",
    retrievalOnly: "Retrieval Only",
    retrievalOnlyDesc: "Show retrieved sources without generating an answer",
    noAnswer: "Insufficient evidence to answer this question.",
    loading: "Searching...",
    generating: "Generating answer...",
    emptyState: "Type a question or try the examples below",
    sampleTitle: "Try these questions",
    statusFullRag: "Full RAG",
    statusRetrievalOnly: "Retrieval Only",
    statusUnavailable: "Unavailable",
    statusConnecting: "Connecting...",
    errorConnection: "Cannot connect to backend",
    errorGeneral: "Request failed, please try again",
    score: "Score",
    snippet: "Snippet",
    modeHybrid: "Hybrid",
    modeDense: "Dense",
    modeBm25: "BM25",
    elapsed: "Elapsed",
  },
} as const;

export type Labels = { [K in keyof (typeof labels)["en"]]: string };

function hasCJK(text: string): boolean {
  return /[一-鿿]/.test(text);
}

export function detectLang(text?: string): Lang {
  if (text && hasCJK(text)) return "zh";
  if (typeof navigator !== "undefined" && navigator.language.startsWith("zh")) return "zh";
  return "en";
}

export function getLabels(lang: Lang): Labels {
  return labels[lang] as Labels;
}

export function useLabels(text?: string): Labels {
  return getLabels(detectLang(text));
}
