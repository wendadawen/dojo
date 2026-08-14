(function attachDojoHomeModel(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.DojoHomeModel = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function createModel() {
  const TYPE_LABELS = { paper: "论文", concept: "概念", note: "笔记", unknown: "未分类" };

  function normalize(value) {
    return String(value || "").normalize("NFKC").toLocaleLowerCase();
  }

  function escapeHtml(value) {
    return String(value || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function typeLabel(type) {
    return TYPE_LABELS[type] || type || "未分类";
  }

  function filterPages(pages, state) {
    const query = normalize(state.query);
    return pages
      .filter((page) => !state.type || page.type === state.type)
      .filter((page) => !state.topic || (page.topics || []).includes(state.topic))
      .filter((page) => {
        if (!query) return true;
        return normalize([
          page.title,
          page.description,
          page.tag,
          ...(page.topics || []),
        ].join(" ")).includes(query);
      })
      .slice()
      .sort((a, b) => a.title.localeCompare(b.title, "zh-CN"));
  }

  function getFilterOptions(pages) {
    return {
      types: [...new Set(pages.map((page) => page.type).filter(Boolean))].sort(),
      topics: [...new Set(pages.flatMap((page) => page.topics || []))]
        .sort((a, b) => a.localeCompare(b, "zh-CN")),
    };
  }

  function nodeFor(page, role = "normal") {
    return {
      data: {
        id: page.id,
        label: page.title,
        path: page.path,
        type: page.type,
        topics: page.topics || [],
        role,
      },
    };
  }

  function edgeFor(edge) {
    return {
      data: {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        count: edge.count,
      },
    };
  }

  function makeGlobalElements(catalog, visibleIds) {
    const allowed = visibleIds || new Set(catalog.pages.map((page) => page.id));
    return {
      nodes: catalog.pages.filter((page) => allowed.has(page.id)).map((page) => nodeFor(page)),
      edges: catalog.edges
        .filter((edge) => allowed.has(edge.source) && allowed.has(edge.target))
        .map(edgeFor),
    };
  }

  function makeLocalGraph(catalog, centerId) {
    const pages = new Map(catalog.pages.map((page) => [page.id, page]));
    const center = pages.get(centerId);
    if (!center) return { nodes: [], edges: [] };
    const incoming = new Set(center.incoming || []);
    const outgoing = new Set(center.outgoing || []);
    const neighbors = [...new Set([...incoming, ...outgoing])].sort();
    const nodes = [nodeFor(center, "center")];
    for (const id of neighbors) {
      const page = pages.get(id);
      if (!page) continue;
      let role = incoming.has(id) ? "incoming" : "outgoing";
      if (incoming.has(id) && outgoing.has(id)) role = "both";
      nodes.push(nodeFor(page, role));
    }
    const ids = new Set(nodes.map((node) => node.data.id));
    const edges = catalog.edges
      .filter((edge) => ids.has(edge.source) && ids.has(edge.target))
      .filter((edge) => edge.source === centerId || edge.target === centerId)
      .map(edgeFor);
    return { nodes, edges };
  }

  function renderCard(page) {
    const topics = (page.topics || [])
      .map((topic) => `<span class="topic-chip">${escapeHtml(topic)}</span>`)
      .join("");
    return `
      <article class="document-card">
        <button class="card-main" type="button" data-open-path="${escapeHtml(page.path)}">
          <span class="card-meta">
            <span class="type-badge type-${escapeHtml(page.type)}">${escapeHtml(typeLabel(page.type))}</span>
            ${topics}
          </span>
          <h2>${escapeHtml(page.title)}</h2>
          <p>${escapeHtml(page.description || "暂无摘要")}</p>
        </button>
        <button class="relation-button" type="button" data-relation-id="${escapeHtml(page.id)}">
          入链 ${page.incoming_count || 0} · 出链 ${page.outgoing_count || 0}
        </button>
      </article>`;
  }

  return {
    escapeHtml,
    filterPages,
    getFilterOptions,
    makeGlobalElements,
    makeLocalGraph,
    renderCard,
    typeLabel,
  };
}));
