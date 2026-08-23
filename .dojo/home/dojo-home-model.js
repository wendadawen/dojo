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

  function shortLabel(page) {
    const title = String(page.title || "");
    const colonIndex = title.search(/[：:]/);
    const lead = colonIndex > 0 ? title.slice(0, colonIndex) : title;
    if (lead.length <= 24) return lead;
    const parentheses = lead.match(/^(.+?)[（(]([^（）()]+)[）)]$/);
    if (parentheses) {
      const prefix = parentheses[1].trim();
      const inner = parentheses[2].trim();
      if (prefix.length <= 16) return prefix;
      if (/^[A-Za-z][A-Za-z0-9-]{1,12}$/.test(inner)) return inner;
    }
    return `${lead.slice(0, 22).trim()}…`;
  }

  function compareByTitle(a, b) {
    return a.title.localeCompare(b.title, "zh-CN") || a.id.localeCompare(b.id);
  }

  function sortPages(pages, sort) {
    const ordered = pages.slice();
    if (sort === "oldest") {
      ordered.sort(
        (a, b) =>
          (a.date || "9999-99-99").localeCompare(b.date || "9999-99-99") ||
          compareByTitle(a, b),
      );
    } else if (sort === "title") {
      ordered.sort(compareByTitle);
    } else {
      ordered.sort(
        (a, b) =>
          (b.date || "").localeCompare(a.date || "") || compareByTitle(a, b),
      );
    }
    return ordered;
  }

  function filterPages(pages, state) {
    const query = normalize(state.query);
    const matched = pages
      .filter((page) => !state.type || page.type === state.type)
      .filter((page) => !state.topic || (page.topics || []).includes(state.topic))
      .filter((page) => {
        if (!query) return true;
        return normalize([
          page.title,
          page.description,
          page.summary,
          page.tag,
          ...(page.topics || []),
        ].join(" ")).includes(query);
      });
    return sortPages(matched, state.sort || "newest");
  }

  function getFilterOptions(pages) {
    return {
      types: [...new Set(pages.map((page) => page.type).filter(Boolean))].sort(),
      topics: [...new Set(pages.flatMap((page) => page.topics || []))]
        .sort((a, b) => a.localeCompare(b, "zh-CN")),
    };
  }

  function nodeFor(page, role = "normal", match = true) {
    return {
      data: {
        id: page.id,
        label: page.title,
        shortLabel: shortLabel(page),
        path: page.path,
        type: page.type,
        topics: page.topics || [],
        role,
        match,
        degree: (page.incoming_count || 0) + (page.outgoing_count || 0),
      },
    };
  }

  function edgeFor(edge, matchIds) {
    const match = !matchIds || matchIds.has(edge.source) || matchIds.has(edge.target);
    return {
      data: {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        count: edge.count,
        match,
      },
      classes: match ? undefined : "is-adjacent",
    };
  }

  function makeGlobalElements(catalog, visibleIds, matchIds) {
    const allowed = visibleIds || new Set(catalog.pages.map((page) => page.id));
    return {
      nodes: catalog.pages
        .filter((page) => allowed.has(page.id))
        .map((page) => {
          const match = !matchIds || matchIds.has(page.id);
          const node = nodeFor(page, "normal", match);
          if (!match) node.classes = "is-adjacent";
          return node;
        }),
      edges: catalog.edges
        .filter((edge) => allowed.has(edge.source) && allowed.has(edge.target))
        .map((edge) => edgeFor(edge, matchIds)),
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
    const topic = (page.topics || [])[0] || page.tag || "";
    const summary = page.summary || page.description || "暂无摘要";
    return `
      <article class="document-card">
        <button class="card-main" type="button" data-open-path="${escapeHtml(page.path)}">
          <span class="card-meta">
            <span class="type-badge type-${escapeHtml(page.type)}">${escapeHtml(typeLabel(page.type))}</span>
            ${topic ? `<span class="topic-chip">${escapeHtml(topic)}</span>` : ""}
            ${page.date ? `<time class="date-chip" datetime="${escapeHtml(page.date)}">${escapeHtml(page.date)}</time>` : ""}
          </span>
          <h2>${escapeHtml(page.title)}</h2>
          <p>${escapeHtml(summary)}</p>
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
