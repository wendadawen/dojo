window.DojoHomeModel = (function () {
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

  function filterPages(pages, state) {
    const query = normalize(state.query);
    const matched = pages
      .filter((page) => !state.topic || page.tag === state.topic)
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
    return matched.slice().sort(
      (a, b) =>
        (b.date || "").localeCompare(a.date || "") || compareByTitle(a, b),
    );
  }

  // 标签是单一字段，按页数从多到少排，同数按拼音——比主题更均匀，也更好扫。
  function getFilterOptions(pages) {
    const counts = new Map();
    for (const page of pages) {
      const tag = page.tag;
      if (tag) counts.set(tag, (counts.get(tag) || 0) + 1);
    }
    return {
      tags: [...counts.entries()]
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], "zh-CN"))
        .map(([name, count]) => ({ name, count })),
    };
  }

  function makeGraphData(catalog, visibleIds, matchIds) {
    const allowed = visibleIds || new Set(catalog.pages.map((page) => page.id));
    return {
      nodes: catalog.pages
        .filter((page) => allowed.has(page.id))
        .map((page) => ({
          id: page.id,
          shortLabel: shortLabel(page),
          degree: (page.incoming_count || 0) + (page.outgoing_count || 0),
          match: !matchIds || matchIds.has(page.id),
        })),
      links: catalog.edges
        .filter((edge) => allowed.has(edge.source) && allowed.has(edge.target))
        .map((edge) => ({
          source: edge.source,
          target: edge.target,
          match: !matchIds || matchIds.has(edge.source) || matchIds.has(edge.target),
        })),
    };
  }

  function summaryOf(page) {
    return page.summary || page.description || "暂无摘要";
  }

  function renderCard(page) {
    // 与标签行同源：都取 tag，避免卡片写「注意力机制」、筛选写「注意力」两套说法。
    const topic = page.tag || "";
    return `
      <article class="document-card">
        <button class="card-main" type="button" data-open-path="${escapeHtml(page.path)}">
          <span class="card-meta">
            ${topic ? `<span class="card-tag">${escapeHtml(topic)}</span>` : ""}
            ${page.date ? `<time class="date-chip" datetime="${escapeHtml(page.date)}">${escapeHtml(page.date)}</time>` : ""}
          </span>
          <h2>${escapeHtml(page.title)}</h2>
          <p>${escapeHtml(summaryOf(page))}</p>
        </button>
      </article>`;
  }

  return {
    escapeHtml,
    filterPages,
    getFilterOptions,
    makeGraphData,
    renderCard,
    summaryOf,
  };
}());
