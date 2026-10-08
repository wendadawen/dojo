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
      topics: [...new Set(pages.flatMap((page) => page.topics || []))]
        .sort((a, b) => a.localeCompare(b, "zh-CN")),
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
    const topic = (page.topics || [])[0] || page.tag || "";
    return `
      <article class="document-card">
        <button class="card-main" type="button" data-open-path="${escapeHtml(page.path)}">
          <span class="card-meta">
            ${topic ? `<span class="topic-chip">${escapeHtml(topic)}</span>` : ""}
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
