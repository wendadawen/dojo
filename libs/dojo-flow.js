/* Dojo flow：全屏可交互数据流画布引擎。

   为什么全部放在这里：之前渲染在 Python、交互在 JS，两套坐标与走线规则
   各写一遍，必然对不上（静态路径与拖动后的位置不一致、箭头被框裁掉）。
   现在**单一真相源**——节点坐标是唯一数据，连线在每次变化后由同一段
   代码重新计算、重新绘制。

   用法：
     DojoFlow.mount(document.getElementById('app'), DATA);
   DATA = { views: [{ id, label, nodes: [...], edges: [...], groups: [...] }] }
*/

(function (root) {
  'use strict';

  // ---------- 版面参数 ----------
  var CLEARANCE = 12;    // 连线端点离框的距离（箭头需要这段空间才可见）
  // 绕行走廊离障碍框的横向距离。不能直接用 CLEARANCE：12px 只够箭头，
  // 线会紧贴节点边缘一路擦过去（attncore 里 vr→av 就贴着右侧列走了 2000px）。
  // 走廊要明显宽于箭头，视觉上才像「绕开」而不是「贴着」。
  var CORRIDOR_GAP = 30;
  var MIN_STRAIGHT = 26; // 转弯前/后的最短直线段
  var ROUND = 16;        // 转角半径
  var GRID = 26;         // 网格尺寸
  // 手动缩放下限要低于自动适配的下限，否则小屏适配到 0.12 之后，
  // 随便滚一下滚轮就会被夹回 0.2，出现跳变。
  var MIN_SCALE = 0.06;
  var MAX_SCALE = 2.5;
  // 自动适配允许比手动缩放下限更小：手机上 20 个节点的视图缩到 0.2 仍放不下，
  // 卡在 MIN_SCALE 会让节点跑到视口外。手动缩放仍用 MIN_SCALE。
  var FIT_MIN_SCALE = 0.06;
  var SIDE_PAD = 48;
  // 浮层（标签栏 / 缩放按钮）会随视口换行改变高度，适配时按实测尺寸留白，
  // 不能用固定值：小屏上标签栏换行后高 129px，固定 130px 仍会让末行被压住。
  function overlayPad(viewport) {
    var vpr = viewport.getBoundingClientRect();
    var top = 74, bottom = 48, left = SIDE_PAD, right = SIDE_PAD;
    ['.flow-bar', '.flow-zoom', '.flow-cfg-btn'].forEach(function (sel) {
      var e = document.querySelector(sel);
      if (!e) return;
      if (e.hidden || e.offsetParent === null) return;
      var r = e.getBoundingClientRect();
      if (r.top - vpr.top < vpr.height / 2) {          // 在屏幕上半部 → 占顶部
        top = Math.max(top, r.bottom - vpr.top + 10);
      } else {                                          // 下半部 → 占底部
        bottom = Math.max(bottom, vpr.bottom - r.top + 10);
      }
    });
    return { top: top, bottom: bottom, left: left, right: right };
  }

  var SVG_NS = 'http://www.w3.org/2000/svg';

  function el(tag, attrs) {
    var node = document.createElementNS(SVG_NS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    return node;
  }

  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ---------- 走线 ----------

  /** 圆角折线：把直角用二次贝塞尔磨圆。 */
  function roundedPath(points) {
    if (points.length < 2) return '';
    if (points.length === 2) {
      return 'M' + points[0][0] + ',' + points[0][1] +
             ' L' + points[1][0] + ',' + points[1][1];
    }
    var d = 'M' + points[0][0] + ',' + points[0][1];
    for (var i = 1; i < points.length - 1; i++) {
      var p = points[i - 1], c = points[i], n = points[i + 1];
      var lin = Math.abs(c[0] - p[0]) + Math.abs(c[1] - p[1]);
      var lout = Math.abs(n[0] - c[0]) + Math.abs(n[1] - c[1]);
      var r = Math.max(0, Math.min(ROUND, (lin - 6) / 2, (lout - 6) / 2));
      if (r < 0.5) { d += ' L' + c[0] + ',' + c[1]; continue; }
      var ipx = c[0] + (p[0] - c[0]) / (lin || 1) * r;
      var ipy = c[1] + (p[1] - c[1]) / (lin || 1) * r;
      var opx = c[0] + (n[0] - c[0]) / (lout || 1) * r;
      var opy = c[1] + (n[1] - c[1]) / (lout || 1) * r;
      d += ' L' + ipx + ',' + ipy + ' Q' + c[0] + ',' + c[1] + ' ' + opx + ',' + opy;
    }
    var last = points[points.length - 1];
    d += ' L' + last[0] + ',' + last[1];
    return d;
  }

  /** 线段与矩形是否相交（用于避障）。 */
  function segHitsRect(x1, y1, x2, y2, rx, ry, rw, rh) {
    if (Math.abs(y1 - y2) < 0.5) {           // 水平段
      var lo = Math.min(x1, x2), hi = Math.max(x1, x2);
      return y1 > ry && y1 < ry + rh && hi > rx && lo < rx + rw;
    }
    if (Math.abs(x1 - x2) < 0.5) {           // 竖直段
      var lo2 = Math.min(y1, y2), hi2 = Math.max(y1, y2);
      return x1 > rx && x1 < rx + rw && hi2 > ry && lo2 < ry + rh;
    }
    return false;
  }

  /**
   * 计算 a → b 的走线。
   *
   * 数据自上而下：默认从 a 的下沿进入 b 的上沿。依次尝试三种策略：
   *
   *   1. 竖直直连 —— 仅当路径上没有其它节点；
   *   2. 侧向绕行 —— 在障碍框的左右外侧找一条真正空的竖直走廊，
   *      路径为「出框 → 横穿到走廊 → 竖直 → 横向进入目标」；
   *   3. 兜底 —— 取两者中点直连。
   *
   * 走廊必须整体无遮挡：竖线要空，两侧的横穿段也要空。之前只检查竖段，
   * 结果出现「竖线贴着相邻框」和「横线插进别的框」。
   */
  /**
   * 计算 a → b 的走线。
   *
   * used 是本次渲染中已被其它边占用的走廊 x 坐标集合：两条边若走同一条
   * 竖直走廊，视觉上会并成一条粗线（实测 sink→catsink 与 vr→av 都走
   * x=1895，看起来像重叠）。这里让后一条边避开已占用的走廊。
   */
  function routeEdge(a, b, obstacles, used) {
    var acx = a.x + a.w / 2, bcx = b.x + b.w / 2;
    var ay = a.y + a.h, by = b.y - CLEARANCE;
    used = used || {};
    // 两条边走廊之间的最小间距。
    // 之前取 14px——参数上「不重合」，但两条 2px 的线只隔 14px，视觉上仍像
    // 一条粗线（实测 linear_gate 右侧就是这样）。按「线宽 + 明显留白」定，
    // 取 22px 让并行段一眼能看出是两条。
    var CORRIDOR_MIN_GAP = 22;
    function freeCorridor(x) {
      var keys = Object.keys(used);
      for (var i = 0; i < keys.length; i++) {
        if (Math.abs(parseFloat(keys[i]) - x) < CORRIDOR_MIN_GAP) return false;
      }
      return true;
    }

    function hits(x1, y1, x2, y2) {
      for (var i = 0; i < obstacles.length; i++) {
        var ob = obstacles[i];
        if (segHitsRect(x1, y1, x2, y2, ob.x, ob.y, ob.w, ob.h)) return true;
      }
      return false;
    }
    // 竖线覆盖「两框之间的整段」，方向按 y 大小自动取
    var ySpan0 = Math.min(a.y + a.h, b.y - CLEARANCE);
    var ySpan1 = Math.max(a.y + a.h, b.y - CLEARANCE);
    if (b.y < a.y) {                    // 目标在上方
      ySpan0 = Math.min(a.y - CLEARANCE, b.y + b.h);
      ySpan1 = Math.max(a.y - CLEARANCE, b.y + b.h);
    }
    function verticalClear(x) { return !hits(x, ySpan0, x, ySpan1); }

    // 后置校验：候选路径逐点采样，落进任何障碍框就否决。
    // 只靠 verticalClear + 两段水平 hits 不够——圆角转弯处会甩进框里，
    // 实测偏移走廊后 attn 视图新增 2 处穿框。
    function pathClear(pts) {
      var INSET = 2;   // 贴着框边不算穿
      for (var i = 0; i < pts.length - 1; i++) {
        var p0 = pts[i], p1 = pts[i + 1];
        var len = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
        var steps = Math.max(2, Math.ceil(len / 6));
        for (var t = 0; t <= steps; t++) {
          var px = p0[0] + (p1[0] - p0[0]) * t / steps;
          var py = p0[1] + (p1[1] - p0[1]) * t / steps;
          for (var oi = 0; oi < obstacles.length; oi++) {
            var ob = obstacles[oi];
            if (px > ob.x + INSET && px < ob.x + ob.w - INSET &&
                py > ob.y + INSET && py < ob.y + ob.h - INSET) return false;
          }
        }
      }
      return true;
    }

    var yTop = a.y + a.h / 2, yBot = b.y + b.h / 2;

    // 超长边强制走最外侧。
    //
    // gate_states→gated 这类边要跨越大半个视图（实测竖段 2100px）。如果让它
    // 占用中间的空隙通道，就会和所有短边挤在同一条竖线上（它一个人重合了 6 对
    // 边）。长边走最外侧，既自己干净、也不挡短边。
    var spanY = Math.abs((b.y + b.h / 2) - (a.y + a.h / 2));
    if (spanY > 900) {
      var allL = Math.min.apply(null, [a.x, b.x].concat(
        obstacles.map(function (o) { return o.x; })));
      var allR = Math.max.apply(null, [a.x + a.w, b.x + b.w].concat(
        obstacles.map(function (o) { return o.x + o.w; })));
      var yA0 = (b.y >= a.y) ? ay : a.y - CLEARANCE;
      var yB0x = (b.y >= a.y) ? by : b.y + b.h + CLEARANCE;
      var outerSides = (a.x <= b.x) ? [allL - CORRIDOR_GAP, allR + CORRIDOR_GAP]
                                    : [allR + CORRIDOR_GAP, allL - CORRIDOR_GAP];
      for (var oi2 = 0; oi2 < outerSides.length; oi2++) {
        var ox = outerSides[oi2];
        var oxa = (a.x <= ox && ox <= a.x + a.w) ? acx : (ox < a.x ? a.x : a.x + a.w);
        var oxb = (b.x <= ox && ox <= b.x + b.w) ? bcx : (ox < b.x ? b.x : b.x + b.w);
        var opts = [[oxa, yA0], [ox, yA0], [ox, yB0x], [oxb, yB0x]];
        if (pathClear(opts)) { used[ox] = true; return roundedPath(opts); }
      }
    }


    // 策略 1：竖直直连。
    //
    // 必须在这里也检查走廊是否已被占用：直连的 x 总是落在端点框的中线/边缘上，
    // 多条边会取到同一个 x 而并排重叠——实测 attn 视图里 x≈2210 挤了 6 条边、
    // x≈2230 挤了 4 条。之前这里只看「竖线是否畅通」，完全绕过了去重。
    var verticals = [];
    if (a.x <= bcx && bcx <= a.x + a.w) verticals.push(bcx);
    if (b.x <= acx && acx <= b.x + b.w) verticals.push(acx);
    if (Math.abs(acx - bcx) < 1) verticals.push(acx);
    var vCand = [];
    verticals.forEach(function (vx) {
      if (!verticalClear(vx)) return;
      var pts = [[vx, ay], [vx, by]];
      if (!pathClear(pts)) return;
      vCand.push(vx);
    });
    // 优先用没被占用的；被占用时就近微调，不再原地复用同一条竖线。
    // 之前「全被占 → 仍取原 x」会让十几条边叠在同一条线上（实测 x=1954
    // 挤了 18 条），这正是「线条重复」的来源。
    for (var vk = 0; vk < vCand.length; vk++) {
      if (freeCorridor(vCand[vk])) {
        used[vCand[vk]] = true;
        return roundedPath([[vCand[vk], ay], [vCand[vk], by]]);
      }
      for (var vd = 1; vd <= 6; vd++) {
        var voff = vd * CORRIDOR_MIN_GAP;
        var vs = [vCand[vk] - voff, vCand[vk] + voff];
        for (var vsi = 0; vsi < 2; vsi++) {
          var vv = vs[vsi];
          if (!freeCorridor(vv) || !verticalClear(vv)) continue;
          if (!pathClear([[vv, ay], [vv, by]])) continue;
          used[vv] = true;
          return roundedPath([[vv, ay], [vv, by]]);
        }
      }
    }

    // 策略 1.5：直连被「夹在中间的框」挡住时，绕到那个框的旁边。
    //
    // 典型情况：同一列上下两个节点（q_nope → qcat），中间还夹着第三个
    // （k_pass_ln）。此时任何竖直直连都会穿过中间那个框，必须在它左右让开。
    // 先把「夹在 a、b 之间且挡住中线的框」找出来，再贴着它的外侧走。
    var blockers = [];
    obstacles.forEach(function (ob) {
      var overlapsX = ob.x < acx && acx < ob.x + ob.w;
      var betweenY = ob.y + ob.h > ay && ob.y < by;
      if (overlapsX && betweenY) blockers.push(ob);
    });
    if (blockers.length) {
      // 取这些框的整体左右边界，从两侧分别找一条贯通走廊。
      // 这里不强求「走廊没被别的线占用」——绕开夹在中间的框比避免与别的线
      // 并行更重要（线交叉可以接受，穿框不行）。只要求竖线本身不与任何框相交。
      var bl = Math.min.apply(null, blockers.map(function (o) { return o.x; }));
      var br = Math.max.apply(null, blockers.map(function (o) { return o.x + o.w; }));
      // 用真实跨度（ay/by），不要用两框中心：中心会漏掉中间的框
      var yA15 = (b.y >= a.y) ? ay : a.y - CLEARANCE;
      var yB15 = (b.y >= a.y) ? by : b.y + b.h + CLEARANCE;
      var sides2 = [bl - CORRIDOR_GAP, br + CORRIDOR_GAP];
      for (var s2 = 0; s2 < sides2.length; s2++) {
        var sx = sides2[s2];
        var xa3 = (a.x <= sx && sx <= a.x + a.w) ? acx : (sx < a.x ? a.x : a.x + a.w);
        var xb3 = (b.x <= sx && sx <= b.x + b.w) ? bcx : (sx < b.x ? b.x : b.x + b.w);
        var p3 = [[xa3, yA15], [sx, yA15], [sx, yB15], [xb3, yB15]];
        if (pathClear(p3)) { used[sx] = true; return roundedPath(p3); }
      }
    }

    // 策略 2：在障碍框左右外侧找空走廊。
    //
    // 候选位置取「离所有障碍框尽量远」的点，而不是「离某个框固定偏移」：
    // 两列之间只有 40px 空隙时，固定偏移 30px 会让线离另一列只剩 10px，
    // 看起来像贴着节点走。取中点则自动落在空隙正中。
    var candidates = [];
    obstacles.forEach(function (ob) {
      candidates.push(ob.x - CORRIDOR_GAP, ob.x + ob.w + CORRIDOR_GAP);
    });
    // 相邻两个障碍框之间的空隙：按比例切成多条候选走廊。
    // 只放「中点」一条时走廊数量不够——实测 attn 视图有 25 条边需要竖向绕行，
    // 而可用候选只有 18 个，怎么分配都会撞。
    var sorted = obstacles.slice().sort(function (p, q) { return p.x - q.x; });
    for (var si = 0; si < sorted.length - 1; si++) {
      var curR = sorted[si].x + sorted[si].w, nxtL = sorted[si + 1].x;
      var gap = nxtL - curR;
      if (gap <= 0) continue;
      var lanes = Math.max(1, Math.floor(gap / CORRIDOR_MIN_GAP));
      for (var li = 1; li <= lanes; li++) {
        candidates.push(curR + gap * li / (lanes + 1));
      }
    }
    candidates.push(Math.min(a.x, b.x) - CORRIDOR_GAP);
    candidates.push(Math.max(a.x + a.w, b.x + b.w) + CORRIDOR_GAP);
    candidates = candidates.filter(function (v, i, arr) { return arr.indexOf(v) === i; });


    function clearance(x) {
      var best = Infinity;
      obstacles.forEach(function (ob) {
        if (x > ob.x && x < ob.x + ob.w) { best = 0; return; }
        best = Math.min(best, x <= ob.x ? ob.x - x : x - (ob.x + ob.w));
      });
      return best;
    }
    // 排序规则：谁绕得短谁优先。
    //
    // 试过两种按「余量」排序的写法，都会把线推到画布最外缘：
    //   · 「离障碍越远越好」→ gate→gate_states 直线 356px 却走了 3013px；
    //   · 「余量 ≥ CORRIDOR_GAP 的优先」→ 绕行 476px 的走廊因贴着某个框
    //     （余量 0）被排在绕行 1604px 的后面，仍然绕远。
    // 其实「这条走廊通不通」已经由 verticalClear + pathClear 判定了，排序
    // 只需要挑最短的那条即可。
    candidates.sort(function (p, q) {
      var pc = Math.abs(p - acx) + Math.abs(p - bcx);
      var qc = Math.abs(q - acx) + Math.abs(q - bcx);
      return pc - qc;
    });

    function tryCorridor(x) {
      // 用「源框下沿 → 目标框上沿」这段真实跨度（ay/by），不要用两框中心
      // （yTop/yBot）：中心点会漏掉夹在中间的框，导致线穿框。
      if (!verticalClear(x)) return null;
      var xa = (a.x <= x && x <= a.x + a.w) ? acx : (x < a.x ? a.x : a.x + a.w);
      var xb = (b.x <= x && x <= b.x + b.w) ? bcx : (x < b.x ? b.x : b.x + b.w);
      // 目标在上方时，起点应从 a 的上沿出、终点从 b 的下沿入
      var yA = (b.y >= a.y) ? ay : a.y - CLEARANCE;
      var yB0 = (b.y >= a.y) ? by : b.y + b.h + CLEARANCE;
      var yB = entryY(yB0);
      if (hits(xa, yA, x, yA)) return null;
      if (hits(x, yB, xb, yB)) return null;
      if (!pathClear([[xa, yA], [x, yA], [x, yB], [xb, yB]])) return null;
      used[x] = true;
      return roundedPath([[xa, yA], [x, yA], [x, yB], [xb, yB]]);
    }

    // 从同一个框出发的多条边，起点侧也要错开。
    //
    // 实测 x→kva 与 x→indexer 都从 x 框的左下角出发、走同一条竖线（x=1680），
    // 前 267 个像素完全重合。按「该源节点的第几条出边」把走廊错开。
    if (!used.__src) used.__src = {};
    var srcKey = a.id || (Math.round(a.x) + '_' + Math.round(a.y));
    var srcN = (used.__src[srcKey] = (used.__src[srcKey] || 0) + 1);
    if (srcN > 1) {
      // 让后续边避开同一侧，优先换到另一侧的外缘
      var altSides = (a.x <= b.x)
        ? [a.x + a.w + CORRIDOR_GAP, a.x - CORRIDOR_GAP]
        : [a.x - CORRIDOR_GAP, a.x + a.w + CORRIDOR_GAP];
      var yA_s = (b.y >= a.y) ? ay : a.y - CLEARANCE;
      var yB_s = (b.y >= a.y) ? by : b.y + b.h + CLEARANCE;
      for (var ai2 = 0; ai2 < altSides.length; ai2++) {
        var ax2 = altSides[ai2];
        var axa = (a.x <= ax2 && ax2 <= a.x + a.w) ? acx : (ax2 < a.x ? a.x : a.x + a.w);
        var axb = (b.x <= ax2 && ax2 <= b.x + b.w) ? bcx : (ax2 < b.x ? b.x : b.x + b.w);
        var apts = [[axa, yA_s], [ax2, yA_s], [ax2, yB_s], [axb, yB_s]];
        if (!freeCorridor(ax2)) continue;
        if (pathClear(apts)) { used[ax2] = true; return roundedPath(apts); }
      }
    }

    // 进入同一目标节点的多条边，最后那一段水平线不能共用同一个 y。
    //
    // 实测 value_states→attncore、masked→attncore、sink→attncore 三条边
    // 都汇聚到 y=2590 再横向进 attncore，那一段完全重合（最多 614 个像素
    // 重合），看起来就是一条线。这里按「目标节点的第几条入边」把入口错开。
    if (!used.__entry) used.__entry = {};
    var entryKey = b.id || (Math.round(b.x) + '_' + Math.round(b.y));
    var entryN = (used.__entry[entryKey] = (used.__entry[entryKey] || 0) + 1);
    var entryOff = (entryN - 1) * 12;         // 每条边错开 12px
    var maxOff = Math.max(0, b.h - 16);
    if (entryOff > maxOff) entryOff = (entryN % 3) * 12 % Math.max(1, maxOff);
    function entryY(base) {
      // 从目标框上沿进入时向下错开；从下沿进入时向上错开
      return base + (b.y >= a.y ? entryOff : -entryOff);
    }
    // 走廊若已被别的边占用，就在它附近找一条同样畅通、但错开的竖线。
    // 不能「先跳过、实在找不到再退回原处」——实测某些视图只有一条畅通走廊，
    // 退回后两条边又重合（sink→catsink 与 vr→av 都走 x=1895）。
    for (var ci = 0; ci < candidates.length; ci++) {
      var x0 = candidates[ci];
      if (!verticalClear(x0)) continue;
      if (freeCorridor(x0)) {
        var r0 = tryCorridor(x0);
        if (r0) return r0;
      }
      for (var k = 1; k <= 4; k++) {
        var off = k * CORRIDOR_MIN_GAP;
        var sides = [x0 + off, x0 - off];
        for (var s = 0; s < 2; s++) {
          var xv = sides[s];
          if (!freeCorridor(xv) || !verticalClear(xv)) continue;
          var r1 = tryCorridor(xv);
          if (r1) return r1;
        }
      }
    }

    // 策略 2.5：走廊都被占用时，改成「横穿过去」而不是「贴着别的线并排走」。
    //
    // 正交走线里，两条线十字交叉一眼就能看出是两条；并排贴在 2px 内则会被
    // 当成一条线（实测 qsplit→q_nope 与 rope_in→rope 相差 2px、重叠 318px）。
    // 所以宁可交叉也不要并排：这里不再放宽间距去复用邻近走廊，而是直接走一条
    // 横穿的 L 形路径（从起点向下 → 横穿 → 到目标），它必然与其他竖线交叉，
    // 但不会与它们平行重叠。
    if (acx !== bcx) {
      // 先向下走到两层之间，再横穿，最后进入目标
      var midY = Math.max(ay, by) + (yBot - yTop) / 2;
      var through = [[acx, ay], [acx, midY], [bcx, midY], [bcx, by]];
      if (pathClear(through)) return roundedPath(through);
    }

    // 策略 3：兜底。不直接取中点连过去——那样会穿过中间的节点（实测 attn
    // 视图有 2 处）。改为退到「所有障碍框之外」的一侧绕行。
    var farLeft = Math.min.apply(null, [a.x, b.x].concat(
      obstacles.map(function (o) { return o.x; }))) - CORRIDOR_GAP;
    var farRight = Math.max.apply(null, [a.x + a.w, b.x + b.w].concat(
      obstacles.map(function (o) { return o.x + o.w; }))) + CORRIDOR_GAP;
    var fallback = [farLeft, farRight];
    for (var fi = 0; fi < fallback.length; fi++) {
      var fx = fallback[fi];
      var xa2 = (a.x <= fx && fx <= a.x + a.w) ? acx : (fx < a.x ? a.x : a.x + a.w);
      var xb2 = (b.x <= fx && fx <= b.x + b.w) ? bcx : (fx < b.x ? b.x : b.x + b.w);
      var fpts = [[xa2, yTop], [fx, yTop], [fx, yBot], [xb2, yBot]];
      if (pathClear(fpts)) {
        used[fx] = true;
        return roundedPath(fpts);
      }
    }
    // 实在无路可走才退回中点直连（宁可难看也不要丢边）
    var mid = (acx + bcx) / 2;
    return roundedPath([[acx, ay], [mid, ay], [mid, by], [bcx, by]]);
  }

  // ---------- 尺寸估算（SVG 无自动换行，框宽须由内容反推）----------
  // 每行 [文本, 类名, 行高, 单字宽]；类名决定渲染样式。
  function textWidth(text, perChar) {
    var w = 0;
    for (var i = 0; i < text.length; i++) {
      w += perChar * (text.charCodeAt(i) > 0x2E80 ? 2 : 1);
    }
    return w;
  }

  // 参考标准：节点里只有两行——算子名 + 形状。字号相对框要大，留白要小。
  // [行高, 单字宽]
  // [行高, 单字宽]。单字宽是估算用，之前按 9.6 / 8.6 / 7.8 设，实测比真实
  // 渲染宽约 1.24 倍（用 canvas.measureText 对同一批文本量过），会把框系统性
  // 撑宽、每个视图都顶到宽度上限。这里按实测比值校准回来。
  var LINE_SPEC = {
    nm: [24, 7.7],   // 算子名（600 17px）
    sh: [22, 6.9],   // 形状 / 维度（15px monospace）
    dt: [20, 6.3]    // 依据：权重 / 开关 / 构成统计（12.5px）
  };
  var PAD_X = 14;    // 文本区左右内边距
  var PAD_Y = 12;    // 文本区上下内边距

  /**
   * 按字符宽度估算折行。
   * 批注不做折行时会变成一条很长的横线，把整张图的包围盒拉宽，自动适配的
   * 缩放比随之变小，图就被挤得很小。这里限制单行最大宽度后再折行。
   */
  function wrapText(text, perChar, maxWidth) {
    var out = [];
    String(text).split('\n').forEach(function (paragraph) {
      if (!paragraph) { out.push(''); return; }
      // 按「词」折行：逐字折会把英文标识符从中间切断（实测 MoE 被断成 M / oE）。
      // 中文没有空格，整段会被当成一个词，所以再对超长的词做逐字回退。
      var tokens = paragraph.match(/[A-Za-z0-9_.:/+\-()\[\]]+|[\s]+|[\s\S]/g) || [];
      var line = '';
      tokens.forEach(function (tk) {
        if (!line && /^\s+$/.test(tk)) return;      // 行首不留空格
        if (textWidth(line + tk, perChar) > maxWidth && line) {
          out.push(line.replace(/\s+$/, ''));
          line = /^\s+$/.test(tk) ? '' : tk;
        } else {
          line += tk;
        }
        // 单个词本身就超宽时按字符切
        while (textWidth(line, perChar) > maxWidth) {
          var cut = line.length - 1;
          while (cut > 1 && textWidth(line.slice(0, cut), perChar) > maxWidth) cut--;
          out.push(line.slice(0, cut));
          line = line.slice(cut);
        }
      });
      if (line) out.push(line);
    });
    return out;
  }

  function nodeLines(node) {
    var lines = [[node.name || '', 'nm']];
    if (node.shape) {
      String(node.shape).split('\n').forEach(function (s) { lines.push([s, 'sh']); });
    }
    // 参考标准的节点最多三行：算子名 / 形状 / 依据。
    // 第三行放「用哪个权重」「配置开关→行为」「分层构成统计」这类可回查的硬信息，
    // 它是参考图里信息密度最高的部分，此前被我挪到节点外，导致关键内容读不到。
    if (node.detail) {
      String(node.detail).split('\n').forEach(function (s) { lines.push([s, 'dt']); });
    }
    return lines;
  }

  function measureNode(node) {
    var lines = nodeLines(node);
    var widest = 0, tall = 0;
    lines.forEach(function (l) {
      var spec = LINE_SPEC[l[1]] || LINE_SPEC.sh;
      widest = Math.max(widest, textWidth(l[0], spec[1]));
      tall += spec[0];
    });
    // 手动指定尺寸的优先（作者可按参考图精确定位）
    var w = node.w || Math.ceil(widest + PAD_X * 2);
    var h = node.h || Math.max(56, tall + PAD_Y * 2);
    return { w: w, h: h, lines: lines.length };
  }

  /**
   * 视图内统一的节点宽度。
   *
   * 参考标准（实测）：一个视图里所有框等宽，只有高度随内容变——主干图每个框
   * 都是 440px，MLA 细节图 321px，层视图 596px。此前是每个节点按自己最长一行
   * 独立测宽，同一视图内从 150 到 707px 都有，横向参差不齐。
   *
   * 取该视图里最宽节点的自然宽度作为统一宽度，并夹在 NODE_W_MIN~NODE_W_MAX 之间；
   * 超长的行不撑宽框，改由 wrapLines 折行。
   */
  var NODE_W_MIN = 240;
  var NODE_W_MAX = 560;

  function uniformWidth(nodes) {
    var widest = 0;
    nodes.forEach(function (node) {
      var m = measureNode(node);
      widest = Math.max(widest, m.w);
    });
    return Math.max(NODE_W_MIN, Math.min(NODE_W_MAX, widest));
  }

  /**
   * 把节点内容按统一宽度折行，返回可直接渲染的行列表。
   *
   * measure 与 render 必须共用这一个函数，否则「测量时说两行、渲染时出来三行」，
   * 文字就会溢出框外。返回 [{text, cls, lh}]，lh 是该行的行高。
   */
  function wrapLines(node, width) {
    var maxText = Math.max(40, width - PAD_X * 2);
    var out = [];
    nodeLines(node).forEach(function (l) {
      var text = l[0], cls = l[1];
      var spec = LINE_SPEC[cls] || LINE_SPEC.sh;
      wrapText(text, spec[1], maxText).forEach(function (piece) {
        out.push({ text: piece, cls: cls, lh: spec[0] });
      });
    });
    return out;
  }

  /** 给定统一宽度，算出某个节点的实际尺寸（高度由折行后的行数决定）。 */
  function sizeFor(node, width) {
    var lines = wrapLines(node, width);
    var tall = 0;
    lines.forEach(function (l) { tall += l.lh; });
    return {
      w: node.w || width,
      h: node.h || Math.max(56, tall + PAD_Y * 2),
      lines: lines
    };
  }

  // ---------- 视图渲染 ----------

  function FlowView(app, data) {
    this.app = app;
    this.data = data;
    this.view = data.views[0];
    this.scale = 1;
    this.tx = 0;
    this.ty = 0;
    this.positions = {};   // id -> {x, y, w, h}
    this.nodeEls = {};
    this.build();
  }

  FlowView.prototype.build = function () {
    var self = this;

    // 从视口移出 600px 之外的辅助画布：SVG 尺寸只用来定义坐标系，
    // 节点拖出去后由 overflow:visible 保证仍然可见，因此没有边界限制。
    var viewport = document.createElement('div');
    viewport.className = 'flow-viewport';

    var svg = el('svg', { class: 'flow-svg' });
    var defs = el('defs');
    var marker = el('marker', {
      id: 'flow-arrow', viewBox: '0 0 10 10', refX: '10', refY: '5',
      markerWidth: '9', markerHeight: '9', orient: 'auto-start-reverse'
    });
    marker.appendChild(el('path', { d: 'M0,0 L10,5 L0,10 z', class: 'flow-arrowhead' }));
    defs.appendChild(marker);
    // 起点圆点：只有终点箭头时，读者无法判断一根线是「进入」还是「离开」某个框。
    // 起点画实心小圆 + 终点画箭头，方向就明确了（圆点=出发，箭头=到达，
    // 都没有=只是路过）。
    var startMarker = el('marker', {
      id: 'flow-dot', viewBox: '0 0 10 10', refX: '5', refY: '5',
      markerWidth: '7', markerHeight: '7', orient: 'auto-start-reverse'
    });
    startMarker.appendChild(el('circle', { cx: '5', cy: '5', r: '4', class: 'flow-dot-head' }));
    defs.appendChild(startMarker);
    svg.appendChild(defs);

    this.edgeLayer = el('g', { class: 'flow-edges' });
    this.nodeLayer = el('g', { class: 'flow-nodes' });
    this.scene = el('g', { class: 'flow-scene' });
    this.scene.appendChild(this.edgeLayer);
    this.scene.appendChild(this.nodeLayer);
    svg.appendChild(this.scene);
    viewport.appendChild(svg);
    this.viewport = viewport;
    this.svg = svg;

    // 把画布挂进页面。漏掉这一步时节点都建好了但不在文档里，
    // 页面看起来「什么都没有」。
    this.app.insertBefore(viewport, this.app.firstChild);

    // 顶栏与事件
    this.tabsEl = this.app.querySelector('.flow-bar');
    this.applyView(this.data.views[0].id);
    this.bindEvents();
    this.resize();
    window.addEventListener('resize', function () { self.resize(); });
  };

  FlowView.prototype.resize = function () {
    // viewBox 跟随视口；配合 CSS 的 overflow:visible，画布是无边界的
    var rect = this.viewport.getBoundingClientRect();
    this.svg.setAttribute('viewBox', '0 0 ' + rect.width + ' ' + rect.height);
    this.svg.setAttribute('width', rect.width);
    this.svg.setAttribute('height', rect.height);
  };

  FlowView.prototype.applyView = function (id) {
    var self = this;
    this.view = this.data.views.filter(function (v) { return v.id === id; })[0] || this.data.views[0];
    this.positions = {};
    this.edgeRoutes = {};
    this.edgeLabels = {};
    this.movedNodes = {};
    // 切视图时重新适配屏幕：一个视图几十个节点，按 100% 打开只看得到一角
    this.needsFit = true;
    // 默认按宽度适配（保持文字可读）；复位按钮用 'all' 做整图总览
    if (!this.fitMode) this.fitMode = 'width';

    // 两种布局：
    //   1. 节点自带 x/y → 用作者给的坐标（手工摆位，完全可控）；
    //   2. 没有坐标 → 交给 ELK 自动分层 + 边路由。
    // ELK 是异步的：算完会自己调用 render，这里不要抢先画。
    var hasManual = this.view.nodes.some(function (n) {
      return typeof n.x === 'number' && typeof n.y === 'number';
    });
    if (hasManual) {
      this.layoutManual();
    } else {
      this.edgeRoutes = {};
      this.layoutAuto();
    }
    if (this.tabsEl) {
      Array.prototype.forEach.call(this.tabsEl.children, function (btn) {
        btn.classList.toggle('on', btn.dataset.view === self.view.id);
      });
    }
    if (history.replaceState) history.replaceState(null, '', '#' + this.view.id);
    if (hasManual) this.render();
  };

  /**
   * 手工布局：节点自带 x/y，同列等宽。
   * 参考图的框在一列内宽度一致，按内容各自取宽会出现 53~238px 的混排。
   */
  FlowView.prototype.layoutManual = function () {
    var self = this;
    var w = uniformWidth(this.view.nodes);
    var sizes = {};
    this.view.nodes.forEach(function (n) { sizes[n.id] = sizeFor(n, w); });
    this.view.nodes.forEach(function (n) {
      var size = sizes[n.id];
      self.positions[n.id] = {
        id: n.id, x: n.x, y: n.y,
        w: size.w, h: size.h
      };
    });
  };

  /**
   * 自动布局：用 ELK 的 layered 算法。
   *
   * 为什么不用 dagre：dagre 只算节点位置，边要自己画，结果就是「多条边挤在
   * 同一水平线上糊成一片、横线穿过框」。ELK 同时负责**边路由**
   * （elk.edgeRouting = ORTHOGONAL），折点由它算好，页面只负责画。
   *
   * ELK 的 layout 是异步的，因此这里把结果缓存到 this.pendingLayout，
   * 算完后再触发一次 render。
   */
  FlowView.prototype.layoutAuto = function () {
    var self = this;
    var nodes = this.view.nodes, edges = this.view.edges;

    if (!root.ELK) {
      // 没加载 ELK 时退化为竖排，页面仍可读
      nodes.forEach(function (n, i) {
        var s = measureNode(n);
        self.positions[n.id] = { x: 60, y: 60 + i * 80, w: s.w, h: s.h };
      });
      return;
    }

    // 视图内等宽（参考标准）：所有框用同一宽度，高度按折行后的实际行数算
    var w = uniformWidth(nodes);
    var sizes = {};
    nodes.forEach(function (n) { sizes[n.id] = sizeFor(n, w); });

    var graph = {
      id: 'root',
      layoutOptions: {
        'elk.algorithm': 'layered',
        'elk.direction': 'DOWN',
        // 关键：边由 ELK 走正交线并避开节点
        'elk.edgeRouting': 'ORTHOGONAL',
        // 层间距要同时容纳箭头和标签，太窄会让箭头贴住下一个框
        'elk.layered.spacing.nodeNodeBetweenLayers': '88',
        'elk.spacing.nodeNode': '56',
        'elk.spacing.edgeNode': '28',
        'elk.spacing.edgeEdge': '14',
        'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
        'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
        // 边标签单独占位，避免压到线或框
        'elk.edgeLabels.placement': 'CENTER'
      },
      children: nodes.map(function (n) {
        return { id: n.id, width: sizes[n.id].w, height: sizes[n.id].h };
      }),
      edges: edges.map(function (e, i) {
        var edge = { id: 'e' + i, sources: [e.from], targets: [e.to] };
        if (e.label) {
          edge.labels = [{
            text: e.label,
            width: textWidth(e.label, 6.4) + 10,
            height: 16
          }];
        }
        return edge;
      })
    };

    this.layoutSeq = (this.layoutSeq || 0) + 1;
    var seq = this.layoutSeq;
    this.edgeRoutes = {};
    this.edgeLabels = {};

    new root.ELK().layout(graph).then(function (out) {
      if (seq !== self.layoutSeq) return;      // 已切到别的视图，丢弃这次结果
      out.children.forEach(function (c) {
        self.positions[c.id] = {
          id: c.id, x: c.x, y: c.y, w: c.width, h: c.height
        };
      });
      out.edges.forEach(function (e) {
        var idx = parseInt(String(e.id).replace('e', ''), 10);
        if (isNaN(idx)) return;
        var lab = e.labels && e.labels[0];
        if (lab && lab.x != null) {
          self.edgeLabels[idx] = {
            x: lab.x, y: lab.y, w: lab.width || 40, h: lab.height || 16
          };
        }
        var sec = e.sections && e.sections[0];
        if (!sec) return;
        var pts = [[sec.startPoint.x, sec.startPoint.y]];
        (sec.bendPoints || []).forEach(function (b) { pts.push([b.x, b.y]); });
        pts.push([sec.endPoint.x, sec.endPoint.y]);
        // ELK 给的是绝对坐标；节点已按左上角摆放，路径直接用绝对坐标
        self.edgeRoutes[idx] = pts;
      });
      // ELK 会把主干链在竖直方向左右推开（同层有分支时，主干被挤到一侧，
      // 实测 main 里主干链出现两种 x、layer 里跨 694px）。参考标准里主干
      // 始终是一条直线，所以这里做一次后处理把它拉直。
      self.render();
    });
  };

  FlowView.prototype.render = function () {
    var self = this;
    // 自动布局是异步的：ELK 还没算完时位置为空，直接跳过这次渲染，
    // 否则会把空画布画出来（看起来「什么都没有」）。
    if (!this.positions || Object.keys(this.positions).length === 0) return;
    this.nodeLayer.textContent = '';
    this.edgeLayer.textContent = '';

    this.renderGroups();

    // 节点
    this.view.nodes.forEach(function (n) { self.renderNode(n); });

    // 标签与批注
    this.noteBoxes = [];
    var placed = [];
    (this.view.notes || []).forEach(function (note) {
      var t = self.positions[note.at];
      if (!t) return;
      // 宽度按文本估算：固定 260px 会把长批注裁掉，且没被算进适配范围
      var MAX_NOTE_W = 260;
      var wrapped = wrapText(note.text, 6.6, MAX_NOTE_W - 12);
      var w = Math.ceil(Math.max.apply(null, wrapped.map(function (s) {
        return textWidth(s, 6.6);
      }))) + 12;
      // 行高 14px + 上下各 3px 余量：给足余量，否则最后一行会被 foreignObject 裁掉
      var h = 14 * wrapped.length + 8;
      var x = t.x + t.w + (note.dx || 16);
      var y = t.y + (note.dy == null ? t.h / 2 - h / 2 : note.dy);
      // 避让：批注默认贴在锚点右侧，若压到任何节点就依次换到左 / 上 / 下；
      // 四个方位都被占则退到全图最右侧，绝不盖住节点。
      var allBoxes = Object.keys(self.positions).map(function (k) {
        return self.positions[k];
      });
      function overlaps(a, b) {
        return !(a.x + a.w <= b.x || a.x >= b.x + b.w ||
                 a.y + a.h <= b.y || a.y >= b.y + b.h);
      }
      function hitsAny(box) {
        for (var i = 0; i < allBoxes.length; i++) if (overlaps(box, allBoxes[i])) return true;
        for (var j = 0; j < placed.length; j++) if (overlaps(box, placed[j])) return true;
        return false;
      }
      var candidates = [
        { x: t.x + t.w + (note.dx || 16), y: y },
        { x: t.x - w - 16, y: y },
        { x: t.x + t.w / 2 - w / 2, y: t.y - h - 12 },
        { x: t.x + t.w / 2 - w / 2, y: t.y + t.h + 12 }
      ];
      var chosen = null;
      for (var ci = 0; ci < candidates.length; ci++) {
        var box = { x: candidates[ci].x, y: candidates[ci].y, w: w, h: h };
        if (!hitsAny(box)) { chosen = box; break; }
      }
      if (!chosen) {
        var rightMost = Math.max.apply(null, allBoxes.map(function (p) { return p.x + p.w; }));
        chosen = { x: rightMost + 24, y: y, w: w, h: h };
      }
      placed.push(chosen);
      x = chosen.x; y = chosen.y;
      var fo = el('foreignObject', { x: x, y: y, width: w, height: h });
      var d = document.createElement('div');
      d.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');
      d.className = 'flow-note';
      d.textContent = wrapped.join('\n');
      d.setAttribute('style', 'white-space:pre-line');
      fo.appendChild(d);
      self.nodeLayer.appendChild(fo);
      self.noteBoxes.push({ x: x, y: y, w: w, h: h });
    });

    this.renderEdges();
    // 不在这里调用 KaTeX：节点与连线标签都是纯文本，走 KaTeX 会把整段文字
    // 当成公式拆开（实测会把 `post × out + residual` 竖排成 74px 高）。
    // 页面正文里的公式由 <head> 的 auto-render 负责，画布内不放公式。
    if (this.needsFit) {
      this.fitToScreen(this.fitMode);
      this.needsFit = false;
      this.fitMode = 'width';
    }
    this.applyTransform();
  };

  /**
   * 把图画缩放进视口。
   *
   * mode='width'（默认视图）：只保证横向放下、顶部对齐，纵向允许超出后滚动/
   *   拖动画布查看。attn 这类 20 节点、纵向很长的图，若按整图适配会被缩到
   *   0.39，节点文字小到看不清；按宽度适配能保持可读。
   * mode='all'（复位按钮）：整图放进视口，便于总览。
   */
  FlowView.prototype.fitToScreen = function (mode) {
    var ids = Object.keys(this.positions);
    if (!ids.length) return;
    var rect = this.viewport.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    ids.forEach(function (id) {
      var p = this.positions[id];
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x + p.w);
      maxY = Math.max(maxY, p.y + p.h);
    }, this);
    // 分组框（含它的标签）也要算进来，否则框顶与组名会被顶栏压住
    (this.groupBoxes || []).forEach(function (b) {
      minX = Math.min(minX, b.x);
      minY = Math.min(minY, b.y - 2);
      maxX = Math.max(maxX, b.x + b.w);
      maxY = Math.max(maxY, b.y + b.h);
    });
    // 批注也要算进来，否则长批注会缩到视口外
    (this.noteBoxes || []).forEach(function (b) {
      minX = Math.min(minX, b.x);
      minY = Math.min(minY, b.y);
      maxX = Math.max(maxX, b.x + b.w);
      maxY = Math.max(maxY, b.y + b.h);
    });

    var pad = overlayPad(this.viewport);
    var availW = rect.width - pad.left - pad.right;
    var availH = rect.height - pad.top - pad.bottom;
    var w = Math.max(maxX - minX, 1), h = Math.max(maxY - minY, 1);
    // 只缩不放：小图保持 100%
    var scale = mode === 'all'
      ? Math.min(1, availW / w, availH / h)
      : Math.min(1, availW / w);
    scale = Math.max(FIT_MIN_SCALE, scale);
    this.scale = scale;
    this.tx = pad.left + (availW - w * scale) / 2 - minX * scale;
    // 纵向：整图适配时居中，按宽度适配时顶部对齐
    var slack = availH - h * scale;
    this.ty = (mode === 'all' && slack > 0 ? pad.top + slack / 2 : pad.top) - minY * scale;
  };

  /**
   * 分组框。
   *
   * 取成员节点的并集外接矩形，并检查框内是否混进了非成员节点：ELK 会把
   * 同一层的节点排在一行，直接取并集会顺带圈住旁边无关的节点（看起来像
   * 分组错了）。这里逐层收紧：某一行若含非成员节点，该行的横向范围就不
   * 参与外接矩形，避免把无关节点圈进来。
   */
  FlowView.prototype.renderGroups = function () {
    var self = this;
    this.groupBoxes = [];
    (this.view.groups || []).forEach(function (g) {
      var boxes = g.members.map(function (m) { return self.positions[m]; }).filter(Boolean);
      if (!boxes.length) return;

      // 按 y 分组成行，行内若混入非成员，只保留成员自身的横向范围
      var rows = [];
      boxes.slice().sort(function (a, b) { return a.y - b.y; }).forEach(function (b) {
        var row = rows.filter(function (r) {
          return Math.abs(r.y - b.y) < Math.max(r.h, b.h) / 2;
        })[0];
        if (!row) rows.push({ y: b.y, h: b.h, xs: [b.x, b.x + b.w] });
        else {
          row.y = Math.min(row.y, b.y);
          row.h = Math.max(row.h, b.h);
          row.xs[0] = Math.min(row.xs[0], b.x);
          row.xs[1] = Math.max(row.xs[1], b.x + b.w);
        }
      });

      var outside = self.view.nodes.filter(function (n) {
        return g.members.indexOf(n.id) === -1;
      }).map(function (n) { return self.positions[n.id]; }).filter(Boolean);

      var x1 = Infinity, x2 = -Infinity;
      rows.forEach(function (r) {
        var blocked = outside.some(function (o) {
          return !(o.y + o.h < r.y - 6 || o.y > r.y + r.h + 6) &&
                 o.x < r.xs[1] + 24 && o.x + o.w > r.xs[0] - 24;
        });
        if (blocked) return;
        x1 = Math.min(x1, r.xs[0]);
        x2 = Math.max(x2, r.xs[1]);
      });
      if (!isFinite(x1)) { x1 = Math.min.apply(null, boxes.map(function (b) { return b.x; })); }
      if (!isFinite(x2)) { x2 = Math.max.apply(null, boxes.map(function (b) { return b.x + b.w; })); }
      var y1 = Math.min.apply(null, boxes.map(function (b) { return b.y; })) - 30;
      var y2 = Math.max.apply(null, boxes.map(function (b) { return b.y + b.h; })) + 16;

      var box = {
        x: x1 - 16, y: y1, w: x2 - x1 + 32, h: y2 - y1
      };
      self.groupBoxes.push(box);
      self.nodeLayer.appendChild(el('rect', {
        class: 'flow-group', x: x1 - 16, y: y1, width: x2 - x1 + 32, height: y2 - y1, rx: 8,
        // 分组一律不填色：只留一条虚线边界。彩色底块会和节点抢注意力，
        // 叠起来也显脏（参考标准里的分组框同样是纯白 + 虚线）。
        fill: 'none', stroke: g.stroke || '#aab4c4'
      }));
      // 标签宽度按文本测量：写死 280px 会把长组名截断
      var labelW = Math.ceil(textWidth(g.label, 6.8)) + 16;
      var label = el('foreignObject',
        { x: x1 - 4, y: y1 + 6, width: labelW, height: 18 });
      var div = document.createElement('div');
      div.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');
      div.className = 'flow-group-label';
      // 组名用统一的中性色：分组只表示「这几步属于同一子层」，
      // 不需要再用颜色区分种类。
      div.setAttribute('style', 'color:#6b7484');
      div.textContent = g.label;
      label.appendChild(div);
      self.nodeLayer.appendChild(label);
    });
  };

  FlowView.prototype.renderNode = function (n) {
    var self = this;
    var p = this.positions[n.id];
    var g = el('g', { class: 'flow-node' + (n.drill ? ' drill' : ''), 'data-id': n.id });
    var shape;
    if (n.kind === 'cache') {
      var ry = 9;
      shape = el('path', {
        class: 'cache',
        d: 'M' + p.x + ',' + (p.y + ry) + ' v' + (p.h - 2 * ry) +
           ' a' + (p.w / 2) + ',' + ry + ' 0 0 0 ' + p.w + ',0 v-' + (p.h - 2 * ry)
      });
      g.appendChild(shape);
      g.appendChild(el('ellipse', {
        class: 'cache', cx: p.x + p.w / 2, cy: p.y + ry, rx: p.w / 2, ry: ry
      }));
    } else {
      shape = el('rect', {
        // port = 跨视图传入/传出的量：画成虚线框，与视图内的实线框区分
        class: n.kind === 'op' ? 'op' : (n.kind === 'port' ? 'port' : 'tensor'),
        // 参考标准：张量框是直角黑边，算子框是大圆角浅蓝
        x: p.x, y: p.y, width: p.w, height: p.h, rx: n.kind === 'op' ? 10 : 0
      });
      g.appendChild(shape);
    }

    // 与 sizeFor 用同一个折行结果：否则测量与渲染不一致，文字会溢出
    var lines = wrapLines(n, p.w);
    var block = 0;
    lines.forEach(function (l) { block += l.lh; });
    var top = p.y + p.h / 2 - block / 2;
    var cursor = top;
    lines.forEach(function (line) {
      var lh = line.lh;
      var fo = el('foreignObject', { x: p.x + 5, y: cursor, width: p.w - 10, height: lh });
      var d = document.createElement('div');
      d.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');
      d.className = 'flow-cell';
      // 行高锁死为 foreignObject 的高度：让默认行高（约 1.2em）撑出 1px 溢出
      d.setAttribute('style', 'line-height:' + lh + 'px');
      var span = document.createElement('span');
      span.className = line.cls;
      span.textContent = line.text;
      span.title = line.text;
      d.appendChild(span);
      fo.appendChild(d);
      g.appendChild(fo);
      cursor += lh;
    });

    if (n.drill) {
      g.addEventListener('click', function (e) {
        if (self.dragMoved) return;
        e.stopPropagation();
        // 让目标视图自己决定缩放：applyView 内部会按宽度适配并置中，
        // 这里不要再把 scale 强设成 1，否则跳过去只看到一角。
        self.fitMode = 'width';
        self.applyView(n.drill);
      });
    }
    this.nodeLayer.appendChild(g);
    this.nodeEls[n.id] = g;
  };

  FlowView.prototype.renderEdges = function () {
    var self = this;
    this.edgeLayer.textContent = '';
    // 本次渲染里已被占用的走廊：避免两条边走同一条竖线而看起来重叠
    var usedCorridors = {};
    this.view.edges.forEach(function (edge, index) {
      var a = self.positions[edge.from], b = self.positions[edge.to];
      if (!a || !b) return;
      var d;
      // 两端都没被拖动过 → 用 ELK 的折点（它保证不穿框、不重线）；
      // 只要有一端被拖过，ELK 的折点就失效了，改用本地重算。
      var touched = self.movedNodes &&
        (self.movedNodes[edge.from] || self.movedNodes[edge.to]);
      if (self.edgeRoutes && self.edgeRoutes[index] && !touched) {
        // ELK 已经算好折点：直接用，保证不穿框、不重线
        var pts = self.edgeRoutes[index];
        // 终点回退，给箭头留出空间
        d = roundedPath(shortenTail(pts, CLEARANCE));
      } else {
        var obstacles = self.view.nodes
          .filter(function (n) { return n.id !== edge.from && n.id !== edge.to; })
          .map(function (n) { return self.positions[n.id]; })
          .filter(Boolean);
        d = routeEdge(a, b, obstacles, usedCorridors);
      }
      var path = el('path', {
        class: 'flow-edge',
        d: d,
        'marker-start': 'url(#flow-dot)',
        'marker-end': 'url(#flow-arrow)'
      });
      self.edgeLayer.appendChild(path);

      // 边标签：有 ELK 坐标就用它的，拖动后按路径中点重新摆放
      if (edge.label) {
        var lab = self.edgeLabels && self.edgeLabels[index];
        if (!lab || touched) {
          var mid = path.getPointAtLength(path.getTotalLength() / 2);
          lab = {
            x: mid.x - (textWidth(edge.label, 6.4) + 12) / 2,
            y: mid.y - 8,
            w: textWidth(edge.label, 6.4) + 12,
            h: 16
          };
        }
        var lfo = el('foreignObject', { x: lab.x, y: lab.y, width: lab.w, height: lab.h });
        var ldiv = document.createElement('div');
        ldiv.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');
        ldiv.className = 'flow-edge-label';
        ldiv.textContent = edge.label;
        lfo.appendChild(ldiv);
        self.edgeLayer.appendChild(lfo);
      }
    });
  };

  /**
   * 把路径末尾沿最后一段方向回退 dist 像素。
   * 箭头需要这段空间，否则会被目标框的边缘压住看不见。
   */
  function shortenTail(points, dist) {
    if (points.length < 2) return points;
    var out = points.map(function (p) { return [p[0], p[1]]; });
    var last = out[out.length - 1], prev = out[out.length - 2];
    var vx = last[0] - prev[0], vy = last[1] - prev[1];
    var len = Math.hypot(vx, vy);
    if (len <= dist) return out;
    out[out.length - 1] = [last[0] - vx / len * dist, last[1] - vy / len * dist];
    return out;
  }

  FlowView.prototype.applyTransform = function () {
    this.scene.setAttribute('transform',
      'translate(' + this.tx + ',' + this.ty + ') scale(' + this.scale + ')');
  };

  // ---------- 交互 ----------

  FlowView.prototype.bindEvents = function () {
    var self = this;
    var dragging = null, panning = false, lastX = 0, lastY = 0;
    this.dragMoved = false;

    this.viewport.addEventListener('mousedown', function (e) {
      var g = e.target.closest ? e.target.closest('.flow-node') : null;
      self.dragMoved = false;
      lastX = e.clientX; lastY = e.clientY;
      if (g) {
        var id = g.getAttribute('data-id');
        var p = self.positions[id];
        dragging = { id: id, startX: e.clientX, startY: e.clientY, ox: p.x, oy: p.y };
      } else {
        panning = true;
        self.viewport.classList.add('is-panning');
      }
      e.preventDefault();
    });

    window.addEventListener('mousemove', function (e) {
      if (dragging) {
        var dx = (e.clientX - dragging.startX) / self.scale;
        var dy = (e.clientY - dragging.startY) / self.scale;
        if (Math.abs(e.clientX - dragging.startX) > 2 ||
            Math.abs(e.clientY - dragging.startY) > 2) {
          if (!self.dragMoved) {
            // 只在真正开始拖动时才把节点提到最上层。
            // 不能在 mousedown 里做：appendChild 会把元素移出再插入，
            // 浏览器随之丢弃这次 click，节点上的「点击下钻」就永远不触发。
            var gEl = self.nodeEls[dragging.id];
            if (gEl) gEl.parentNode.appendChild(gEl);
          }
          self.dragMoved = true;
        }
        self.moveNode(dragging.id, dragging.ox + dx, dragging.oy + dy);
      } else if (panning) {
        self.tx += e.clientX - lastX;
        self.ty += e.clientY - lastY;
        lastX = e.clientX; lastY = e.clientY;
        self.applyTransform();
      }
    });

    window.addEventListener('mouseup', function () {
      dragging = null;
      panning = false;
      self.viewport.classList.remove('is-panning');
    });

    // 滚轮缩放：步长与 deltaY 成正比。固定系数会让触控板一次滑动放大几十倍。
    this.viewport.addEventListener('wheel', function (e) {
      e.preventDefault();
      var dy = e.deltaY;
      if (e.deltaMode === 1) dy *= 16;
      else if (e.deltaMode === 2) dy *= 100;
      dy = Math.max(-120, Math.min(120, dy));
      var next = Math.min(MAX_SCALE, Math.max(MIN_SCALE, self.scale * Math.exp(-dy * 0.0015)));
      if (Math.abs(next - self.scale) < 0.0005) return;
      var rect = self.viewport.getBoundingClientRect();
      var px = e.clientX - rect.left, py = e.clientY - rect.top;
      var k = next / self.scale;
      self.tx = px - (px - self.tx) * k;
      self.ty = py - (py - self.ty) * k;
      self.scale = next;
      self.applyTransform();
      self.updateZoomLabel();
    }, { passive: false });
  };

  /** 移动节点并重画受影响的连线。无边界限制，坐标可以是任意实数。 */
  FlowView.prototype.moveNode = function (id, x, y) {
    var p = this.positions[id];
    if (!p) return;
    p.x = x; p.y = y;
    // 被拖动的节点：不再用 ELK 的静态折点，改由本地实时重算，
    // 否则连线会停在原位（ELK 只在布局时算一次，不知道节点被拖走了）。
    if (!this.movedNodes) this.movedNodes = {};
    this.movedNodes[id] = true;
    var g = this.nodeEls[id];
    if (g) {
      var shape = g.querySelector('rect, path');
      if (shape && shape.tagName === 'path') {
        // 圆柱：整体重画
        this.nodeLayer.removeChild(g);
        var n = this.view.nodes.filter(function (v) { return v.id === id; })[0];
        this.renderNode(n);
      } else if (shape) {
        shape.setAttribute('x', x);
        shape.setAttribute('y', y);
        var node = this.view.nodes.filter(function (v) { return v.id === id; })[0];
        // 与 renderNode 一致：用折行后的行列表定位每一行
        var lines = wrapLines(node, p.w);
        var block = 0;
        lines.forEach(function (l) { block += l.lh; });
        var cursor = y + p.h / 2 - block / 2;
        Array.prototype.forEach.call(g.querySelectorAll('foreignObject'), function (fo, i) {
          var lh = lines[i] ? lines[i].lh : 15;
          fo.setAttribute('x', x + 5);
          fo.setAttribute('y', cursor);
          cursor += lh;
        });
      }
    }
    this.renderEdges();
  };

  FlowView.prototype.updateZoomLabel = function () {
    var label = this.app.querySelector('.flow-zoom-label');
    if (label) label.textContent = Math.round(this.scale * 100) + '%';
  };

  FlowView.prototype.zoomBy = function (factor) {
    var next = Math.min(MAX_SCALE, Math.max(MIN_SCALE, this.scale * factor));
    var rect = this.viewport.getBoundingClientRect();
    var px = rect.width / 2, py = rect.height / 2;
    var k = next / this.scale;
    this.tx = px - (px - this.tx) * k;
    this.ty = py - (py - this.ty) * k;
    this.scale = next;
    this.applyTransform();
    this.updateZoomLabel();
  };

  FlowView.prototype.resetView = function () {
    this.scale = 1; this.tx = 0; this.ty = 0;
    // 清掉拖动痕迹：复位后所有边重新采用 ELK 的折点，
    // 否则被拖过的边会一直用本地重算的简单路径。
    this.movedNodes = {};
    // 复位 = 重建布局 + 整图适配（总览用），与切视图时的按宽度适配区分开
    this.fitMode = 'all';
    this.applyView(this.view.id);
    this.updateZoomLabel();
  };

  root.DojoFlow = {
    mount: function (app, data) {
      var fv = new FlowView(app, data);
      var zoom = app.querySelector('.flow-zoom');
      if (zoom) {
        zoom.addEventListener('click', function (e) {
          var act = e.target.getAttribute('data-act');
          if (act === 'in') fv.zoomBy(1.25);
          else if (act === 'out') fv.zoomBy(0.8);
          else if (act === 'reset') fv.resetView();
        });
      }
      if (app.querySelector('.flow-bar')) {
        app.querySelector('.flow-bar').addEventListener('click', function (e) {
          var btn = e.target.closest('.flow-tab');
          if (!btn) return;
          fv.tx = 0; fv.ty = 0; fv.scale = 1;
          fv.applyView(btn.dataset.view);
          fv.updateZoomLabel();
        });
      }
      var initial = (location.hash || '').replace('#', '');
      if (initial && data.views.some(function (v) { return v.id === initial; })) {
        fv.applyView(initial);
      }
      window.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') { fv.applyView(data.views[0].id); }
      });
      fv.updateZoomLabel();
      return fv;
    }
  };
}(typeof globalThis !== 'undefined' ? globalThis : this));
