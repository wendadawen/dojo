# step 3：初稿

按 `step2-大纲.md` 把内容写到 `content/concepts/<name>/index.html`。模板从 `templates/concept/index.html` 拷贝。

## 输入

- `step2-大纲.md` — 正文结构、顶部模块、子概念分配、折叠块分配
- `step1-术语.md` — 术语的小白解释和首次出现位置
- `step1-数字.md` — 数值和来源
- `step1-推导链条.md` — 推导的完整步骤
- `step1-读者障碍.md` — 读者会卡在哪、解决方案
- `step1-来源.md` — 每条论断的出处

## 写作顺序

1. 拷贝模板：`cp templates/concept/index.html content/concepts/<name>/index.html`
2. 写顶部模块：标题 → callout → 小白版定义
3. 按 `step2-大纲.md` 的认知阶段顺序写正文
4. 删掉模板里所有 HTML 注释、占位符、施工标记

## 写作要求

按 `content-guide.md` 写。补充：

- 顶部 callout 里出现的术语必须在 callout 里就地解释
- 每个数字按 `step1-数字.md` 标来源
- 每条论断按 `step1-来源.md` 标出处，集中在证据边界声明
- 推导折叠块按 `step1-推导链条.md` 的步骤写，不能跳步
- `step1-读者障碍.md` 里的每个障碍点都要在对应位置解决

## 审核标准

AI 检查 `index.html` 是否：

- 模板 HTML 注释、占位符、施工标记全部删除
- 顶部模块完整（标题、callout、小白版定义）
- 覆盖 `step2-大纲.md` 的所有认知阶段
- 核心子概念都有完整推导 + 数值例子 + 可运行代码
- 每个术语首次出现时已就地解释
- 每个数字有来源
- 每条论断有出处，集中在证据边界声明
- `step1-读者障碍.md` 里的障碍点都有对应解决位置
- 推导步骤和 `step1-推导链条.md` 一致，没跳步

完整覆盖才进 step 4。
