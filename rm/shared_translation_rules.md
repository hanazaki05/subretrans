# Shared English-Chinese Subtitle Rules

These rules are authoritative for refinement and semantic QA.

### Chinese Subtitle Rules
1. **Translation Quality:** Ensure accuracy, fluency, natural flow, and maintain consistency with context/character voices. Use conversational language (avoid overly formal).
2. **Formatting:** Do not copy English italic tags (`<i>`, `</i>`, `{\i1}`, `{\i0}`) into Chinese. Preserve any other ASS formatting tags already present in Chinese exactly.
3. **Punctuation:**
    - **Remove periods or commas** at the end of sentences.
    - Keep question marks (?) or exclamation marks (!) if appropriate.
    - **Use English ellipsis (`...`)** in Chinese text instead of the standard Chinese ellipsis (`……`).
4. **Terminology & Name Handling (Strict Priority Order):**
    - **PRIORITY 1 (Glossary):** Strictly follow the "User Terminology" list below.
    - **PRIORITY 2 (Acronyms):** Keep initial-based nicknames (e.g., "AJ", "DJ", "CC") in English.
    - **PRIORITY 3 (Standard):** Transliterate other personal names into standard Mandarin (e.g., Chris -> 克里斯, Fry -> 弗莱).
    - **Rank Handling:** Format Ranks typically as [Name] + [Rank] in Chinese (e.g., Lieutenant Carlisle -> 卡莱尔中尉).
5. **Specific treatment for "Sir" / "Ma'am" (Honorifics):**
    - **Courtroom Context (Highest Priority):** Translate as **"法官阁下"** when addressing the Judge/Presiding Officer during legal proceedings, regardless of gender or their military rank.
    - **Military Context:** Translate both as **"长官"** when addressing a superior officer in chain-of-command settings (e.g., addressing Mac or Harm).
    - **Civilian Context:** Distinguish by gender: translate "Sir" as **"先生"** and "Ma'am" as **"女士"**.
    - **Ambiguity Rule:** If a character is a superior officer but currently acting as a judge, prioritize the activity (e.g., sitting at the bench is **"法官阁下"**, not "长官").

### Context & Specific Handling (JAG TV Show)
- **Military Ranks:** Interpret "Commander, Captain, Major, Admiral" as **U.S. Navy or Marine Corps ranks** in Chinese.
- **Time Notation:** Rewrite "Zulu" notation to "Greenwich Time":
  - *Example:* Input `0930 Zulu` → Output `Greenwich Time 09:30` (in Chinese phrase context).
- **Location Notations:** Translate “cover my six” / “on my six” with “six” meaning the rear/behind position (not a time reference), e.g., “注意我后方” / “在我后方”
- **Word for Federal Agents:** If the word "agent" refers to a federal agent, translate it as "探员"

### User Terminology (Authoritative Glossary)
- Admiral: 将军
- Brumby: 布伦比
- Bud: 巴德
- Bud J Roberts: 小巴德·罗伯特
- Bud Roberts: 巴德·罗伯特
- Bud Roberts Jr: 小巴德·罗伯特
- CAG: 舰载机联队长
- Carolyn: 卡罗琳
- Chegwidden: 切格维登
- Colonel: 中校
- Colonel MacKenzie: 麦肯齐中校
- Colonel Sarah MacKenzie: 莎拉·麦肯齐中校
- Commander: 中校
- Commander Harmon Rabb: 哈蒙·拉布中校
- Commander Harmon Rabb, Jr: 小哈蒙·拉布中校
- Commander Harmon Rabb, Junior: 小哈蒙·拉布中校
- Commander Rabb: 拉布中校
- Coulter: 考尔特
- Dunsmore: 邓斯莫尔
- Galindez: 加林德兹
- Harm: 哈姆
- Harmon: 哈蒙
- Harmon Rabb: 哈蒙·拉布
- Harmon Rabb Jr: 小哈蒙·拉布
- Harriet: 哈丽特
- Imes: 艾姆斯
- JAG: 军法署
- Judge Advocate General: 军法署
- Latham: 莱瑟姆
- Lieutenant Bud Roberts: 巴德·罗伯特上尉
- Lieutenant J. Bud Roberts: 小巴德·罗伯特上尉
- Lieutenant J. Roberts: 小罗伯特上尉
- Lieutenant Roberts: 罗伯特上尉
- Lieutenant Sims: 西姆斯上尉
- Lieutenant Singer: 辛格上尉
- Loren Singer: 劳伦·辛格
- Mac: 麦可
- MacKenzie: 麦肯齐
- Mattoni: 马托尼
- Mic: 米克
- mutiny: 兵变
- Naval Criminal Investigative Service: 海军刑事调查局
- Navy's Judge Advocate General Corps: 海军军法署
- NCIS: 海军刑事调查局
- Petty officer: 士官
- Rabb: 拉布
- Sarah: 莎拉
- Sims: 西姆斯
- Tiner: 泰纳
- Webb: 韦布
- XO: 大副
- Yeoman: 事务员
- Zulu time: 格林尼治时间
- Dr. Walden: 沃尔登医生
- Walden: 沃尔登医生
- Mikey: 麦奇

### Alignment (English-Chinese Matching)
- **Bilingual Alignment:** Keep English and Chinese semantically aligned to the same spoken content.
- **3-Line Window:** For cross-line sentences/ideas, alignment and adjustments may span adjacent entries but MUST NOT exceed **3 consecutive lines**.
- **Chinese Reordering:** Within the same 3-line window, you MAY reorder Chinese clauses for a more natural flow, as long as meaning remains faithful and no content is added/removed.
- **No Re-segmentation:** Do NOT merge/split entries; do NOT change `id`s, order, or item count—only edit text within each item.
