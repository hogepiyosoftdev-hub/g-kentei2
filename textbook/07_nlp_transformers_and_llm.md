# 第7章：自然言語処理・Transformer・LLM（現代AIの核心）

## 7-1. 自然言語処理の基礎と単語分散表現（Q306..Q320）

```mermaid
graph TD
    Text["生のテキスト文"] --> Morph["形態素解析（MeCab, Janome, Sudachi）<br>品詞分解・原形復元"]
    Morph --> Rep["ベクトル表現への変換"]
    Rep --> BoW["古典的手法：BoW / TF-IDF<br>単語の出現頻度のみ（語順・意味を無視）"]
    Rep --> Embed["分散表現：Word2Vec / FastText<br>高次元密ベクトル（意味の類似性とアナロジー）"]
```

### TF-IDF（単語の重要度スコア計算）
文書群の中で特定の単語がどれだけ重要で特徴的かを評価する指標。
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
- **TF（Term Frequency）**: 文書 $d$ 内における単語 $t$ の出現頻度。
- **IDF（Inverse Document Frequency）**: 全文書数 $N$ を単語 $t$ が出現する文書数 $\text{DF}(t)$ で割った対数。
  $$\text{IDF}(t, D) = \log \frac{N}{\text{DF}(t)}$$
  - 「て・に・を・は」のようにすべての文書に現れる単語は IDF が $0$ に近づき、特定の文書にしか現れない専門用語は IDF が跳ね上がる。

### 分布仮説（Distributional Hypothesis）とWord2Vec
言語学者ジョン・ルパート・ファース（J.R. Firth）の**「単語の意味は、その周辺の単語によって決まる」**という分布仮説に基づき、トマス・ミコロフらが2013年に提案した単語埋め込み手法。

```mermaid
flowchart LR
    subgraph CBOW ["① CBOW（文脈から中心語を予測）"]
        C_In["周辺の文脈単語<br>w(t-2), w(t-1), w(t+1), w(t+2)"] --> C_Hidden["隠れ層（平均）"] --> C_Out["中心単語 w(t)"]
    end
    subgraph Skipgram ["② Skip-gram（中心語から文脈を予測）"]
        S_In["中心単語 w(t)"] --> S_Hidden["隠れ層"] --> S_Out["周辺の文脈単語<br>w(t-2), w(t-1), w(t+1), w(t+2)"]
    end
```

| 比較項目 | CBOW (Continuous Bag-of-Words) | Skip-gram |
| :--- | :--- | :--- |
| **予測方向** | 周囲の文脈単語 $\to$ **中心単語を予測** | 中心単語 $\to$ **周囲の文脈単語を予測** |
| **計算速度** | **高速**（文脈ベクトルを平均して処理） | 相対的に低速（予測回数が多い） |
| **単語適性** | 高頻度語の学習に適している | **低頻度語（珍しい単語）の表現力に優れる** |
| **アナロジー演算** | ベクトルの加減算が可能（$\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx \vec{\text{Queen}}$） |
| **高速化技法** | **ネガティブサンプリング（Negative Sampling）**: 全語彙のソフトマックス分母計算を避け、数個の偽単語（負例）との2値分類に落とし込む手法 |

---

## 7-2. 系列モデルの進化（RNN $\to$ LSTM $\to$ Seq2Seq）（Q321..Q325）

### RNN（リカレントニューラルネットワーク）の限界
隠れ状態 $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$ を時系列方向にループさせる構造。
- **弱点**: 系列長が長くなると、時間の逆伝播（BPTT: Backpropagation Through Time）において勾配消失または勾配爆発が発生し、**長い過去の文脈を記憶できない（長期依存性の破綻）**。

### LSTM（Long Short-Term Memory, 1997）
セル状態 $C_t$（情報の高速道路）と3つのゲート機構によって長期記憶を可能にしたモデル。

```mermaid
graph LR
    Input["入力 x_t, h_{t-1}"] --> Forget["忘却ゲート（Forget Gate）<br>過去の記憶 C_{t-1} のうち何を捨てるかを決定"]
    Input --> InputGate["入力ゲート（Input Gate）<br>新しい情報のどれをセルに書き込むかを決定"]
    Input --> OutputGate["出力ゲート（Output Gate）<br>更新されたセル状態から次の隠れ状態 h_t を出力"]
```

---

## 7-3. Transformerアーキテクチャの完全解剖（Q326..Q340）

2017年にGoogleの研究チーム（Vaswani et al.）が発表した論文『Attention Is All You Need』によって登場した、現代AIの基盤構造です。RNNのような再帰構造を完全に撤廃し、**自己注意機構（Self-Attention）のみ**で系列全体を並列処理します。

```mermaid
graph TD
    subgraph TransformerBlock ["Transformer層の構成"]
        In["入力埋め込み ＋ 位置エンコーディング（Positional Encoding）"] --> SA["Multi-Head Self-Attention"]
        SA --> Add1["Add & Layer Normalization（残差接続＋層正規化）"]
        Add1 --> FFN["Position-wise Feed-Forward Network（GELU活性化）"]
        FFN --> Add2["Add & Layer Normalization"]
        Add2 --> Out["次の層または出力"]
    end
```

### Self-Attention（自己注意機構）の数理
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$
- **クエリ $Q$（Query）**: 「何を探しているか」を表す検索ベクトル。
- **キー $K$（Key）**: 「どんな内容を持っているか」を表すインデックスベクトル。
- **バリュー $V$（Value）**: 「実際の情報の中身」を表す値ベクトル。
- **$\sqrt{d_k}$ で割る理由（Scaled Dot-Product）**:
  次元数 $d_k$ が大きくなると内積の値が極端に巨大化し、ソフトマックス関数の勾配が極小化（飽和）して学習が停滞するため、分散を $1$ にスケーリングして勾配消失を防ぐ。
- **計算量とメモリ消費量**:
  系列長（トークン数）を $N$ としたとき、全トークン同士のペア（$N \times N$）を一度に計算するため、計算量およびメモリ消費量は **$O(N^2)$（系列長の二乗に比例）**となる。

### Encoder-Decoder Attention（Cross-Attention）の構造と役割
新シラバス（G2024#6）で出題が強化された、機械翻訳やSeq2Seq（T5など）における核心メカニズムです。

```mermaid
flowchart LR
    subgraph Encoder ["エンコーダ（入力文側）"]
        EncOut["エンコーダ最終層の隠れ状態<br>（入力文全体の文脈ベクトル）"]
    end
    subgraph Decoder ["デコーダ（生成文側）"]
        DecState["デコーダ現ステップの隠れ状態<br>（今まさに翻訳・生成している文脈）"]
    end

    EncOut -->|線形射影| Key["キー（Key: K）<br>『入力文のどこにどんな単語があるか』"]
    EncOut -->|線形射影| Val["バリュー（Value: V）<br>『入力文の各単語の持つ意味情報』"]
    DecState -->|線形射影| Query["クエリ（Query: Q）<br>『デコーダが今求めている情報』"]
    Query & Key --> AttnScore["内積スコア QK^T / √d_k<br>（関連度合いの照合）"]
    AttnScore --> Softmax["Softmax（確率重み付け）"]
    Softmax & Val --> Context["文脈加重和ベクトル<br>（デコーダ次のトークン予測へ結合）"]
```

| 比較項目 | Self-Attention（自己注意） | Encoder-Decoder Attention（Cross-Attention） |
| :--- | :--- | :--- |
| **Query ($Q$) の供給元** | **自分自身の入力系列**（エンコーダまたはデコーダ） | **デコーダ側**（現在生成中の系列） |
| **Key ($K$) の供給元** | **自分自身の入力系列** | **エンコーダ側**（入力文・原文の最終出力） |
| **Value ($V$) の供給元** | **自分自身の入力系列** | **エンコーダ側**（入力文・原文の最終出力） |
| **機能・目的** | 文中の単語同士の依存関係・文脈を把握する | 原文のどの単語に注目して次の単語を生成すべきか照合する |

---

## 7-4. BERT vs GPT：基盤モデルの2大潮流（Q341..Q350）

```mermaid
flowchart TD
    Transformer["Transformer (2017)"] --> EncoderOnly["エンコーダ単体<br>【双方向表現・文脈理解】"]
    Transformer --> DecoderOnly["デコーダ単体<br>【自己回帰・文章生成】"]
    EncoderOnly --> BERT["BERT (Devlin et al., 2018)<br>・双方向Self-Attention<br>・事前学習：MLM（マスク予測）＋NSP<br>・文書分類、固有表現抽出、検索に最適"]
    DecoderOnly --> GPT["GPTシリーズ (OpenAI, 2018〜)<br>・因果的マスク（Causal Mask / 未来を隠す）<br>・事前学習：次の単語の自己回帰予測<br>・チャット、要約、コード生成に最適"]
```

| 比較項目 | BERT (Bidirectional Encoder) | GPT (Generative Pre-trained Transformer) |
| :--- | :--- | :--- |
| **ベース構造** | Transformerの**エンコーダ** | Transformerの**デコーダ** |
| **注意の向き** | **双方向（前後の文脈を同時に参照）** | **片方向（左から右への自己回帰的マスク）** |
| **事前学習タスク** | **MLM（Masked LM: 15%をマスクして穴埋め）** ＋ **NSP（次文予測）** | **Next Token Prediction（次の単語を自己回帰予測）** |
| **得意領域** | 文書分類、感情分析、質問応答、意味検索 | 対話生成、文章作成、コード生成、翻訳 |

---

## 7-5. LLMの進化・プロンプティング・微調整・アライメント（Q351..Q375）

### スケーリング則（Scaling Laws）と創発的能力
- **Kaplan則 vs Chinchilla則（DeepMind）**:
  計算予算（Compute）が与えられたとき、モデルパラメータ数 $N$ と事前学習トークン数 $D$ を**同等の比率（等比級数的）でバランスよく拡大**することが最適解である（パラメータだけを巨大化してデータが不足していると未学習になる）。
- **創発的能力（Emergent Abilities）**:
  ある一定のパラメータ規模（通常数億〜数十億パラメータ以上）を突破した瞬間に、事前の明示的な学習なしに突然発現する高度な推論力、算術能力、文脈内学習能力。

### プロンプティング技術の体系
```mermaid
flowchart TD
    Root["<b>プロンプティング技術の体系</b><br>文脈内学習（In-Context Learning）と推論支援"]

    Root --> ZS["<b>Zero-shot プロンプティング</b><br>・タスク指示文のみを与え、追加の例示なしで推論<br>・モデル本来の基礎能力と事前学習に依存"]
    Root --> FS["<b>Few-shot プロンプティング</b><br>・入力と正解の具体例を2〜3件提示して推論<br>・パラメータ更新なしで出力形式や文脈を誘導"]
    Root --> CoT["<b>Chain-of-Thought（CoT: 思考の連鎖）</b><br>・『段階的に考えてみましょう』と指示<br>・途中計算・推論ステップの言語化により論理的推論力を劇的向上"]
    Root --> SC["<b>Self-Consistency（自己一貫性）</b><br>・同一プロンプトで複数のCoT推論を生成<br>・最も頻出する最終回答を多数決で決定"]
    Root --> ReAct["<b>ReAct（Reasoning + Acting）</b><br>・思考（Thought）→ 行動（Action: 検索や計算ツール）→ 観察（Observation）のループ<br>・外部API連携とハルシネーション抑制"]

    style Root fill:#202a3a,stroke:#e5a93b,stroke-width:2px,color:#fff
    style ZS fill:#18202c,stroke:#38d9d6,color:#e6edf3
    style FS fill:#18202c,stroke:#38d9d6,color:#e6edf3
    style CoT fill:#18202c,stroke:#38d9d6,color:#e6edf3
    style SC fill:#18202c,stroke:#38d9d6,color:#e6edf3
    style ReAct fill:#18202c,stroke:#e5a93b,color:#e6edf3
```

### PEFT（パラメータ効率的ファインチューニング）
モデル全体の全パラメータを更新する「フルファインチューニング」は巨大なGPUメモリを消費するため、少数の追加パラメータのみを更新する技術が必須となりました。

- **LoRA（Low-Rank Adaptation）**:
  元の重み行列 $W \in \mathbb{R}^{d \times k}$ を完全に凍結したまま、低ランク行列 $B \in \mathbb{R}^{d \times r}$ と $A \in \mathbb{R}^{r \times k}$（ランク $r \ll \min(d, k)$）の積 $\Delta W = B \times A$ のみを学習させる。学習パラメータ数を元の 0.1% 以下に削減可能。
- **QLoRA**:
  元のモデル重みを 4-bit NormalFloat (NF4) に量子化し、メモリを極限まで節約しつつ LoRA のみを高精度学習する手法。家庭用GPU1枚で数十億パラメータのモデルを調整可能。

### アライメント（安全性・有用性の調整）
```mermaid
sequenceDiagram
    participant Pre as 事前学習済みLLM
    participant SFT as ① 教師あり微調整（SFT）
    participant RM as ② 報酬モデル（Reward Model）
    participant PPO as ③ 強化学習（PPO）

    Pre->>SFT: 高品質な対話プロンプト・正解回答ペアで学習
    Note over RM: 人間の評価者が複数回答の優劣（ランキング）を判定
    SFT->>RM: ランキングデータから『人間の好ましさ』を点数化するモデルを学習
    RM->>PPO: 報酬スコアをフィードバックし方策を最適化（RLHF）
    Note over PPO: 安全で有用・誠実なモデルの完成！
```
- **DPO（Direct Preference Optimization）**: 報酬モデルを明示的に挟むことなく、人間の選好データ（好ましい回答 vs 好ましくない回答）から直接数理的にLLMを最適化する最新の簡略手法。

---

## 7-6. RAG（検索拡張生成）の構造と実装（Q376..Q385）

LLMの2大弱点である「最新情報の欠落」と「ハルシネーション（嘘の生成）」を解消する最重要ソリューション。

```mermaid
flowchart TD
    UserQ["ユーザーの質問<br>『当社の今年の有給休暇規定は？』"] --> Embed["質問文を埋め込みベクトル化"]
    Embed --> HybridSearch["ハイブリッド検索（Hybrid Search）<br>① ベクトル類似度検索（コサイン類似度）<br>＋ ② キーワード検索（BM25）"]
    Docs["社内文書・マニュアル・DB"] --> Chunk["チャンク分割・ベクトルDB"]
    Chunk --> HybridSearch
    HybridSearch --> Rerank["リランカー（Re-ranker）<br>関連度順に再スコアリング・上位抽出"]
    Rerank --> Context["コンテキスト（根拠文書）"]
    UserQ & Context --> Prompt["統合プロンプト<br>『以下の資料のみに基づいて回答してください：...』"]
    Prompt --> LLM["LLM（生成モデル）"]
    LLM --> Answer["根拠付き・ハルシネーションのない正確な回答"]
```

---
