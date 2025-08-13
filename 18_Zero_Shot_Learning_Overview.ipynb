{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Zero-Shot Learning 講義ノート\n",
    "## 劉 慶豊 @ Hosei University\n",
    "### \\today"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. はじめに：医用画像解析の例"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Zero-Shot Learning（ZSL）は、訓練時に観測されなかったクラス（未学習クラス）に対しても、正確な分類を行うことを目指す機械学習の枠組みである。従来の教師あり学習では、訓練時とテスト時のクラスが一致していることを前提とするが、ZSLではこの前提を取り払う。\n",
    "\n",
    "このような汎化を実現するために、ZSLでは各クラスに対応する**意味特徴ベクトル**（semantic attribute vector）を導入し、視覚空間と意味空間の橋渡しを行う必要がある。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 医用画像解析の設定\n",
    "\n",
    "- **訓練時に観測された疾患（既知クラス）**：肺がん、肺炎、結核、気胸、サルコイドーシスなど20種類\n",
    "- **未知クラス（診断すべき疾患）**：ランゲルハンス細胞組織球症、リンパ脈管筋腫症、肺胞蛋白症など\n",
    "- **視覚特徴（入力空間）**：胸部CT画像から抽出された放射線特徴ベクトル $x \\in \\mathbb{R}^{d_x}$\n",
    "- **意味特徴（意味空間）**：疾患の臨床記述（例：「上肺野の多発結節」「喫煙歴との関連」「10歳以下での発症」など）をBioBERT等でベクトル化した $a_y \\in \\mathbb{R}^{d_z}$\n",
    "- **目的**：新しい症例画像が与えられたときに、**訓練に使われていない疾患名**に分類・診断すること"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. 問題設定と記号"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "入力空間を $\\mathcal{X} \\subset \\mathbb{R}^{d_x}$、出力空間（クラス集合）を $\\mathcal{Y}$ とする。$\\mathcal{Y}$ は既知クラス（学習済みクラス）と未知クラス（未学習クラス）に分割される：\n",
    "\n",
    "$$\n",
    "\\mathcal{Y} = \\mathcal{Y}^{\\text{seen}} \\cup \\mathcal{Y}^{\\text{unseen}}, \\quad \\mathcal{Y}^{\\text{seen}} \\cap \\mathcal{Y}^{\\text{unseen}} = \\emptyset\n",
    "$$\n",
    "\n",
    "訓練データ：\n",
    "$$\n",
    "D_{\\text{train}} = \\{(x_i, y_i)\\}_{i=1}^n, \\quad x_i \\in \\mathcal{X}, \\quad y_i \\in \\mathcal{Y}^{\\text{seen}}\n",
    "$$\n",
    "\n",
    "各クラス $y \\in \\mathcal{Y}$ に対して、意味特徴ベクトル $a_y \\in \\mathbb{R}^{d_z}$（例えば属性ベクトルやword2vecなど）が与えられていると仮定する。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. 手法の基本構造"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "ZSLでは、視覚特徴ベクトル $x \\in \\mathbb{R}^{d_x}$ を意味空間 $\\mathbb{R}^{d_z}$ に写像する線形変換 $W \\in \\mathbb{R}^{d_z \\times d_x}$ を学習する。\n",
    "\n",
    "### マッピングの定義\n",
    "$$\n",
    "\\hat{z} = W x\n",
    "$$\n",
    "\n",
    "ここで、$\\hat{z}$ は視覚特徴ベクトル $x$ を意味空間へ射影したベクトルである。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. What is Zero-Shot Learning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Zero-Shot Learning（ZSL）は、訓練時に観測されなかったクラス（未学習クラス）に対しても、正確な分類を行うことを目指す機械学習の枠組みである。従来の教師あり学習では、訓練時とテスト時のクラスが一致していることを前提とするが、ZSLではこの前提を取り払う。\n",
    "\n",
    "このような汎化を実現するために、ZSLでは各クラスに対応する**意味特徴ベクトル**（semantic attribute vector）を導入し、視覚空間と意味空間の橋渡しを行う必要がある。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. 学習：射影行列 $W$ の最適化"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "訓練データに対し、以下の目的関数を最小化することで $W$ を学習する：\n",
    "\n",
    "$$\n",
    "\\min_W \\sum_{i=1}^n \\left\\| W x_i - a_{y_i} \\right\\|^2 + \\lambda \\|W\\|_F^2\n",
    "$$\n",
    "\n",
    "- $x_i$：画像から抽出された視覚特徴ベクトル\n",
    "- $a_{y_i}$：クラス $y_i$ に対応する意味ベクトル\n",
    "- $\\lambda$：正則化項の係数\n",
    "- $\\|\\cdot\\|_F$：Frobeniusノルム"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. 推論：未知クラスの予測"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "テスト時には、未知クラス $y \\in \\mathcal{Y}^{\\text{unseen}}$ に対して以下を最大化することで予測を行う：\n",
    "\n",
    "$$\n",
    "\\hat{y} = \\arg\\max_{y \\in \\mathcal{Y}^{\\text{unseen}}} F(x, y) = \\arg\\max_y \\langle W x, a_y \\rangle\n",
    "$$\n",
    "\n",
    "ここで内積 $\\langle W x, a_y \\rangle$ は、$x$ の意味空間への射影 $W x$ とクラス $y$ の意味ベクトル $a_y$ との類似度を表す。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. スコア最大化と損失最小化の等価性"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "損失最小化とスコア最大化は、以下の変形により等価であることが示されている：\n",
    "\n",
    "\\begin{align}\n",
    "\t\\|W x_i - a_{y_i}\\|^2 &= (W x_i)^\\top (W x_i) - 2 (W x_i)^\\top a_{y_i} + \\|a_{y_i}\\|^2 \\\\\n",
    "\t&= \\text{条件によってほぼ一定} - 2 \\langle W x_i, a_{y_i} \\rangle + \\text{定数}\n",
    "\\end{align}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 補足：スコア最大化と損失最小化の関係とその解釈"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "ZSL の損失関数とスコア最大化の関係は、以下のように整理できる。\n",
    "\n",
    "#### 二乗損失の展開\n",
    "\n",
    "損失最小化問題：\n",
    "$$\n",
    "\\min_W \\sum_{i=1}^n \\left\\| W x_i - a_{y_i} \\right\\|^2 + \\lambda \\|W\\|_F^2\n",
    "$$\n",
    "\n",
    "損失項の展開：\n",
    "\\begin{align}\n",
    "\t\\left\\| W x_i - a_{y_i} \\right\\|^2 \n",
    "\t&= \\|W x_i\\|^2 - 2 (W x_i)^\\top a_{y_i} + \\|a_{y_i}\\|^2 \\\\\n",
    "\t&= x_i^\\top W^\\top W x_i - 2 x_i^\\top W^\\top a_{y_i} + \\|a_{y_i}\\|^2\n",
    "\\end{align}\n",
    "\n",
    "ここで：\n",
    "- 第1項 $x_i^\\top W^\\top W x_i = \\|W x_i\\|^2$ は $W$ に依存する\n",
    "- 第2項 $-2 x_i^\\top W^\\top a_{y_i}$ はスコア項に対応\n",
    "- 第3項 $\\|a_{y_i}\\|^2$ は $W$ に依存しないため、定数として無視できる"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 「$W^\\top W$ が一定」という表現の解釈\n",
    "\n",
    "文献によっては「$W^\\top W$ が一定」または「第1項が定数」と記述されるが、これは厳密には正しくない。以下のような条件下において近似的に成り立つと考えられる：\n",
    "\n",
    "1. **$W$ に対して強い正則化** $\\lambda \\|W\\|_F^2$ を課しており、$W$ の変動が小さい場合\n",
    "2. **入力ベクトル $x_i$ がすべて $\\|x_i\\| = 1$ に正規化**されており、$x_i^\\top W^\\top W x_i$ のばらつきが小さい場合\n",
    "3. **$W^\\top W = I$（直交性制約）**などが課されている場合\n",
    "\n",
    "したがって、「$W^\\top W$ が一定」とは、実際には $W$ の構造が安定しており、第1項 $\\|W x_i\\|^2$ の寄与が目的関数全体に対してあまり変化を与えない状況を指している。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### なぜ直交性制約 $W^\\top W = I$ を通常は課さないのか？"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Zero-Shot Learning（ZSL）において、写像行列 $W$ に直交性制約 $W^\\top W = I$ を課すことは一般的ではない。主な理由は以下のとおりである：\n",
    "\n",
    "1. **直交性制約は最適化が難しくなる**  \n",
    "直交性制約 $W^\\top W = I$ は非線形かつ非凸な制約条件であり、最適化が複雑になる。特に深層学習や大規模データセットを扱う場合、このような制約を厳密に満たすように $W$ を更新することは計算コストが高く、効率的でない。\n",
    "\n",
    "2. **汎用性の高いモデル設計に向かない**  \n",
    "ZSL では、視覚特徴を意味空間へ柔軟にマッピングすることが求められる。$W$ に直交性のような構造的制約を課すと、写像の自由度が制限され、必要な表現力が損なわれる可能性がある。\n",
    "\n",
    "3. **通常は正則化項で十分**  \n",
    "実際の応用では、Frobeniusノルム $\\|W\\|_F^2$ を目的関数に加えることで、$W$ のサイズや振る舞いを効果的に制御できる。このため、明示的に直交性を強制する必要性は低く、より簡便で汎用的な正則化で十分な性能が得られる。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### より正確な記述例\n",
    "\n",
    "> 「$\\|W x_i\\|^2$ は厳密には $W$ に依存するが、$W$ のノルムを強く制約したり、$W^\\top W$ に構造的な制約を課すことで、その影響は抑えられる。このような仮定のもとでは、スコア項の最大化と損失関数の最小化はほぼ等価な目的となる。」\n",
    "\n",
    "このため、最大スコアに基づく推論と二乗誤差による訓練が本質的に整合している。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. 閉形式解（Closed-form Solution）"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "最小二乗法による $W$ の解析解は以下の通りである：\n",
    "\n",
    "$$\n",
    "W^* = A X^\\top (X X^\\top + \\lambda I)^{-1}\n",
    "$$\n",
    "\n",
    "- $X = [x_1, \\dots, x_n] \\in \\mathbb{R}^{d_x \\times n}$\n",
    "- $A = [a_{y_1}, \\dots, a_{y_n}] \\in \\mathbb{R}^{d_z \\times n}$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. 図解：意味空間への写像"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "[視覚特徴 x] → [線形写像 W] → [射影ベクトル Wx] → [類似度計算] → [意味ベクトル ay]\n",
    "```\n",
    "\n",
    "*図1: 意味空間への写像プロセス*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. 応用"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 画像認識（未知の物体カテゴリの識別）\n",
    "\n",
    "- **目的**：テスト時に初めて出現するカテゴリ（例：「アフリカスイギュウ」「ホバークラフト」など）を、ラベルなし画像から自動で識別する。\n",
    "- **入力**：画像 $x$（JPEG等）を CNN で処理し、視覚特徴ベクトル $W x \\in \\mathbb{R}^{d_z}$ に変換。\n",
    "- **意味空間**：クラス名（日本語または英語）を word2vec や CLIP Text Encoder でベクトル化した $a_y$ を用意。\n",
    "- **推論**：\n",
    "$$\n",
    "\\hat{y} = \\arg\\max_{y \\in \\mathcal{Y}^{\\text{unseen}}} \\langle W x, a_y \\rangle\n",
    "$$\n",
    "- **評価**：Top-1 / Top-5 精度により、未知クラスへの分類精度を評価する。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 文書分類（新しいトピックの自動分類）\n",
    "\n",
    "- **目的**：AI倫理、宇宙資源、量子社会など、学習データに含まれていないトピックを自動分類する。\n",
    "- **入力**：文書 $x$（ニュース記事や論文）を BERT などでエンコードし、文埋め込みベクトルに変換。\n",
    "- **意味空間**：クラス名（トピック名）および代表キーワード（例：「倫理」「リスク」「透明性」）からベクトル $a_y$ を構成。\n",
    "- **推論**：\n",
    "$$\n",
    "\\hat{y} = \\arg\\max_{y \\in \\mathcal{Y}^{\\text{unseen}}} \\langle W x, a_y \\rangle\n",
    "$$\n",
    "- **評価**：新トピック文書に対して分類精度（accuracy）や F1 スコアを用いて評価。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 医用画像解析（稀少疾患の識別）"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**構造とプロセス**：\n",
    "\n",
    "- **訓練時に観測された疾患**：肺がん、肺炎、結核、気胸、サルコイドーシスなど一般的な疾患（計20種類）\n",
    "- **未知クラス（診断すべき疾患）**：ランゲルハンス細胞組織球症、リンパ脈管筋腫症、肺胞蛋白症など稀少疾患\n",
    "- **入力（視覚特徴）**：CT 画像から放射線特徴を抽出（3D CNNやradiomic feature）し、ベクトル $x \\in \\mathbb{R}^{d_x}$ を得る\n",
    "- **意味空間（疾患記述）**：各疾患の臨床的記述を日本語または英語でまとめたテキスト（例：「10歳以下で発症」「上肺野に多発性結節」「間質性陰影」「喫煙歴との関連」）を医療用言語モデル（BioBERT、PubMedBERTなど）でベクトル化して $a_y \\in \\mathbb{R}^{d_z}$ を構成\n",
    "- **推論**：\n",
    "$$\n",
    "\\hat{y} = \\arg\\max_{y \\in \\mathcal{Y}^{\\text{unseen}}} \\langle W x, a_y \\rangle\n",
    "$$\n",
    "- **評価**：テストセットには診断済みの症例（ただし訓練時に未使用）が含まれており、Top-1 精度や診断一致率で評価される。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**ZSLが有効となる医学的背景と意義**：\n",
    "\n",
    "- 稀少疾患の症例は非常に少なく、従来の教師あり学習では対応困難\n",
    "- 新型感染症や変異株など、「前例のない病態」への即時対応が求められる\n",
    "- 意味空間を用いた類似性ベース分類は、既知疾患の知識を活かして未知に対処できる\n",
    "- 医師の補助判断ツールとして、診断支援の信頼性向上に貢献"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**ZSLの究極的な目標（医療応用）**：\n",
    "\n",
    "> 「どのような病気か全くわからない医用画像を与えられたときに、訓練データに一度も登場しなかった疾患名に対しても、意味情報をもとに正確な分類・診断を行う」\n",
    "\n",
    "これは、医療DXの一環としてAI診断補助システムにとって極めて重要な要素となる。"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 }
}
