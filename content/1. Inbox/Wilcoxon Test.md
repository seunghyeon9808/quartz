---
dg-publish: "true"
---

# Wilcoxon Test (윌콕슨 검정)

**윌콕슨 검정(Wilcoxon Test)**은 화학자이자 통계학자인 **프랭크 윌콕슨(Frank Wilcoxon)**이 1945년 논문에서 제안한 **순위(Rank) 기반 비모수 통계 검정법(Non-parametric statistical test)**이다. 1) 데이터가 정규분포를 따르지 않거나(정규성 가정 위배), 2) 서열척도(Ordinal scale)이거나, 3) 표본 크기가 작고 이상치(Outlier)가 존재할 때 모수적 검정(t-test)의 대안으로 사용된다.

일반적으로 'Wilcoxon Test'라고 부를 때는 문맥에 따라 아래 두 가지 검정 중 하나(주로 대응표본인 부호순위 검정)를 지칭하므로 구분이 필요하다.

| 구분            | 1. 윌콕슨 부호순위 검정<br>(Wilcoxon Signed-Rank Test)                    | 2. 윌콕슨 순위합 검정<br>(Wilcoxon Rank-Sum Test)                                    |
| :------------ | :--------------------------------------------------------------- | :--------------------------------------------------------------------------- |
| **표본 특성**     | **대응표본 (Paired / Dependent samples)**<br>동일 대상의 전·후 비교, 짝지어진 데이터 | **독립표본 (Unpaired / Independent samples)**<br>서로 독립된 두 집단 간 비교                |
| **모수적 대응 기법** | 대응표본 t-검정 (Paired t-test)                                        | 독립표본 t-검정 (Independent two-sample t-test)                                    |
| **동일/확장 기법**  | 1표본 또는 대응표본 중앙값 차이 검정                                            | **Mann–Whitney U Test**와 수학적으로 동치<br>(3개 이상 집단 확장 시 **Kruskal–Wallis Test**) |
| **핵심 아이디어**   | 각 쌍의 차이($d_i$)의 절댓값에 순위를 매긴 후, 원래 부호($+$/$-$)를 복원하여 양수/음수 순위합 비교 | 두 집단의 데이터를 하나로 합쳐 전체 순위를 매긴 후, 각 집단의 순위합을 비교                                 |

---

## 1. 원문 논문 (Original Foundational Paper) 정리

### 서지 정보 (Bibliographic Info)
- **논문 제목**: *Individual Comparisons by Ranking Methods*
- **저자**: Frank Wilcoxon (American Cyanamid Company)
- **학술지**: *Biometrics Bulletin*, Vol. 1, No. 6 (Dec., 1945), pp. 80–83 (4 pages)
- **DOI**: [10.2307/3001968](https://doi.org/10.2307/3001968)
- **원문 링크**: [JSTOR (3001968)](https://www.jstor.org/stable/3001968) | [Semantic Scholar](https://www.semanticscholar.org/paper/Individual-Comparisons-by-Ranking-Methods-Wilcoxon/18367763f7d8eb7aa8b3143ce2ad8bb9c0dd11a6)

### 논문 배경 및 핵심 기여
1. **계산의 간소화와 실용성**: 당시 연구 현장에서 주로 쓰이던 Student's t-test나 분산분석(ANOVA)은 계산량이 많고 데이터의 정규성(Normality)을 가정해야 했다. 윌콕슨은 실제 측정값 대신 **순위(Rank)로 대체**하여 암산이나 간단한 덧셈만으로도 두 처리(Treatment) 간 유의미한 차이가 있는지 신속하게 판별할 수 있는 방법을 고안했다.
2. **비모수 통계학의 기틀 마련**: 단 4쪽 분량의 이 논문에서 현대 비모수 통계학의 양대 산맥인 **독립표본 순위합 검정(Rank-Sum Test)**과 **대응표본 부호순위 검정(Signed-Rank Test)**을 동시에 최초로 제시했다.

### 원문 논문의 주요 내용 및 실험 예제
논문은 크게 두 가지 실험 상황과 확률 계산법으로 구성되어 있다.

#### ① Unpaired Experiments (독립표본 비교 → 현 Wilcoxon Rank-Sum Test)
- **원문 예제 (Fly Spray 실험)**: 두 가지 파리 살충제 제제(Sample A, Sample B)의 살충률(%)을 각각 8번씩 총 16회 반복 측정한 데이터를 비교했다.
- **절차**:
  1. 두 그룹(A, B)의 관측치 16개를 합쳐 가장 작은 값(1위)부터 가장 큰 값(16위)까지 순위를 매긴다.
  2. 동일한 측정값(동점, Tie)이 있을 경우 해당 순위들의 **평균 순위(Mean rank)**를 부여한다 (예: 12위와 13위가 동점이면 각각 $12.5$위 부여).
  3. 한 그룹의 순위 총합을 구한 뒤, 귀무가설 하에서 그 합보다 같거나 극단적인 순위합이 나올 확률을 조합론적 공식으로 계산한 임계값 표와 비교한다.

#### ② Paired Comparisons (대응표본 비교 → 현 Wilcoxon Signed-Rank Test)
- **원문 예제 (Charles Darwin의 옥수수 데이터 & 살충제 일별 비교)**: 피셔(R. A. Fisher)의 저서에도 인용된 찰스 다윈의 실험 데이터(타가수정 옥수수와 자가수정 옥수수 15쌍의 높이 차이)를 예제로 사용했다.
- **절차**:
  1. 각 쌍(Pair)의 수치 차이($d_i$)를 구한다.
  2. 부호($+$, $-$)를 무시하고 **차이의 절댓값($|d_i|$) 크기 순서대로** 1위부터 순위를 매긴다 (동점은 평균 순위 부여).
  3. 부여된 순위에 원래 차이의 부호($+$ 또는 $-$)를 다시 붙인다.
  4. 양($+$)의 순위합과 음($-$)의 순위합을 각각 계산한다. 두 처리 간 차이가 없다면(귀무가설) 양수 순위합과 음수 순위합이 비슷하게 나누어져야 하며, 둘 중 작은 순위합($T$)이 임계값 이하일 때 통계적으로 유의한 차이가 있다고 판단한다.

---

## 2. Wilcoxon Signed-Rank Test 상세 이론 및 수식

### 기본 가정 (Assumptions)
1. **대응표본 (Paired / Dependent observations)**: 동일한 대상에 대한 반복 측정(전·후)이거나 짝지어진 쌍으로 구성되어야 하며, 서로 다른 쌍끼리는 독립적이어야 한다.
2. **연속형 또는 서열 척도 (Metric or Ordinal data)**: 두 관측치 간 차이($d_i$)의 크기를 비교해 순위를 매길 수 있어야 한다 (명목척도 불가).
3. **차이 분포의 대칭성 (Symmetry of differences)**: 정규분포를 따를 필요는 없으나, 중앙값(Median)의 위치 이동(Location shift) 검정으로 해석하려면 쌍별 차이($d_i$)의 분포가 중앙값을 기준으로 대칭에 가까워야 한다.

### 가설 설정 (Hypotheses)
두 대응표본 $X, Y$의 쌍별 차이를 $d_i = Y_i - X_i$라 할 때 (대칭성 가정 하):
- **귀무가설 ($H_0$)**: 두 집단 차이의 중앙값은 $0$이다 ($M_d = 0$, 즉 전·후 중심 차이가 없다).
- **대립가설 ($H_1$)**: 두 집단 차이의 중앙값은 $0$이 아니다 ($M_d \neq 0$, 양측검정 기준).

### 검정 통계량 계산 단계
1. **쌍별 차이 계산**: 각 쌍 $i = 1, \dots, N$에 대해 차이 $d_i = Y_i - X_i$를 계산한다.
2. **차이가 0인 쌍 제외**: $d_i = 0$인 데이터는 분석에서 제외하고, 남은 유효 쌍의 개수를 $n$이라 한다.
3. **절댓값 순위 부여**: $|d_i|$를 오름차순으로 정렬하여 순위 $R_i \in \{1, \dots, n\}$를 부여한다.
   - **동점(Tied ranks) 처리**: 절댓값이 같은 관측치들이 있으면 해당 순위들의 평균을 부여한다.
4. **부호별 순위합 계산**:
   - 양수 차이($d_i > 0$)들의 순위합: $W^+ = \sum_{d_i > 0} R_i$
   - 음수 차이($d_i < 0$)들의 순위합: $W^- = \sum_{d_i < 0} R_i$
   - 전체 순위합 관계식: $W^+ + W^- = \frac{n(n+1)}{2}$
   - 검정 통계량 $W$ (또는 $T$, $V$): 소프트웨어 및 문헌에 따라 $\min(W^+, W^-)$를 쓰거나 $W^+$를 사용한다 (예: R의 `wilcox.test`는 $V = W^+$ 출력, SciPy는 $\min(W^+, W^-)$ 기본 출력).

### 정규 근사 (Normal Approximation) 및 동점 보정
표본 크기 $n$이 작을 때는 정확 검정(Exact distribution)을 사용하고, $n$이 충분히 클 때는 정규분포 근사를 통해 $z$-통계량을 계산한다.

- **귀무가설 하의 기댓값 ($\mu_W$)**:
  $$\mu_W = \frac{n(n+1)}{4}$$
- **귀무가설 하의 표준편차 ($\sigma_W$, 동점이 없을 때)**:
  $$\sigma_W = \sqrt{\frac{n(n+1)(2n+1)}{24}}$$
- **동점(Ties)이 존재할 때의 표준편차 보정식**:
  각 동점 그룹 $k$에 속한 관측치 수를 $t_k$라 할 때,
  $$\sigma_W = \sqrt{\frac{n(n+1)(2n+1) - \sum_{k} \frac{t_k^3 - t_k}{2}}{24}}$$
- **표준화 검정 통계량 ($z$-value)**:
  $$z = \frac{W - \mu_W}{\sigma_W}$$
  *(연속성 보정(Continuity correction) 적용 시 분자의 $|W - \mu_W|$에서 $0.5$를 차감)*

### 효과 크기 (Effect Size, $r$)
귀무가설 기각 여부(p-value) 외에 실제 차이의 크기를 나타내기 위해 상관계수 형태의 효과 크기 $r$을 주로 사용한다:
$$r = \frac{z}{\sqrt{n}}$$
- **Cohen(1988) 기준 해석**:
  - $|r| < 0.1$: 효과 거의 없음
  - $0.1 \le |r| < 0.3$: 작은 효과 (Small effect)
  - $0.3 \le |r| < 0.5$: 중간 효과 (Medium effect)
  - $|r| \ge 0.5$: 큰 효과 (Large effect)

---

## 3. R 및 Python 실습 코드

### R 코드 (`stats::wilcox.test`)
```r
# 1. 정규성 검정 (Shapiro-Wilk test)
shapiro.test(dat$Price) # p-value < 0.05 이면 정규성 기각 -> 비모수 검정 수행

# 2. 윌콕슨 순위합 검정 (독립표본, Wilcoxon Rank-Sum / Mann-Whitney U Test)
wilcox.test(Price ~ Origin, data = dat, paired = FALSE)

# 3. 윌콕슨 부호순위 검정 (대응표본, Wilcoxon Signed-Rank Test)
wilcox.test(weight ~ group, data = dat2, paired = TRUE)

# (참고) 3개 이상 독립 집단 비모수 비교: Kruskal-Wallis Test
kruskal.test(Horsepower ~ Cylinders, data = dat)
```

### Python 코드 (`scipy.stats`)
```python
from scipy import stats

# 1. 정규성 검정 (대응표본의 경우 쌍별 차이의 정규성 확인)
diffs = [after - before for before, after in zip(before_list, after_list)]
stat_norm, p_norm = stats.shapiro(diffs)

# 2. 윌콕슨 부호순위 검정 (대응표본, Wilcoxon Signed-Rank Test)
stat_w, p_val = stats.wilcoxon(before_list, after_list, alternative='two-sided')

# 3. 윌콕슨 순위합 / 만-위트니 U 검정 (독립표본, Mann-Whitney U Test)
stat_u, p_val_u = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
```

---

## 4. 레퍼런스 요약 및 출처 (References)

- **[원문 논문] Wilcoxon, F. (1945).** *Individual Comparisons by Ranking Methods*. Biometrics Bulletin, 1(6), 80–83.
  - 링크: https://doi.org/10.2307/3001968 | https://www.jstor.org/stable/3001968
  - 요약: 순위(Rank)를 활용해 모수적 가정 없이 두 집단(독립표본 및 대응표본)의 차이를 검정하는 방법을 최초로 제안한 원전 논문.

- **[1] 네이버 블로그 (의학논문 치트키 / 리서치팩토리) - [R 통계분석] 14. 그룹 간 중심 비교**
  - 링크: https://m.blog.naver.com/paperfactor_ceo/222221772118
  - 요약: 정규성 검정(`shapiro.test`)을 만족하지 못할 때 사용하는 비모수 검정 3종(Wilcoxon Rank-Sum Test, Wilcoxon Signed-Rank Test, Kruskal-Wallis Test)의 개념을 평균 대신 **중위수(Median/중심) 비교** 관점에서 쉽게 설명하고, R의 `wilcox.test()` 및 `kruskal.test()` 실습 예제와 결과 해석법을 정리한 자료.

- **[2] numiqo Tutorial - Wilcoxon signed-rank test (Dr. Hannah Volk-Jesussek)**
  - 링크: https://numiqo.com/tutorial/wilcoxon-test
  - 요약: 대응표본 t-검정의 비모수 대안인 윌콕슨 부호순위 검정의 6가지 기본 가정, 가설 설정, 검정력(Test power) 비교, 단계별 수식 계산 과정(동점 순위 평균 처리 및 표준편차 동점 보정, 연속성 보정, 정규 근사 $z$-값 산출), 그리고 Cohen(1988) 기준의 효과 크기($r = z/\sqrt{n}$) 해석까지 체계적으로 다룬 튜토리얼.

- **[3] Wikipedia - Wilcoxon signed-rank test**
  - 링크: https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
  - 요약: 윌콕슨 부호순위 검정의 수학적 정의, 역사적 배경(Frank Wilcoxon 1945년 제안 및 Pratt의 0-차이 처리법 등), 검정 통계량 분포, 점근적 상대 효율(Asymptotic Relative Efficiency, 정규분포 데이터에서도 t-test 대비 약 $95.5\%$의 효율을 보임) 및 타 검정과의 관계를 정리한 백과사전 문서.
