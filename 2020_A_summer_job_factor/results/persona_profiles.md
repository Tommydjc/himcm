# High-school personas (K-Means, $K=3$)

Input: Thompson scores `results/student_factor_scores.csv` ($N=50$, $m=3$). Contest $K=3$. Elbow (inertia curvature) $K=2$. Max mean silhouette $K=7$ (at contest $K$, silhouette $=0.260$).

Silhouette by $K$: K=2: 0.232, K=3: 0.260, K=4: 0.237, K=5: 0.249, K=6: 0.263, K=7: 0.281, K=8: 0.235.

sklearn labels were permuted onto designed prototypes (Cluster 0 $\approx$ high $F_1$, Cluster 1 $\approx$ high $F_2$, Cluster 2 $\approx$ low $F_3$). **Extracted axes are not the designed cash / resume / sun triad**; read the centroid numbers, not the contest nicknames, as the evidence.

| cluster | contest name | $n$ | $\mu_{F_1}$ | $\mu_{F_2}$ | $\mu_{F_3}$ | template SSE |
|---|---|---:|---:|---:|---:|---:|
| 0 | 务实经济型 / The Pragmatist | 23 | +0.900 | +0.063 | +0.030 | 0.127 |
| 1 | 履历投资型 / The Career-Builder | 10 | -0.763 | +1.229 | +0.325 | 0.133 |
| 2 | 自由舒适型 / The Leisure-Seeker | 17 | -0.769 | -0.809 | -0.232 | 2.282 |

## Axis names (extracted)

- $F_1$: Human Capital & Networking Value
- $F_2$: Autonomy & Commute Convenience
- $F_3$: Physical & Mental Burden

## Cluster readings

### Cluster 0: 务实经济型 (The Pragmatist)

Designed story: Designed signature: very high F1 (cash/tips), moderate F2 and F3.

n=23. On this extraction, Cluster 0 is most distinctive on Human Capital & Networking Value. Human Capital & Networking Value is high (centroid +0.900); Autonomy & Commute Convenience is near-average (centroid +0.063); Physical & Mental Burden is near-average (centroid +0.030). F1 here is not hourly-wage yield; F2 is not resume value; F3 is burden with a positive flexibility loading in the EFA, so 'afraid of work' is only valid as a low-F3 reading, not as a designed cash-vs-leisure split.

### Cluster 1: 履历投资型 (The Career-Builder)

Designed story: Designed signature: very high F2, lower F1 (tutoring / camp).

n=10. On this extraction, Cluster 1 is most distinctive on Autonomy & Commute Convenience. Human Capital & Networking Value is low (centroid -0.763); Autonomy & Commute Convenience is high (centroid +1.229); Physical & Mental Burden is near-average (centroid +0.325). F1 here is not hourly-wage yield; F2 is not resume value; F3 is burden with a positive flexibility loading in the EFA, so 'afraid of work' is only valid as a low-F3 reading, not as a designed cash-vs-leisure split.

### Cluster 2: 自由舒适型 (The Leisure-Seeker)

Designed story: Designed signature: very low F3 burden, prefers easy indoor work.

n=17. On this extraction, Cluster 2 is most distinctive on Autonomy & Commute Convenience. Human Capital & Networking Value is low (centroid -0.769); Autonomy & Commute Convenience is low (centroid -0.809); Physical & Mental Burden is near-average (centroid -0.232). F1 here is not hourly-wage yield; F2 is not resume value; F3 is burden with a positive flexibility loading in the EFA, so 'afraid of work' is only valid as a low-F3 reading, not as a designed cash-vs-leisure split.

