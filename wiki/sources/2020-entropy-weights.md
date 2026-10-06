# Source：2020 entropy_weights.csv

学生 Thompson 得分 50×3 的熵权。\(w=(0.479,0.319,0.202)\)。

```yaml
type: source
year: 2020
status: ingested
sources:
  - 2020_A_summer_job_factor/results/entropy_weights.csv
  - 2020_A_summer_job_factor/results/job_entropy_utility.csv
```

## 摘录

F3 仅在 \(U=w^\top(F_1,F_2,-F_3)\) 取负。岗位效用排序以 `job_entropy_utility.csv` 为准（Tutor 第一、Lifeguard 第八，见 method 页）。

## Related

- [熵权](../methods/entropy-weight-objective-scoring.md)
- [2020 问题](../problems/2020-a-summer-job.md)
