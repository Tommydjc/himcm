# paper

数字只从 `../results/*.csv` 抄。周步西缘与日步中心格不横填。

| 文件 | 用途 |
| --- | --- |
| `himcm_paper.tex` | 主论文：核、1 ha 卷积、KPP、官方 SAW、课程帕累托/PRISMS |
| `summary_sheet.tex` | 一页摘要（拒绝未入表的 2.8 m / 214 / 38.6% / 84.5% / 68.2%） |
| `hoa_community_statement.tex` | HOA/公园局「两区两阶段」指南信（课程扩展） |
| `hoa_letter.tex` | 旧一页信，数字已过期；以 `hoa_community_statement.tex` 为准 |
| `notes/` | 人写心得 |

编译（在本目录）::

    pdflatex himcm_paper.tex
    pdflatex summary_sheet.tex
    pdflatex hoa_community_statement.tex
