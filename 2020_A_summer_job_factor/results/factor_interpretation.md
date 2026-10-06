# Factor interpretation (LLMFactor)

Semantic names and job coordinates for the **extracted** Varimax factors in `results/factor_loadings.csv`. Numbers in the top-|loading| tables are copied from that CSV. This file does not invent a new rotation.

- Backend: `ollama`
- Model: `qwen2.5:7b`
- Student scores (same axes): `results/student_factor_scores.csv`
- Questionnaire: `data/raw/OriginalData.csv` (see `DATA_SOURCE.txt`; 15-col table is SYNTHETIC_FALLBACK if GitHub was 7-D)

## Contest narrative vs this extraction

A designed 15-item story maps F1→Immediate Financial Yield (wage/tips), F2→Human Capital & Career Value (skill/resume/network), F3→Ergonomic & Physical Burden (fatigue/sun/inflexibility). **That triad is not what Kaiser-normalized Varimax recovered here.** LLMFactor therefore names the extracted axes; the designed names are kept only as a foil.

| Designed $F_j$ | zh | en | designed anchors |
|---|---|---|---|
| designed 1 | 即时经济流动性回报 | Immediate Financial Yield | Designed items: high hourly wage and tip potential. |
| designed 2 | 长程人力资本投资 | Human Capital & Career Value | Designed items: skill acquisition, resume value, social networking. |
| designed 3 | 身心环境负荷代价 | Ergonomic & Physical Burden | Designed items: fatigue, sun exposure, poor schedule flexibility. |

## Extracted factors (top-4 $|\lambda^*_{ij}|$)

### $F_1$: 人力资本与网络价值 (Human Capital & Networking Value)

| rank | item | $\lambda^*$ | $|\lambda^*|$ |
|---|---|---:|---:|
| 1 | `safety_level` | +0.7503 | 0.7503 |
| 2 | `social_networking` | +0.6181 | 0.6181 |
| 3 | `skill_acquisition` | +0.5955 | 0.5955 |
| 4 | `tip_potential` | +0.5935 | 0.5935 |

该因素综合了技能获取、社交网络和小费潜在收益，反映了青少年对长期人力资本投资和个人网络构建的重视。正向加载表明这些因素对他们的积极影响。

**CSV grounding (authoritative):** `safety_level` (+0.750); `social_networking` (+0.618); `skill_acquisition` (+0.596); `tip_potential` (+0.593).

### $F_2$: 自主与通勤便利 (Autonomy & Commute Convenience)

| rank | item | $\lambda^*$ | $|\lambda^*|$ |
|---|---|---:|---:|
| 1 | `autonomous_control` | +0.7613 | 0.7613 |
| 2 | `commute_convenience` | +0.6527 | 0.6527 |
| 3 | `mental_stress` | +0.5027 | 0.5027 |
| 4 | `hourly_wage_need` | +0.4818 | 0.4818 |

该因素主要由自主控制和通勤便利性构成，反映了青少年对工作自主性和通勤便捷性的关注。正向加载表明这些因素对他们工作的积极影响。

**CSV grounding (authoritative):** `autonomous_control` (+0.761); `commute_convenience` (+0.653); `mental_stress` (+0.503); `hourly_wage_need` (+0.482).

### $F_3$: 身体与心理负荷 (Physical & Mental Burden)

| rank | item | $\lambda^*$ | $|\lambda^*|$ |
|---|---|---:|---:|
| 1 | `physical_strength` | +0.7174 | 0.7174 |
| 2 | `boss_fairness` | -0.6706 | 0.6706 |
| 3 | `flexible_hours` | +0.5770 | 0.5770 |
| 4 | `outdoor_sun_exposure` | +0.3931 | 0.3931 |

该因素由体力需求、上司公平性和灵活工时组成，反映了青少年在体力和心理上的工作负担。正向加载表明体力需求和灵活工时的积极影响，而反向加载（-0.6706）则显示了较高的上司公平性对心理压力的积极影响。

**CSV grounding (authoritative):** `physical_strength` (+0.717); `boss_fairness` (-0.671); `flexible_hours` (+0.577); `outdoor_sun_exposure` (+0.393).

## Job coordinates $F_{\mathrm{job}}\in[-3,3]^3$

Each of 8 ads in `data/job_descriptions/` is scored on the **extracted** names above (not on the designed triad). Values are clipped to $[-3,3]$.

| job_id | title | $F_1$ | $F_2$ | $F_3$ |
|---|---|---:|---:|---:|
| 0 | Lifeguard | -1.000 | +0.000 | +1.000 |
| 1 | Camp Counselor | -0.500 | +0.500 | +0.500 |
| 2 | Tutor | +0.600 | -0.480 | -0.390 |
| 3 | Fast Food Cashier | -0.300 | +0.000 | +0.600 |
| 4 | Retail Clerk | +0.200 | -0.100 | +0.400 |
| 5 | Caddy | +0.580 | -0.260 | +0.660 |
| 6 | Pet Sitter | +0.120 | +0.260 | -0.240 |
| 7 | Swim Coach | +0.480 | +0.250 | +0.520 |

### Rationales

**0 Lifeguard** (`0_lifeguard`)

Lifeguard duties are physically demanding with no significant opportunity for skill acquisition or networking. Pay is standard for summer jobs, but the high physical and mental burden, including 6-8 hours of outdoor sun exposure and the necessity for constant visual surveillance, significantly outweighs any potential benefits. The role offers no significant autonomy or commute convenience, with strict schedules and local commutes.

**1 Camp Counselor** (`1_camp_counselor`)

The job of a Camp Counselor involves significant interpersonal and leadership duties, which contribute to its human capital value, but the moderate pay and modest resume benefits place it slightly below average in this factor. The work offers some autonomy and flexible scheduling, especially for day camps, which improves mental and physical convenience. However, the high physical load and long hours, along with the emotional labor of managing homesickness and bullying, make the job quite demanding.

**2 Tutor** (`2_private_tutor`)

The tutor role offers significant skill acquisition and networking value, which aligns with a score of +0.60 on F1. The high pay and scheduling flexibility contribute to a score of -0.48 on F2, as the autonomy and commute convenience are moderate, with no tips and some mental stress. The sedentary nature and minimal physical load suggest a score of -0.39 on F3, considering the cognitive effort involved without heavy physical strain.

**3 Fast Food Cashier** (`3_fast_food_crew`)

Fast Food Cashier duties focus on repetitive tasks like taking orders and assembling food, which do not significantly contribute to skill acquisition or networking. The pay is low, and the job does not offer much in the way of resume enhancement. Autonomy is limited, with managers building the weekly roster, and scheduling is not highly flexible. The physical demands, including prolonged standing and lifting, are notable, but the mental stress and safety concerns are relatively low. Outdoor sun exposure is minimal, and the environment is primarily indoor with some heat and noise.

**4 Retail Clerk** (`4_retail_sales`)

For F1 (Human Capital & Networking Value): The retail job offers moderate resume value in sales, merchandising, and customer service, which could be beneficial for business-related roles but not as strong as tutoring or lifeguarding. The social networking aspect is average as it involves customer interactions but not as high-level networking. Skill acquisition is limited to basic sales and customer service skills. Tip potential is not a factor in U.S. retail. For F2 (Autonomy & Commute Convenience): Autonomy control is limited as employees follow scripts and may face monitoring, though scheduling can be flexible. Commute convenience is average, with the need to travel to a mall setting. Mental stress is moderate due to customer service scripts and potential for theft monitoring. Hourly wage is adequate but may fluctuate based on sales performance. For F3 (Physical & Mental Burden): Physical load is moderate with long standing and some lifting but no significant heat or outdoor exposure. Mental burden comes from following scripts and maintaining a visually appealing store, with potential for stress from theft monitoring. Boss fairness is generally positive but can vary based on individual management styles.

**5 Caddy** (`5_golf_caddy`)

The caddie job offers significant human capital and networking value due to the interaction with high-net-worth professionals and the potential for receiving internship referrals. However, the primary skill acquisition is more about customer service and local knowledge rather than traditional academic skills. Autonomy is somewhat limited as the caddie’s schedule is managed by the caddie master, and there is a physical burden from carrying the bag and walking long distances. Mental stress is moderate due to the etiquette and heat pressure, but the autonomy and commute convenience are somewhat lower due to the mandatory show-up at the barn and the outdoor nature of the work.

**6 Pet Sitter** (`6_pet_sitter`)

Pet sitting and dog walking provide moderate networking opportunities through client interactions and the potential for recommending services to neighbors, which could be considered a low to moderate Human Capital & Networking Value (F1: 0.12). The autonomy to schedule visits and the lack of a rigid work environment offer some level of control and convenience, but the commute to multiple homes can be time-consuming, giving it a low Autonomy & Commute Convenience score (F2: 0.26). The physical load and occasional sun exposure are moderate, as pet sitting does not typically require heavy lifting or long-standing positions, which gives it a slightly negative score for Physical & Mental Burden (F3: -0.24).

**7 Swim Coach** (`7_swim_instructor`)

Swim Coach duties involve teaching and demonstrating skills, which contribute to skill acquisition (F1: +0.60). The pay is typically above lifeguard deck pay, which is higher than average for summer jobs (F2: +0.65). The work environment, while physically demanding, is less stressful than a lifeguard post (F3: -0.67), especially since the Swim Coach is not the primary surveillance. The job offers opportunities for networking with parents and children, enhancing social networking value (F1: +0.62).

