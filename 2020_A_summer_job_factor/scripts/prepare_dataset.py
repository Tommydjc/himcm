#!/usr/bin/env python3
"""Download or synthesize HiMCM 2020 A questionnaire assets and job corpora.

Attempts GitHub raw files first. If the network fails or the payload is not a
50 x 15 Likert matrix on scale 1-10 (and 50 job IDs in 0-7), writes an explicit
synthetic fallback so later EFA / LLMFactor stages have a complete local store.

Paths are relative to the 2020_A_summer_job_factor pack root.
"""

from __future__ import annotations

import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

PACK_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
RAW_DIR: Final[Path] = PACK_ROOT / "data" / "raw"
JOB_DIR: Final[Path] = PACK_ROOT / "data" / "job_descriptions"

GITHUB_BASE: Final[str] = "https://raw.githubusercontent.com/HiMCM2020/HiMCM_2020/main/"
ORIGINAL_URL: Final[str] = GITHUB_BASE + "OriginalData.txt"
CHOICE_URL: Final[str] = GITHUB_BASE + "work_choice.txt"
HTTP_TIMEOUT_S: Final[float] = 12.0
SYNTHETIC_SEED: Final[int] = 2020
N_STUDENTS: Final[int] = 50
N_FEATURES: Final[int] = 15
N_JOBS: Final[int] = 8
DOWNLOAD_USER_AGENT: Final[str] = "GoHiMCM-2020A-prepare-dataset/1.0"

FEATURE_COLUMNS: Final[list[str]] = [
    "hourly_wage_need",
    "tip_potential",
    "flexible_hours",
    "commute_convenience",
    "physical_strength",
    "outdoor_sun_exposure",
    "mental_stress",
    "safety_level",
    "resume_value",
    "skill_acquisition",
    "social_networking",
    "teamwork_atmosphere",
    "free_meals_perks",
    "boss_fairness",
    "autonomous_control",
]

JOB_FILES: Final[list[str]] = [
    "0_lifeguard.txt",
    "1_camp_counselor.txt",
    "2_private_tutor.txt",
    "3_fast_food_crew.txt",
    "4_retail_sales.txt",
    "5_golf_caddy.txt",
    "6_pet_sitter.txt",
    "7_swim_instructor.txt",
]

ANSI_RESET: Final[str] = "\033[0m"
ANSI_GREEN: Final[str] = "\033[32m"
ANSI_YELLOW: Final[str] = "\033[33m"
ANSI_RED: Final[str] = "\033[31m"
ANSI_CYAN: Final[str] = "\033[36m"
ANSI_BOLD: Final[str] = "\033[1m"

JOB_CORPORA: Final[dict[str, str]] = {
    "0_lifeguard.txt": """\
OCCUPATION: Lifeguard (BLS SOC 33-9092, Recreation Protective Service Workers)
SETTING: Outdoor municipal or community pool / lake beach, summer season (Memorial Day to Labor Day).

HOURLY PAY
Typical U.S. summer high-school / college lifeguard wage: $15.50 to $18.00 per hour before overtime.
Some municipal parks add a small certification differential ($0.50-$1.00) after American Red Cross
Lifeguarding / First Aid / CPR / AED and Waterfront modules. Overtime after 40 hours is uncommon
because shifts are split (morning 6:30-13:00, afternoon 12:30-19:00). Tips are rare; a few private
clubs pay a season-end bonus.

DUTIES
1. Continuous visual surveillance of assigned zone; enforce capacity and no-diving rules.
2. Prevent drowning and spinal injury: whistle, reach, throw, then in-water rescue if required.
3. Provide first aid for cuts, heat exhaustion, and chemical eye irritation from chlorinated water.
4. Test free chlorine, combined chlorine, and pH at opening and midday; log readings.
5. Clean decks, empty skimmer baskets, and restock first-aid kits.
6. Document incidents; radio the pool manager for EMS when indicated.

WORK ENVIRONMENT
High outdoor sun exposure for 6-8 hours per shift. Deck surface temperature often exceeds 40 C.
Glare off water requires polarized sunglasses. Noise from splash and PA systems. Chemical odor
from hypochlorite. Limited shade except during 15-minute rotations off the stand.

PHYSICAL LOAD
Must maintain 500-yard swim, brick retrieval, and 2-minute treading without hands as certification
prerequisites. Repeated climbs onto the stand, sprint entries, and spinal backboard lifts (with a
partner) of 40-90 kg. Heat stress is the dominant chronic load; acute load is rescue-related.

MENTAL / SAFETY
High responsibility: a missed 10-second scan can be fatal. Liability is shared with the facility
but personal certification can be revoked after a preventable incident. Weather cancellations
cut hours. COVID-era crowding rules may still appear in municipal SOPs.

RESUME VALUE
Strong signal for nursing, EMT, fire, kinesiology, and education applications because of documented
emergency-response hours. Weak signal for purely quantitative internships. List certifications
with expiry dates (typically 2 years).

SCHEDULING
Shifts posted weekly. Weekend and holiday coverage is required. Little remote work. Commute is
local (pool within a town). Season length is 10-12 weeks.

PERKS
Free pool use off-shift at many facilities. Rare free meals. Uniform (swimsuit + rash guard)
usually provided. Sunscreen sometimes reimbursed.
""",
    "1_camp_counselor.txt": """\
OCCUPATION: Summer Camp Counselor / Recreation Worker (BLS SOC 39-9032 Recreation Workers)
SETTING: Residential or day camp in a state park, farm, or campus, typically 8-10 consecutive weeks.

HOURLY PAY / COMPENSATION
Cash wage is often modest ($12-$16/hour equivalent) because room and board are included at
residential camps. Some agencies quote a season stipend ($2,800-$4,200) plus meals and a bunk.
Day camps pay hourly without lodging. Overtime is culturally expected rather than paid: staff
are on duty from wake-up (07:00) through lights-out (22:00) with a short afternoon break.

DUTIES
1. Supervise a cabin or age-group of 8-14 campers; know allergies, medications, and pickup rules.
2. Lead activities: hiking, canoeing, crafts, drama, sports, and evening campfire programs.
3. Enforce buddy systems, waterfront rules, and food-allergy tables.
4. Communicate with parents at drop-off and in weekly letters.
5. De-escalate homesickness and peer conflict; escalate to the camp director when needed.
6. Complete incident logs and end-of-session evaluations.

WORK ENVIRONMENT
Mix of outdoor trails, dining hall, cabins without air conditioning, and lakes. High interpersonal
density: counselors eat, sleep, and work with the same group. Rain days move programs indoors.

PHYSICAL LOAD
Long standing and walking on uneven ground. Lifting canoes, coolers, and luggage. Sleep deficit
is common. Heat and mosquitoes in July. Not a heavy lifting job compared with landscaping, but
endurance (14-16 hour days) is high.

MENTAL / SAFETY
Strong interpersonal and leadership demand. Background checks and Mandated Reporter training
are standard. Waterfront and high-ropes require extra certifications. Homesickness and bullying
create emotional labor. Physical assault risk is low if ratios and night checks are followed.

RESUME VALUE
High for education, social work, outdoor recreation, and any role that screens for leadership
and teamwork. Document camper-to-staff ratio, certifications (Wilderness First Aid, lifeguard),
and a measurable outcome (retention, parent satisfaction). Weaker for finance internships unless
framed as operations / logistics.

SCHEDULING
Residential: almost no free evenings. One 24-hour leave per two weeks is typical. Day camp:
08:00-16:30 weekdays, some Friday overnights.

PERKS
Room and board at residential sites (large in-kind value). Staff night, laundry, and camp T-shirts.
Free meals are a core perk. Little cash tip income.
""",
    "2_private_tutor.txt": """\
OCCUPATION: Private / Learning-Center Tutor (BLS SOC 25-3041 Tutors)
SETTING: Student home, library, or air-conditioned tutoring center; some sessions by video.

HOURLY PAY
Highest cash wage among the eight summer options: $22 to $35 per hour for high-school STEM or
SAT/ACT prep in U.S. metro areas. Center employees may start at $18-$22; independent tutors
who set their own rate capture the upper end after agency fees. No overtime culture; sessions
are 60-90 minutes. Tips are uncommon; families pay invoices.

DUTIES
1. Diagnose gaps with a short diagnostic or recent test scores.
2. Plan sessions aligned to school syllabi or College Board domains.
3. Explain worked examples, assign targeted practice, and review errors.
4. Report progress to parents weekly (email or portal).
5. Maintain academic integrity: no completing graded homework for the student.
6. For center roles: follow scripted curricula and log attendance.

WORK ENVIRONMENT
Indoor, climate-controlled, low noise compared with kitchens or pools. Lighting and chairs vary.
Commute can be multi-stop if you travel between houses; centers have a single site.

PHYSICAL LOAD
Sedentary. Minimal sun exposure. Wrist and neck strain from laptops if hours are stacked.
No heavy lifting. Energy cost is cognitive, not metabolic.

MENTAL / SAFETY
High academic bar: families expect subject mastery one to two years above the student's course.
Performance anxiety when a student has a midterm the next day. Background checks and, in some
states, fingerprinting. Home sessions require clear parent-present policies for minors.

RESUME VALUE
Strong for teaching, college applications in STEM, and any role that values communication of
quantitative ideas. List subjects, grade bands, and measurable score changes. Weaker as evidence
of physical teamwork. Independent tutoring also signals small-business billing skills.

SCHEDULING
Highest schedule flexibility among indoor jobs, but demand clusters 16:00-21:00 and weekends.
Summer intensive camps (8:30-12:30) exist at centers. Cancellation clauses (24-hour) protect
income.

PERKS
No free meals as a rule. Occasional family-provided snacks. No employee discount. Equipment is
your laptop and a whiteboard. Commute reimbursement is rare.
""",
    "3_fast_food_crew.txt": """\
OCCUPATION: Fast Food Crew Member (BLS SOC 35-3023 Fast Food and Counter Workers)
SETTING: Quick-service restaurant (QSR) kitchen and front counter; high-volume lunch and dinner.

HOURLY PAY
Lowest entry wage of the eight options, often at or just above state minimum wage (illustrative
summer range $12-$15/hour in many U.S. states; some metro QSRs $15-$17). Overtime after 40 hours
is paid at 1.5x when scheduled. Counter tips are small unless the store uses a tip jar. Crew
meals are a material in-kind benefit.

DUTIES
1. Take orders at register or drive-through headset; handle cash and card.
2. Assemble sandwiches, fry baskets, and drinks to board times (typically under 90-180 seconds).
3. Hold food at safe temperatures; discard product past hold timers (food safety).
4. Clean fryers, floors, restrooms, and lobby on a closing checklist.
5. Restock cups, lids, sauces; rotate FIFO in the walk-in cooler.
6. Follow allergy and waste logs.

WORK ENVIRONMENT
Hot kitchen: fryer and grill radiant heat; indoor temperatures often 28-35 C at peak. Grease
aerosol, loud headsets, and customer queue pressure. Air conditioning in the lobby does not
fully cool the cookline. Uniform and non-slip shoes required.

PHYSICAL LOAD
Prolonged standing, repetitive reaching, lifting 15-20 kg fryer oil or frozen cases, and wet
floors (slip risk). Burns from oil and steam are the main acute injuries. Outdoor sun is low
except for drive-through window and lot trash runs.

MENTAL / SAFETY
Low hiring barrier: on-the-job training in days. Customer conflict at the window. OSHA hot-oil
and knife rules. Late-night shifts raise robbery awareness at some sites; high-school workers
are usually scheduled off late graveyard hours.

RESUME VALUE
Low prestige on a college STEM resume if listed only as "crew." Still valid evidence of
punctuality, cash handling, and food-safety certification (ServSafe Food Handler). Frame as
operations under time pressure if applying to industrial engineering or hospitality.

SCHEDULING
Manager-built weekly roster. Weekends and nights are high demand. Shift swaps via app. Little
autonomy over hours.

PERKS
Employee meal (often 50% off or one free meal per shift) is the primary perk. Free fountain
drinks. Occasional free product. No high-status networking.
""",
    "4_retail_sales.txt": """\
OCCUPATION: Retail Salesperson (BLS SOC 41-2031)
SETTING: Enclosed shopping mall or big-box store; climate-controlled sales floor.

HOURLY PAY
Typical teen / young-adult apparel or electronics sales: $13-$17/hour plus modest commission
in some specialty stores (1-3% of personal sales) or a team bonus. Mall kiosks may pay a draw
against commission. Overtime is seasonal (back-to-school, not always summer). Tips are not
standard in U.S. retail.

DUTIES
1. Greet customers, size product, and complete POS transactions.
2. Maintain visual merchandising: folding, steaming, mannequin dressing, recovery after rushes.
3. Process returns and loyalty accounts; follow loss-prevention bag-check rules.
4. Stock from the backroom; RFID or barcode cycle counts.
5. Open/close: tills, fitting-room checks, alarm codes with a keyholder.
6. Hit conversion and units-per-transaction goals where the retailer tracks them.

WORK ENVIRONMENT
Air-conditioned mall ambient (typically 21-24 C). Fluorescent or LED lighting. Standing on
hard floors. Music and crowd noise on weekends. Fitting rooms require two-way visibility
policies. Outdoor sun exposure is limited to parking-lot shifts.

PHYSICAL LOAD
Long standing (6-8 hour shifts) with limited sitting. Lifting shipment cartons 5-15 kg.
Repetitive folding. Lower back fatigue is common; heat load is low compared with kitchens
and pool decks.

MENTAL / SAFETY
Customer service scripts and mystery-shopper scores. Theft and organized retail crime require
observation without confrontation (call asset protection). Fitting-room incidents need a
manager. Hiring barrier is moderate: availability plus a short interview.

RESUME VALUE
Moderate: sales metrics, cash handling, and merchandising. Useful for business, marketing,
and communications applications. Weaker than tutoring for academic signaling and weaker than
lifeguarding for emergency-response signaling.

SCHEDULING
Availability matrix; weekends required at malls. Call-outs filled by group text. Hours can
be cut if traffic is low. Commute to the mall is often longer than a neighborhood pool.

PERKS
Employee discount (typically 20-50% on house brand) is the headline perk. Rare free meals.
Climate comfort is a quality-of-life perk relative to outdoor labor.
""",
    "5_golf_caddy.txt": """\
OCCUPATION: Golf Caddie (related BLS groups: 39-3091 Amusement and Recreation Attendants;
39-9031 Fitness Trainers and related outdoor attendants at private clubs)
SETTING: 18-hole private or resort golf course; walking 5-7 km per loop with a bag.

HOURLY PAY / CASH
Base club wage may be low or unpaid "independent contractor" loops. Total cash is dominated
by caddie fees plus tips. A single loop often yields $80-$150 cash (fee + gratuity) for 4-5
hours, which annualizes to a high effective hourly rate on busy summer weekends. Rain-outs
zero the day. No W-2 overtime in many clubs; 1099 tax reporting is common.

DUTIES
1. Carry or push a 10-15 kg tour bag; keep clubs clean and ordered.
2. Read yardage, wind, and green breaks; hand the requested club.
3. Rake bunkers, fix ball marks, tend the flag, and locate errant balls.
4. Track score if asked; remain quiet during the stroke.
5. Know local rules, cart-path-only days, and lightning evacuation.
6. Caddie master assigns loops; punctuality at the barn is mandatory.

WORK ENVIRONMENT
Full outdoor sun, 4-5 hours walking per loop, possibly two loops on Saturday. Terrain includes
hills, wet rough, and sand. Dress code: caddie bib, soft spikes, hat. Early tee times (06:30)
avoid peak heat; afternoon loops are hotter.

PHYSICAL LOAD
Highest sustained walking load of the eight jobs. Shoulder and lumbar load from the bag.
Hydration is critical. Acute injury: errant balls, lightning, sprains. Not a gym-max strength
job, but a long aerobic day.

MENTAL / SAFETY
Etiquette pressure around members. Low formal academic bar; high social calibration. Heat
illness risk. Lightning policy: leave the course when the horn sounds.

RESUME VALUE
Distinctive for networking: members are often high-net-worth professionals. A discreet,
reliable caddie can receive internship referrals. On paper, list customer service, local-knowledge
expertise, and cash-handling of tips. Weaker as a STEM skill signal unless paired with
course-management notes or GIS yardage work.

SCHEDULING
Loop assignments by the caddie master; show-up days without a loop earn nothing. Highest
variance income of the outdoor set. Weekends are peak.

PERKS
Cash tips. Occasional player-bought lunch at the turn. Clubhouse meal discounts at some
clubs. Course access for staff golf on Monday. No free housing.
""",
    "6_pet_sitter.txt": """\
OCCUPATION: Pet Sitter / Dog Walker (BLS SOC 39-2021 Animal Caretakers)
SETTING: Client homes, neighborhood sidewalks, and occasionally a small boarding facility.

HOURLY PAY
Drop-in visits often priced per visit ($20-$30 for 30 minutes) rather than a true hourly W-2.
Dog walks $18-$25 per 30-45 minutes in many U.S. suburbs. Overnight house-sitting $60-$100
per night. Income is lumpy: full when families travel, idle in between. Apps take 20-40%
commission. Tips appear after holiday travel.

DUTIES
1. Feed, medicate, and log intake/output per the client's written protocol.
2. Walk dogs on leash; manage pullers and recall in fenced yards only when authorized.
3. Scoop litter, clean crates, and take out household trash if contracted.
4. Photograph updates and send a same-day report in the app or text.
5. Secure doors, alarms, and mail holds for house-sits.
6. Know emergency vet address and authorization to treat.

WORK ENVIRONMENT
Highly variable indoor climate (client HVAC). Outdoor walks in heat require shortened duration
and paw protection. Low coworker density: you work alone. Travel between homes by bike or car
is a hidden time cost.

PHYSICAL LOAD
Light-to-moderate: walking 3-8 km/day if stacked, lifting 15-40 kg dogs into cars only if
trained. Kneeling for litter boxes. Sun exposure depends on walk density. Much lower heat
and standing load than QSR or retail.

MENTAL / SAFETY
Lowest chronic mental stress of the eight jobs if clients are screened: animals, not crowds.
Risks: bites, dog-dog incidents, unknown people at the property, and liability if a pet
escapes. Insurance (caregiver liability) is recommended. Background checks via platforms.

RESUME VALUE
Signals reliability and independent scheduling. Weak academic signal. Stronger if you document
pet first-aid (PetTech) or run a registered micro-business (invoicing, reviews). Not a
teamwork story unless you staff a boarding kennel.

SCHEDULING
Maximum flexibility: you accept or decline calendar slots. Last-minute holiday demand.
No guaranteed hours. Commute is a star-shaped tour of addresses.

PERKS
Quiet houses, flexible start times, and sometimes permission to use the kitchen. Rare free
meals as a contract term. No boss on site; fairness is encoded in the client agreement
rather than a shift supervisor.
""",
    "7_swim_instructor.txt": """\
OCCUPATION: Swim Instructor (BLS SOC 25-3021 Self-Enrichment Teachers; water-safety instruction)
SETTING: Indoor or outdoor instructional pool; learn-to-swim and stroke clinics.

HOURLY PAY
Typically above lifeguard deck pay: $18-$28 per hour for group lessons; private 30-minute
slots can bill $40-$60 of which the instructor may keep $22-$35 after the facility cut.
Summer lesson blocks are dense (08:00-12:00 parent-and-child, 16:00-19:00 school-age).
Tips after a session series are occasional, not a wage base.

DUTIES
1. Teach water acclimation, breath control, floating, and competitive strokes by level
   (e.g., American Red Cross Learn-to-Swim Stages 1-6).
2. Maintain class ratios (often 1:4 to 1:8 depending on age and deep water).
3. Demonstrate skills in water; use kickboards, noodles, and dive rings.
4. Record skills checklists; conference with parents at the wall.
5. Coordinate with lifeguards; you are not the primary surveillance but you still scan.
6. Hold Water Safety Instructor (WSI) or equivalent plus CPR/AED.

WORK ENVIRONMENT
Pool deck humidity, chlorine, and echo. Outdoor lessons add sun; many municipal programs
still use outdoor 50 m or 25 m pools in summer. Indoor natatoriums are hot and wet but
shade-protected. Less continuous scanning stress than a lifeguard stand, more pedagogical
talking.

PHYSICAL LOAD
In-water demonstration, supporting toddlers at the wall, and treading during classes.
Shoulder load from showing strokes. Heat plus humidity. Lower emergency-sprint demand
than a dedicated guard, but you may still perform a rescue.

MENTAL / SAFETY
Teaching efficacy and anxious parents. Fearful children require patience, not force.
Certification and background checks. Spinal-injury protocols still apply in shallow
teaching areas.

RESUME VALUE
High for education, coaching, kinesiology, and college athletic recruiting supplements.
Document level progressions and WSI number. Complements lifeguard credentials. Weaker
for purely computational internships unless framed as curriculum design.

SCHEDULING
Lesson grids are fixed 8-week sessions; missed classes are sometimes made up. More
predictable than caddying or pet sitting. Early mornings and late afternoons dominate.

PERKS
Pool use, staff discount on family lessons, and strong seasonal demand. Meals not included.
Certification courses sometimes reimbursed if you stay the season.
""",
}


class DatasetPrepError(RuntimeError):
    """Raised when an asset cannot be materialized or fails a contract check."""


def _color(code: str, text: str) -> str:
    """Wrap ``text`` in ANSI color if stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{ANSI_RESET}"


def ensure_directories() -> None:
    """Create ``data/raw/`` and ``data/job_descriptions/`` under the pack root."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)


def download_text(url: str) -> str:
    """Fetch UTF-8 text from ``url``.

    Parameters
    ----------
    url:
        Absolute HTTP(S) URL.

    Returns
    -------
    str
        Decoded body.

    Raises
    ------
    DatasetPrepError
        On HTTP, SSL, DNS, or timeout failures.
    """
    request = urllib.request.Request(
        url,
        headers={"User-Agent": DOWNLOAD_USER_AGENT},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            status = int(getattr(response, "status", 200))
            if status >= 400:
                raise DatasetPrepError(f"HTTP {status} for {url}")
            raw_bytes = response.read()
    except DatasetPrepError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise DatasetPrepError(f"download failed for {url}: {exc}") from exc
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetPrepError(f"non-UTF8 payload at {url}") from exc


def parse_numeric_table(text: str) -> np.ndarray:
    """Parse whitespace or comma separated numbers into a 2-D float array.

    Parameters
    ----------
    text:
        File body. Comment lines starting with ``#`` are ignored.

    Returns
    -------
    np.ndarray
        Shape ``(n_rows, n_cols)``, dtype float64.
    """
    rows: list[list[float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        stripped = stripped.replace(",", " ")
        parts = stripped.split()
        try:
            values = [float(token) for token in parts]
        except ValueError as exc:
            raise DatasetPrepError(f"non-numeric token in line: {stripped[:80]}") from exc
        if values:
            rows.append(values)
    if not rows:
        raise DatasetPrepError("empty numeric table")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise DatasetPrepError("ragged numeric table")
    return np.asarray(rows, dtype=np.float64)


def validate_scores(matrix: np.ndarray) -> np.ndarray:
    """Accept only a 50 x 15 matrix with values in [1, 10].

    Returns integer scores in ``{1,...,10}``.
    """
    if matrix.ndim != 2 or matrix.shape != (N_STUDENTS, N_FEATURES):
        raise DatasetPrepError(
            f"OriginalData shape {matrix.shape} != {(N_STUDENTS, N_FEATURES)}"
        )
    if np.any(~np.isfinite(matrix)):
        raise DatasetPrepError("OriginalData contains non-finite values")
    if matrix.min() < 1.0 - 1e-9 or matrix.max() > 10.0 + 1e-9:
        raise DatasetPrepError(
            f"OriginalData range [{matrix.min()}, {matrix.max()}] outside [1, 10]"
        )
    rounded = np.rint(matrix).astype(np.int64)
    if np.any(rounded < 1) or np.any(rounded > 10):
        raise DatasetPrepError("OriginalData rounding left the 1-10 Likert scale")
    return rounded


def validate_choices(matrix: np.ndarray) -> np.ndarray:
    """Accept 50 job IDs in ``{0,...,7}``, possibly as a column vector."""
    flat = np.rint(matrix.reshape(-1)).astype(np.int64)
    if flat.size != N_STUDENTS:
        raise DatasetPrepError(f"work_choice length {flat.size} != {N_STUDENTS}")
    if np.any(flat < 0) or np.any(flat >= N_JOBS):
        raise DatasetPrepError("work_choice IDs must lie in 0..7")
    return flat


def synthesize_scores(rng: np.random.Generator) -> np.ndarray:
    """Build a 50 x 15 integer Likert matrix with column means in [5, 8].

    Each column is a clipped Normal draw around a column-specific mean sampled
    uniformly from [5, 8], plus a light student-level offset so rows are not IID.
    """
    column_means = rng.uniform(5.0, 8.0, size=N_FEATURES)
    student_offset = rng.normal(0.0, 0.45, size=(N_STUDENTS, 1))
    noise = rng.normal(0.0, 1.35, size=(N_STUDENTS, N_FEATURES))
    continuous = column_means.reshape(1, -1) + student_offset + noise
    scores = np.clip(np.rint(continuous), 1, 10).astype(np.int64)
    for j in range(N_FEATURES):
        mean_j = float(scores[:, j].mean())
        if 5.0 <= mean_j <= 8.0:
            continue
        target = float(np.clip(column_means[j], 5.5, 7.5))
        shift = int(np.rint(target - mean_j))
        scores[:, j] = np.clip(scores[:, j] + shift, 1, 10)
    return scores


def synthesize_choices(rng: np.random.Generator, scores: np.ndarray) -> np.ndarray:
    """Sample 50 job IDs in 0..7 with a mild preference link to wage / outdoor scores."""
    logits = np.zeros((N_STUDENTS, N_JOBS), dtype=np.float64)
    wage = scores[:, 0].astype(np.float64)
    outdoor = scores[:, 5].astype(np.float64)
    social = scores[:, 10].astype(np.float64)
    physical = scores[:, 4].astype(np.float64)
    logits[:, 0] += 0.15 * outdoor + 0.05 * physical
    logits[:, 1] += 0.18 * social
    logits[:, 2] += 0.20 * wage - 0.10 * physical
    logits[:, 3] += 0.04 * (10.0 - wage)
    logits[:, 4] += 0.08 * social
    logits[:, 5] += 0.12 * outdoor + 0.10 * wage
    logits[:, 6] += 0.16 * (10.0 - outdoor)
    logits[:, 7] += 0.10 * outdoor + 0.08 * social
    logits += rng.gumbel(size=logits.shape)
    return np.argmax(logits, axis=1).astype(np.int64)


def try_remote_tables() -> tuple[np.ndarray, np.ndarray, str]:
    """Download and validate remote files.

    Returns
    -------
    scores, choices, note
    """
    original_text = download_text(ORIGINAL_URL)
    choice_text = download_text(CHOICE_URL)
    scores = validate_scores(parse_numeric_table(original_text))
    choices = validate_choices(parse_numeric_table(choice_text))
    note = f"remote OK: {ORIGINAL_URL} and {CHOICE_URL}"
    return scores, choices, note


def write_scores_csv(scores: np.ndarray) -> Path:
    """Write ``data/raw/OriginalData.csv`` with semantic headers."""
    path = RAW_DIR / "OriginalData.csv"
    frame = pd.DataFrame(scores, columns=FEATURE_COLUMNS)
    frame.to_csv(path, index=False)
    return path


def write_choice_csv(choices: np.ndarray) -> Path:
    """Write ``data/raw/work_choice.csv`` with student_id and job_choice_id."""
    path = RAW_DIR / "work_choice.csv"
    frame = pd.DataFrame(
        {
            "student_id": np.arange(N_STUDENTS, dtype=np.int64),
            "job_choice_id": choices.astype(np.int64),
        }
    )
    frame.to_csv(path, index=False)
    return path


def write_source_note(note: str, used_fallback: bool) -> Path:
    """Record whether the Likert table is remote or synthetic."""
    path = RAW_DIR / "DATA_SOURCE.txt"
    status = "SYNTHETIC_FALLBACK" if used_fallback else "REMOTE_GITHUB"
    body = (
        f"status={status}\n"
        f"n_students={N_STUDENTS}\n"
        f"n_features={N_FEATURES}\n"
        f"likert_scale=1-10\n"
        f"synthetic_seed={SYNTHETIC_SEED}\n"
        f"detail={note}\n"
        "Do not treat SYNTHETIC_FALLBACK rows as a fielded HiMCM contest questionnaire.\n"
    )
    path.write_text(body, encoding="utf-8")
    return path


def write_job_descriptions() -> list[Path]:
    """Write eight BLS-style plain-text occupation files."""
    written: list[Path] = []
    for filename in JOB_FILES:
        text = JOB_CORPORA[filename].strip() + "\n"
        path = JOB_DIR / filename
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def assert_contracts() -> None:
    """Fail loudly if on-disk assets violate the EFA / LLMFactor contract."""
    original_path = RAW_DIR / "OriginalData.csv"
    choice_path = RAW_DIR / "work_choice.csv"
    scores = pd.read_csv(original_path)
    if scores.shape != (N_STUDENTS, N_FEATURES):
        raise AssertionError(f"OriginalData.csv shape {scores.shape} != {(N_STUDENTS, N_FEATURES)}")
    if list(scores.columns) != FEATURE_COLUMNS:
        raise AssertionError("OriginalData.csv columns do not match FEATURE_COLUMNS")
    choices = pd.read_csv(choice_path)
    if len(choices) != N_STUDENTS:
        raise AssertionError(f"work_choice.csv rows {len(choices)} != {N_STUDENTS}")
    if list(choices.columns) != ["student_id", "job_choice_id"]:
        raise AssertionError("work_choice.csv columns mismatch")
    txt_files = sorted(JOB_DIR.glob("*.txt"))
    if len(txt_files) != N_JOBS:
        raise AssertionError(f"expected {N_JOBS} job .txt files, found {len(txt_files)}")
    for path in txt_files:
        size = path.stat().st_size
        if size <= 100:
            raise AssertionError(f"{path.name} size {size} B <= 100 B")


def print_report(
    used_fallback: bool,
    note: str,
    scores: np.ndarray,
    job_paths: list[Path],
) -> None:
    """Print a colorized summary of written assets."""
    title = _color(ANSI_BOLD + ANSI_CYAN, "2020 A dataset prepare — summary")
    print(title)
    source_label = "SYNTHETIC FALLBACK" if used_fallback else "REMOTE DOWNLOAD"
    source_color = ANSI_YELLOW if used_fallback else ANSI_GREEN
    print(_color(source_color, f"  source: {source_label}"))
    print(f"  note:   {note}")
    print(f"  OriginalData.csv: {RAW_DIR / 'OriginalData.csv'}")
    print(f"    shape={scores.shape}  col_means={np.round(scores.mean(axis=0), 2).tolist()}")
    print(f"  work_choice.csv:  {RAW_DIR / 'work_choice.csv'}")
    print(f"  DATA_SOURCE.txt:  {RAW_DIR / 'DATA_SOURCE.txt'}")
    print(f"  job_descriptions: {JOB_DIR}  ({len(job_paths)} files)")
    for path in job_paths:
        print(f"    {path.name:24s}  {path.stat().st_size:5d} B")
    print(_color(ANSI_GREEN, "  assertions: PASSED"))


def main() -> int:
    """Create directories, materialize tables and corpora, then assert contracts."""
    ensure_directories()
    used_fallback = False
    note = ""
    try:
        scores, choices, note = try_remote_tables()
    except DatasetPrepError as exc:
        used_fallback = True
        note = f"fallback after remote/schema failure: {exc}"
        rng = np.random.default_rng(SYNTHETIC_SEED)
        scores = synthesize_scores(rng)
        choices = synthesize_choices(rng, scores)
    except Exception as exc:  # noqa: BLE001 — last-resort network/parser shield
        used_fallback = True
        note = f"fallback after unexpected error: {type(exc).__name__}: {exc}"
        traceback.print_exc()
        rng = np.random.default_rng(SYNTHETIC_SEED)
        scores = synthesize_scores(rng)
        choices = synthesize_choices(rng, scores)

    write_scores_csv(scores)
    write_choice_csv(choices)
    write_source_note(note, used_fallback)
    job_paths = write_job_descriptions()
    assert_contracts()
    print_report(used_fallback, note, scores, job_paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
