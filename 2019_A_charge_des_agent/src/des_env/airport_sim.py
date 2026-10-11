"""
SimPy airport charging DES sandbox.

Physical layer for the BDI–DES mapping in arXiv:2602.13878:
cognitive intention updates stay in ``BDIPassenger``; this module owns
``simpy.Environment`` clocks, resource holds, and reneging/boarding timeouts.

Horizon: one operational day 06:00–22:00 (960 minutes).
Arrivals: non-homogeneous Poisson with 3× peaks at 08:00, 13:00, 18:00.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, Iterator, List, Optional, Tuple

import numpy as np
import simpy

# Allow ``from bdi_agents...`` when running as a script / pytest with PYTHONPATH=src.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bdi_agents.passenger_bdi import (
    BATTERY_ANXIETY_THRESHOLD,
    BOARDING_BUFFER_MIN,
    BDIPassenger,
    IntentionState,
    PassengerBeliefs,
    sample_battery_levels,
    sample_boarding_countdowns,
    sample_price_sensitivity,
)

# ---------------------------------------------------------------------------
# Physical constants (minutes unless noted)
# ---------------------------------------------------------------------------

HORIZON_MIN: Final[float] = 16.0 * 60.0  # 06:00 → 22:00
PEAK_TIMES_MIN: Final[Tuple[float, ...]] = (120.0, 420.0, 720.0)  # 08/13/18
PEAK_SIGMA_MIN: Final[float] = 30.0
MEAN_CHARGE_FAST_MIN: Final[float] = 25.0
MEAN_CHARGE_SLOW_MIN: Final[float] = 55.0
POWERBANK_TX_MIN: Final[float] = 1.0
POWERBANKS_PER_KIOSK: Final[int] = 48
BASE_POWERBANK_PRICE: Final[float] = 20.0
DEFAULT_LAMBDA_BASE: Final[float] = 0.35  # passengers / minute off-peak
QUEUE_SAMPLE_DT: Final[float] = 1.0


@dataclass
class AirportResources:
    """Multi-pool physical resources visible to passenger processes."""

    fast_chargers: simpy.Resource
    slow_lockers: simpy.Resource
    powerbank_kiosk: simpy.Resource
    powerbank_stock: List[int]  # length-1 mutable cell: remaining banks


@dataclass
class Telemetry:
    """Accumulators for run-level KPIs (written only by DES processes)."""

    n_arrivals: int = 0
    n_anxious: int = 0
    n_seekers: int = 0
    n_reneged: int = 0
    n_abort: int = 0
    n_served_fast: int = 0
    n_served_slow: int = 0
    total_powerbank_rentals: int = 0
    n_storm_delayed: int = 0
    # BDI intention flow counters (for Sankey diagrams).
    flow_calm: int = 0
    flow_to_fast: int = 0
    flow_to_slow: int = 0
    flow_to_powerbank: int = 0
    flow_to_abort: int = 0
    wait_times: List[float] = field(default_factory=list)
    max_queue_len: int = 0
    max_system_load: int = 0
    busy_fast_min: float = 0.0
    busy_slow_min: float = 0.0
    c_fast: int = 0
    c_slow: int = 0
    c_bank: int = 0
    lambda_base: float = DEFAULT_LAMBDA_BASE
    peak_multiplier: float = 3.0
    arrivals_per_min: List[int] = field(default_factory=list)
    queue_timeline: List[Dict[str, float]] = field(default_factory=list)

    def note_arrival(self, time_minutes: float, horizon_min: float) -> None:
        """Increment the per-minute arrival histogram (bin = floor(t))."""
        idx = int(time_minutes)
        if idx < 0:
            raise RuntimeError(f"negative arrival time: {time_minutes}")
        need = idx + 1
        if len(self.arrivals_per_min) < need:
            self.arrivals_per_min.extend([0] * (need - len(self.arrivals_per_min)))
        # Cap index at horizon for late stragglers (should not happen).
        if idx >= int(horizon_min) + 1:
            idx = int(horizon_min)
            while len(self.arrivals_per_min) <= idx:
                self.arrivals_per_min.append(0)
        self.arrivals_per_min[idx] += 1

    def note_queue(
        self,
        fast: simpy.Resource,
        slow: simpy.Resource,
        *,
        time_minutes: float,
    ) -> None:
        """
        Sample waiting line, in-service headcount, and NHPP intensity.

        ``queue_length`` = waiting only (SimPy ``Resource.queue``).
        ``in_service`` = users holding chargers.
        ``system_load`` = waiting + in-service (correct congestion series for plots).
        """
        q_fast = len(fast.queue)
        q_slow = len(slow.queue)
        u_fast = len(fast.users)
        u_slow = len(slow.users)
        if min(q_fast, q_slow, u_fast, u_slow) < 0:
            raise RuntimeError("negative queue/user count violates DES invariant")
        waiting = q_fast + q_slow
        in_service = u_fast + u_slow
        system_load = waiting + in_service
        if waiting > self.max_queue_len:
            self.max_queue_len = waiting
        if system_load > self.max_system_load:
            self.max_system_load = system_load
        bin_idx = int(time_minutes)
        arrivals_bin = (
            float(self.arrivals_per_min[bin_idx])
            if 0 <= bin_idx < len(self.arrivals_per_min)
            else 0.0
        )
        lam_t = arrival_intensity(
            float(time_minutes),
            self.lambda_base,
            peak_multiplier=self.peak_multiplier,
        )
        self.queue_timeline.append(
            {
                "time_minutes": float(time_minutes),
                "queue_length": float(waiting),
                "queue_fast": float(q_fast),
                "queue_slow": float(q_slow),
                "in_service": float(in_service),
                "in_service_fast": float(u_fast),
                "in_service_slow": float(u_slow),
                "system_load": float(system_load),
                "arrivals_bin": arrivals_bin,
                "lambda_t": float(lam_t),
            }
        )

def arrival_intensity(
    t_min: float,
    lambda_base: float = DEFAULT_LAMBDA_BASE,
    *,
    peak_times: Tuple[float, ...] = PEAK_TIMES_MIN,
    sigma: float = PEAK_SIGMA_MIN,
    peak_multiplier: float = 3.0,
) -> float:
    """
    Non-homogeneous Poisson intensity λ(t) [passengers / minute].

    Parameters
    ----------
    t_min :
        Minutes since 06:00. Shape: scalar.
    lambda_base :
        Off-peak rate λ₀.
    peak_times :
        Centres of boarding-wave pulses (minutes since 06:00).
    sigma :
        Gaussian pulse width (minutes).
    peak_multiplier :
        Peak / off-peak ratio. At a pulse centre,
        ``λ = λ₀ * (1 + (m-1) * exp(0)) = m λ₀``.

    Returns
    -------
    float
        Instantaneous arrival rate (≥ 0).
    """
    if lambda_base < 0.0:
        raise ValueError(f"lambda_base must be non-negative: {lambda_base}")
    if peak_multiplier < 1.0:
        raise ValueError(f"peak_multiplier must be >= 1: {peak_multiplier}")
    bumps = 0.0
    for peak in peak_times:
        bumps += math.exp(-(((t_min - peak) / sigma) ** 2))
    return float(lambda_base * (1.0 + (peak_multiplier - 1.0) * bumps))


def _nhpp_arrival_times(
    horizon: float,
    lambda_base: float,
    rng: np.random.Generator,
    *,
    peak_multiplier: float = 3.0,
    peak_times: Tuple[float, ...] = PEAK_TIMES_MIN,
) -> np.ndarray:
    """
    Sample NHPP arrival epochs on ``[0, horizon)`` via thinning.

    Returns
    -------
    np.ndarray
        Sorted arrival times (minutes), dtype float64.
    """
    lam_max = float(peak_multiplier) * lambda_base
    if lam_max <= 0.0:
        return np.zeros(0, dtype=np.float64)
    times: List[float] = []
    t = 0.0
    while True:
        t += float(rng.exponential(1.0 / lam_max))
        if t >= horizon:
            break
        lam_t = arrival_intensity(
            t,
            lambda_base,
            peak_multiplier=peak_multiplier,
            peak_times=peak_times,
        )
        if rng.uniform() < lam_t / lam_max:
            times.append(t)
    return np.asarray(times, dtype=np.float64)


def _queue_len(resource: simpy.Resource) -> int:
    """Waiting-line length (excludes users in service)."""
    n = len(resource.queue)
    if n < 0:
        raise RuntimeError(f"negative queue length: {n}")
    return n


def _observed_charger_queue(resources: AirportResources) -> int:
    """Aggregate charger queue length shown to the BDI agent."""
    return _queue_len(resources.fast_chargers) + _queue_len(resources.slow_lockers)


def _dynamic_patience_min(passenger: BDIPassenger, env_now: float, deadline: float) -> float:
    """
    BDI tolerance: remaining time until boarding buffer elapses.

    ``patience = max(0, deadline - BOARDING_BUFFER_MIN - env.now)`` in minutes.
    """
    return max(0.0, deadline - BOARDING_BUFFER_MIN - env_now)


def _choose_charger(
    passenger: BDIPassenger,
    resources: AirportResources,
) -> Tuple[simpy.Resource, float, str]:
    """
    Select fast vs slow pool from remaining boarding slack.

    Returns
    -------
    (resource, mean_charge_min, label)
    """
    slack = passenger.beliefs.time_to_boarding - BOARDING_BUFFER_MIN
    prefer_fast = slack < MEAN_CHARGE_SLOW_MIN or (
        passenger.beliefs.battery_level < BATTERY_ANXIETY_THRESHOLD
        and slack < MEAN_CHARGE_FAST_MIN + 20.0
    )
    if prefer_fast:
        return resources.fast_chargers, MEAN_CHARGE_FAST_MIN, "fast"
    return resources.slow_lockers, MEAN_CHARGE_SLOW_MIN, "slow"


def _sample_service_time(mean_min: float, rng: np.random.Generator) -> float:
    """Exp service with given mean; floor at 1 minute for DES stability."""
    return max(1.0, float(rng.exponential(mean_min)))


def passenger_process(
    env: simpy.Environment,
    passenger: BDIPassenger,
    resources: AirportResources,
    telemetry: Telemetry,
    *,
    powerbank_price: float,
    rng: np.random.Generator,
) -> Iterator[simpy.Event]:
    """
    One BDI passenger DES trajectory (arrive → deliberate → seize / renege).

    Parameters
    ----------
    env :
        SimPy environment (monotone clock).
    passenger :
        Cognitive agent; intention updates only via ``evaluate_intentions``.
    resources :
        Physical multi-pool resources.
    telemetry :
        Shared KPI accumulators.
    powerbank_price :
        Posted rental price (already scaled by ``pricing_mode``).
    rng :
        Local RNG for service draws.

    Yields
    ------
    simpy.Event
        Timeouts and resource requests.
    """
    arrival_time = float(env.now)
    # Absolute boarding deadline on the DES clock.
    deadline = arrival_time + float(passenger.beliefs.time_to_boarding)

    # --- Cognitive cycle (instantaneous w.r.t. DES; ACLP-style mapping) ---
    q_obs = _observed_charger_queue(resources)
    intention = passenger.evaluate_intentions(q_obs, powerbank_price)
    if intention == IntentionState.IDLE:
        # High SOC: leave without charging.
        telemetry.flow_calm += 1
        return

    telemetry.n_seekers += 1

    # Drive SEEK_POWER → QUEUE / RENT / ABORT in the same cognitive tick burst.
    # At most a few cycles; DES time does not advance here (coarse ACLP grain).
    for _ in range(4):
        if intention in (
            IntentionState.QUEUE_AT_KIOSK,
            IntentionState.RENT_POWERBANK,
            IntentionState.ABORT_CHARGING,
        ):
            break
        intention = passenger.evaluate_intentions(
            _observed_charger_queue(resources),
            powerbank_price,
        )

    if intention == IntentionState.ABORT_CHARGING:
        telemetry.n_abort += 1
        telemetry.flow_to_abort += 1
        if passenger.did_renege():
            telemetry.n_reneged += 1
        return

    if intention == IntentionState.RENT_POWERBANK:
        yield from _rent_powerbank(
            env, resources, telemetry, powerbank_price=powerbank_price, rng=rng
        )
        telemetry.flow_to_powerbank += 1
        return

    if intention != IntentionState.QUEUE_AT_KIOSK:
        # Unexpected residual state: treat as abort.
        telemetry.n_abort += 1
        return

    # --- Physical seize with reneging + boarding race -----------------------
    resource, mean_charge, label = _choose_charger(passenger, resources)
    join_time = float(env.now)
    patience = _dynamic_patience_min(passenger, env.now, deadline)
    remaining_to_board = max(0.0, deadline - float(env.now))

    req = resource.request()
    reneg_ev = env.timeout(patience)
    board_ev = env.timeout(remaining_to_board)
    result = yield req | reneg_ev | board_ev

    if req not in result:
        # Lost the race: cancel request → reneging leave event.
        req.cancel()
        telemetry.n_reneged += 1
        passenger._set_intention(IntentionState.RENEGING)
        after = passenger._resolve_reneging(powerbank_price)
        if after == IntentionState.RENT_POWERBANK:
            yield from _rent_powerbank(
                env, resources, telemetry, powerbank_price=powerbank_price, rng=rng
            )
            telemetry.flow_to_powerbank += 1
        else:
            telemetry.n_abort += 1
            telemetry.flow_to_abort += 1
        return

    wait = float(env.now) - join_time
    if wait < 0.0:
        raise RuntimeError(f"negative wait time (clock regression): {wait}")
    telemetry.wait_times.append(wait)

    # Perception update only (remaining boarding time); physical hold already won.
    passenger.perceive(time_to_boarding=max(0.0, deadline - float(env.now)))

    # Charge until service ends or boarding deadline forces unplug.
    service = _sample_service_time(mean_charge, rng)
    plug_slack = max(0.0, deadline - float(env.now))
    charge_time = min(service, plug_slack)
    busy_start = float(env.now)
    try:
        if charge_time > 0.0:
            yield env.timeout(charge_time)
    finally:
        resource.release(req)
        busy = float(env.now) - busy_start
        if busy < 0.0:
            raise RuntimeError("negative busy interval")
        if label == "fast":
            telemetry.busy_fast_min += busy
            telemetry.n_served_fast += 1
            telemetry.flow_to_fast += 1
        else:
            telemetry.busy_slow_min += busy
            telemetry.n_served_slow += 1
            telemetry.flow_to_slow += 1


def _rent_powerbank(
    env: simpy.Environment,
    resources: AirportResources,
    telemetry: Telemetry,
    *,
    powerbank_price: float,
    rng: np.random.Generator,
) -> Iterator[simpy.Event]:
    """Seize one kiosk slot for ``POWERBANK_TX_MIN`` if stock remains."""
    _ = powerbank_price
    _ = rng
    if resources.powerbank_stock[0] <= 0:
        telemetry.n_abort += 1
        return
    with resources.powerbank_kiosk.request() as req:
        result = yield req | env.timeout(5.0)
        if req not in result:
            telemetry.n_reneged += 1
            telemetry.n_abort += 1
            return
        if resources.powerbank_stock[0] <= 0:
            telemetry.n_abort += 1
            return
        yield env.timeout(POWERBANK_TX_MIN)
        resources.powerbank_stock[0] -= 1
        telemetry.total_powerbank_rentals += 1


def _queue_monitor(
    env: simpy.Environment,
    resources: AirportResources,
    telemetry: Telemetry,
    horizon: float,
) -> Iterator[simpy.Event]:
    """Periodic max-queue sampler (1-minute DES ticks)."""
    while env.now <= horizon:
        telemetry.note_queue(
            resources.fast_chargers,
            resources.slow_lockers,
            time_minutes=float(env.now),
        )
        if env.now >= horizon:
            break
        yield env.timeout(QUEUE_SAMPLE_DT)


# Storm injection defaults (minutes since 06:00).
STORM_ONSET_MIN: Final[float] = 11.0 * 60.0  # 17:00
STORM_DELAY_FRACTION: Final[float] = 0.40
STORM_DELAY_MIN_LO: Final[float] = 2.0 * 60.0
STORM_DELAY_MIN_HI: Final[float] = 4.0 * 60.0


def _apply_storm_to_passenger(
    *,
    t_arr: float,
    battery: float,
    time_to_boarding: float,
    rng: np.random.Generator,
    storm_onset_min: float = STORM_ONSET_MIN,
    delay_fraction: float = STORM_DELAY_FRACTION,
) -> Tuple[float, float, bool]:
    """
    Inject thunderstorm delay for post-onset departures.

    40% of passengers whose boarding falls at/after 17:00 get +U(2h,4h)
    dwell and halved SOC (doubled drain).

    Returns
    -------
    (battery, time_to_boarding, was_delayed)
    """
    abs_board = float(t_arr) + float(time_to_boarding)
    if abs_board < storm_onset_min:
        return float(battery), float(time_to_boarding), False
    if rng.random() >= delay_fraction:
        return float(battery), float(time_to_boarding), False
    extra = float(rng.uniform(STORM_DELAY_MIN_LO, STORM_DELAY_MIN_HI))
    bat = max(0.5, float(battery) / 2.0)
    return bat, float(time_to_boarding) + extra, True


def _arrival_generator(
    env: simpy.Environment,
    resources: AirportResources,
    telemetry: Telemetry,
    arrival_times: np.ndarray,
    *,
    powerbank_price: float,
    rng: np.random.Generator,
    horizon_min: float,
    storm_injection: bool = False,
) -> Iterator[simpy.Event]:
    """Spawn a ``passenger_process`` at each NHPP epoch."""
    batteries = sample_battery_levels(len(arrival_times), rng=rng)
    boards = sample_boarding_countdowns(len(arrival_times), rng=rng)
    betas = sample_price_sensitivity(len(arrival_times), rng=rng)

    for i, t_arr in enumerate(arrival_times):
        delay = float(t_arr) - float(env.now)
        if delay > 0.0:
            yield env.timeout(delay)
        elif delay < -1e-9:
            raise RuntimeError("arrival schedule not monotone w.r.t. env.now")

        telemetry.n_arrivals += 1
        telemetry.note_arrival(float(env.now), horizon_min)
        bat = float(batteries[i])
        board = float(boards[i])
        if storm_injection:
            bat, board, delayed = _apply_storm_to_passenger(
                t_arr=float(t_arr),
                battery=bat,
                time_to_boarding=board,
                rng=rng,
            )
            if delayed:
                telemetry.n_storm_delayed += 1
        if bat < BATTERY_ANXIETY_THRESHOLD:
            telemetry.n_anxious += 1
        beliefs = PassengerBeliefs(
            battery_level=bat,
            time_to_boarding=board,
            price_sensitivity=float(betas[i]),
        )
        agent = BDIPassenger(passenger_id=i, beliefs=beliefs)
        env.process(
            passenger_process(
                env,
                agent,
                resources,
                telemetry,
                powerbank_price=powerbank_price,
                rng=rng,
            )
        )


# Named traffic scenarios for the interactive sandbox / Stage-2 runners.
SCENARIO_PRESETS: Dict[str, Dict[str, float]] = {
    "normal": {"lambda_base": 0.35, "peak_multiplier": 3.0, "pricing_mode": 1.0},
    "morning_rush": {"lambda_base": 0.55, "peak_multiplier": 4.0, "pricing_mode": 1.0},
    "thunderstorm": {"lambda_base": 0.90, "peak_multiplier": 5.0, "pricing_mode": 1.2},
}


def run_airport_simulation(
    c_fast: int,
    c_slow: int,
    c_bank: int,
    pricing_mode: float = 1.0,
    *,
    lambda_base: float = DEFAULT_LAMBDA_BASE,
    peak_multiplier: float = 3.0,
    scenario: Optional[str] = None,
    seed: int = 42,
    horizon_min: float = HORIZON_MIN,
    storm_injection: bool = False,
) -> Dict[str, Any]:
    """
    Run one 16-hour airport charging DES experiment.

    Parameters
    ----------
    c_fast :
        Capacity of 65W fast charger pool ``C_fast`` (integer ≥ 0).
    c_slow :
        Capacity of 15W slow locker pool ``C_slow``.
    c_bank :
        Number of power-bank kiosk groups ``C_bank`` (each holds 48 banks).
    pricing_mode :
        Multiplier on base power-bank price (default 1.0). Overridden when
        ``scenario`` is set, unless the caller passes an explicit non-default
        after scenario resolution (scenario wins for λ and default price).
    lambda_base :
        Off-peak NHPP rate λ₀ [1/min]. Peaks reach ``peak_multiplier`` λ₀.
    peak_multiplier :
        Peak / off-peak intensity ratio (default 3).
    scenario :
        Optional preset key: ``normal`` | ``morning_rush`` | ``thunderstorm``.
        ``thunderstorm`` also enables ``storm_injection`` (17:00 delay shock).
    seed :
        RNG seed for arrivals and service draws.
    horizon_min :
        Simulation length in minutes (default 960).
    storm_injection :
        If True, from 17:00 onward 40% of remaining departures are delayed
        2–4 h and SOC is halved (doubled drain). Extra stranded arrivals are
        merged into the NHPP after onset.

    Returns
    -------
    dict
        Keys: ``mean_wait_time``, ``max_queue_len``, ``reneging_rate``,
        ``utilization_fast``, ``utilization_slow``, ``total_powerbank_rentals``,
        ``queue_timeline`` (includes ``system_load``, ``in_service``, ``lambda_t``),
        plus diagnostic counts and BDI flow counters.
    """
    if scenario is not None:
        key = scenario.strip().lower().replace(" ", "_")
        if key not in SCENARIO_PRESETS:
            raise ValueError(
                f"unknown scenario {scenario!r}; choose from {list(SCENARIO_PRESETS)}"
            )
        preset = SCENARIO_PRESETS[key]
        lambda_base = float(preset["lambda_base"])
        peak_multiplier = float(preset["peak_multiplier"])
        # Scenario price is a default; explicit pricing_mode from caller still applied
        # when they pass pricing_mode separately — here we adopt preset if using scenario.
        pricing_mode = float(preset["pricing_mode"])
        if key == "thunderstorm":
            storm_injection = True

    if min(c_fast, c_slow, c_bank) < 0:
        raise ValueError("capacities must be non-negative")
    if pricing_mode < 0.0:
        raise ValueError(f"pricing_mode must be non-negative: {pricing_mode}")
    if horizon_min <= 0.0:
        raise ValueError("horizon_min must be positive")

    rng = np.random.default_rng(seed)
    env = simpy.Environment()

    # Capacity 0 is invalid for simpy.Resource; use capacity 1 and block stock.
    fast = simpy.Resource(env, capacity=max(1, c_fast)) if c_fast > 0 else None
    slow = simpy.Resource(env, capacity=max(1, c_slow)) if c_slow > 0 else None
    bank = simpy.Resource(env, capacity=max(1, c_bank)) if c_bank > 0 else None

    # When capacity is 0, install a dummy resource that is always saturated via
    # a pre-hold... simpler: require ≥1 for sandbox; clamp for API friendliness.
    if c_fast == 0:
        fast = simpy.Resource(env, capacity=1)
        # Occupy forever so no one is served on fast.
        def _block(res: simpy.Resource) -> Iterator[simpy.Event]:
            with res.request() as r:
                yield r
                yield env.timeout(horizon_min * 10.0)

        env.process(_block(fast))
    if c_slow == 0:
        slow = simpy.Resource(env, capacity=1)

        def _block_s(res: simpy.Resource) -> Iterator[simpy.Event]:
            with res.request() as r:
                yield r
                yield env.timeout(horizon_min * 10.0)

        env.process(_block_s(slow))
    if c_bank == 0:
        bank = simpy.Resource(env, capacity=1)

        def _block_b(res: simpy.Resource) -> Iterator[simpy.Event]:
            with res.request() as r:
                yield r
                yield env.timeout(horizon_min * 10.0)

        env.process(_block_b(bank))

    assert fast is not None and slow is not None and bank is not None
    stock = c_bank * POWERBANKS_PER_KIOSK
    resources = AirportResources(
        fast_chargers=fast,
        slow_lockers=slow,
        powerbank_kiosk=bank,
        powerbank_stock=[stock],
    )
    telemetry = Telemetry(
        c_fast=c_fast,
        c_slow=c_slow,
        c_bank=c_bank,
        lambda_base=float(lambda_base),
        peak_multiplier=float(peak_multiplier),
        arrivals_per_min=[0] * (int(horizon_min) + 1),
    )
    price = BASE_POWERBANK_PRICE * float(pricing_mode)

    arrivals = _nhpp_arrival_times(
        horizon_min,
        lambda_base,
        rng,
        peak_multiplier=peak_multiplier,
    )
    if storm_injection:
        # Extra stranded arrivals after 17:00 (homogeneous Poisson dwell surge).
        post_times: List[float] = []
        t_extra = float(STORM_ONSET_MIN)
        lam_extra = max(0.5, 2.0 * float(lambda_base))
        while True:
            t_extra += float(rng.exponential(1.0 / lam_extra))
            if t_extra >= horizon_min:
                break
            post_times.append(t_extra)
        if post_times:
            arrivals = np.sort(
                np.concatenate([arrivals, np.asarray(post_times, dtype=np.float64)])
            )
    env.process(
        _arrival_generator(
            env,
            resources,
            telemetry,
            arrivals,
            powerbank_price=price,
            rng=rng,
            horizon_min=horizon_min,
            storm_injection=storm_injection,
        )
    )
    env.process(_queue_monitor(env, resources, telemetry, horizon_min))

    # Drain residual charge / rental after last arrival.
    env.run(until=horizon_min + MEAN_CHARGE_SLOW_MIN + 5.0)

    mean_wait = (
        float(np.mean(telemetry.wait_times)) if telemetry.wait_times else 0.0
    )
    denom = max(1, telemetry.n_seekers)
    reneging_rate = float(telemetry.n_reneged) / float(denom)

    util_fast = (
        telemetry.busy_fast_min / (float(c_fast) * horizon_min)
        if c_fast > 0
        else 0.0
    )
    util_slow = (
        telemetry.busy_slow_min / (float(c_slow) * horizon_min)
        if c_slow > 0
        else 0.0
    )
    # Clamp numerical noise into [0, 1].
    util_fast = float(min(1.0, max(0.0, util_fast)))
    util_slow = float(min(1.0, max(0.0, util_slow)))

    return {
        "mean_wait_time": mean_wait,
        "max_queue_len": int(telemetry.max_queue_len),
        "max_system_load": int(telemetry.max_system_load),
        "reneging_rate": reneging_rate,
        "utilization_fast": util_fast,
        "utilization_slow": util_slow,
        "total_powerbank_rentals": int(telemetry.total_powerbank_rentals),
        "n_arrivals": int(telemetry.n_arrivals),
        "n_anxious": int(telemetry.n_anxious),
        "n_seekers": int(telemetry.n_seekers),
        "n_reneged": int(telemetry.n_reneged),
        "n_served_fast": int(telemetry.n_served_fast),
        "n_served_slow": int(telemetry.n_served_slow),
        "n_abort": int(telemetry.n_abort),
        "n_storm_delayed": int(telemetry.n_storm_delayed),
        # Stage-2 report aliases (same values; no second computation).
        "total_passengers": int(telemetry.n_arrivals),
        "anxious_passengers": int(telemetry.n_anxious),
        "served_fast": int(telemetry.n_served_fast),
        "served_slow": int(telemetry.n_served_slow),
        "served_powerbank": int(telemetry.total_powerbank_rentals),
        "reneged_passengers": int(telemetry.n_reneged),
        "flow_calm": int(telemetry.flow_calm),
        "flow_to_fast": int(telemetry.flow_to_fast),
        "flow_to_slow": int(telemetry.flow_to_slow),
        "flow_to_powerbank": int(telemetry.flow_to_powerbank),
        "flow_to_abort": int(telemetry.flow_to_abort),
        "queue_timeline": list(telemetry.queue_timeline),
        "horizon_min": float(horizon_min),
        "lambda_base": float(lambda_base),
        "peak_multiplier": float(peak_multiplier),
        "pricing_mode": float(pricing_mode),
        "scenario": scenario,
        "storm_injection": bool(storm_injection),
        "c_fast": int(c_fast),
        "c_slow": int(c_slow),
        "c_bank": int(c_bank),
        "seed": int(seed),
    }


if __name__ == "__main__":
    metrics = run_airport_simulation(c_fast=4, c_slow=8, c_bank=2, pricing_mode=1.0)
    assert metrics["n_arrivals"] > 0, "NHPP produced no arrivals"
    assert metrics["max_queue_len"] >= 0
    assert 0.0 <= metrics["reneging_rate"] <= 1.0
    assert 0.0 <= metrics["utilization_fast"] <= 1.0
    assert 0.0 <= metrics["utilization_slow"] <= 1.0
    print("[OK] run_airport_simulation metrics:")
    for key in (
        "mean_wait_time",
        "max_queue_len",
        "reneging_rate",
        "utilization_fast",
        "utilization_slow",
        "total_powerbank_rentals",
        "n_arrivals",
        "n_seekers",
        "n_reneged",
    ):
        print(f"  {key}: {metrics[key]}")
