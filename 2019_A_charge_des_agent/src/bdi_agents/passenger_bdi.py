"""
Airport BDI passenger agent (Belief–Desire–Intention).

Maps cognitive control-loop steps (sense → deliberate → act) onto discrete
intention transitions, following the BDI–DES decoupling in arXiv:2602.13878:
cognitive events stay separate from physical queue events in SimPy.

Utility (desire):
    U_charge = w1 * (40 - E)^+ - w2 * Cost - w3 * WaitTime
Hard constraint: if WaitTime > T_board - 15 min, U_charge = -inf (miss flight).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults (minutes / dimensionless). Cognitive layer only; DES owns clocks.
# ---------------------------------------------------------------------------

BATTERY_ANXIETY_THRESHOLD: Final[float] = 15.0
BATTERY_TARGET_COMFORT: Final[float] = 40.0
BOARDING_BUFFER_MIN: Final[float] = 15.0
BOARDING_T_MIN: Final[float] = 45.0
BOARDING_T_MAX: Final[float] = 120.0
ANXIETY_FRACTION: Final[float] = 0.25
SERVICE_MIN_PER_PERSON: Final[float] = 3.0
DEFAULT_W1: Final[float] = 1.0
DEFAULT_W2: Final[float] = 1.0
DEFAULT_W3: Final[float] = 1.0
DEFAULT_KIOSK_COST: Final[float] = 0.0
AFFORDABLE_PRICE_BASE: Final[float] = 25.0


class IntentionState(str, Enum):
    """Intention stack labels for the BDI passenger agent."""

    IDLE = "IDLE"
    SEEK_POWER = "SEEK_POWER"
    QUEUE_AT_KIOSK = "QUEUE_AT_KIOSK"
    RENEGING = "RENEGING"
    RENT_POWERBANK = "RENT_POWERBANK"
    ABORT_CHARGING = "ABORT_CHARGING"


@dataclass
class PassengerBeliefs:
    """Belief base B: facts the agent currently treats as true."""

    battery_level: float
    time_to_boarding: float
    price_sensitivity: float

    def as_dict(self) -> MutableMapping[str, float]:
        """Return a mutable belief dictionary for DES perception updates."""
        return {
            "battery_level": float(self.battery_level),
            "time_to_boarding": float(self.time_to_boarding),
            "price_sensitivity": float(self.price_sensitivity),
        }

    def validate(self) -> None:
        """Raise if belief components leave their modelling domains."""
        if not 0.0 <= self.battery_level <= 100.0:
            raise ValueError(f"battery_level out of [0,100]: {self.battery_level}")
        if self.time_to_boarding < 0.0:
            raise ValueError(f"time_to_boarding negative: {self.time_to_boarding}")
        if not 0.5 <= self.price_sensitivity <= 2.0:
            raise ValueError(
                f"price_sensitivity β must lie in [0.5, 2.0]: {self.price_sensitivity}"
            )


def sample_battery_levels(
    n: int,
    *,
    anxiety_fraction: float = ANXIETY_FRACTION,
    anxiety_threshold: float = BATTERY_ANXIETY_THRESHOLD,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample initial device SOC (%) with a fixed anxiety mass.

    Parameters
    ----------
    n :
        Passenger count. Shape of the returned vector is ``(n,)``.
    anxiety_fraction :
        Share of travellers with ``E < anxiety_threshold`` (default 0.25).
    anxiety_threshold :
        Anxiety cut-off in percent (default 15).
    rng :
        NumPy Generator; if None, a fresh default_rng() is used.

    Returns
    -------
    np.ndarray
        Battery levels in ``[0, 100]``, dtype float64.

    Notes
    -----
    Anxiety mass ~ Uniform(0, threshold). Complementary mass ~
    Uniform(threshold, 100). This is a modelling prior until a field survey
    CSV is placed under ``data/raw/``.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not 0.0 <= anxiety_fraction <= 1.0:
        raise ValueError(f"anxiety_fraction must be in [0,1]: {anxiety_fraction}")
    gen = rng if rng is not None else np.random.default_rng()
    n_anxious = int(round(n * anxiety_fraction))
    n_calm = n - n_anxious
    anxious = gen.uniform(0.0, anxiety_threshold, size=n_anxious)
    calm = gen.uniform(anxiety_threshold, 100.0, size=n_calm)
    levels = np.concatenate([anxious, calm])
    gen.shuffle(levels)
    return levels.astype(np.float64)


def sample_boarding_countdowns(
    n: int,
    *,
    t_min: float = BOARDING_T_MIN,
    t_max: float = BOARDING_T_MAX,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample boarding countdowns ``T_board ~ Uniform(t_min, t_max)`` (minutes).

    Parameters
    ----------
    n :
        Passenger count. Output shape ``(n,)``.
    t_min, t_max :
        Inclusive lower / exclusive-style upper bounds of the uniform (minutes).
    rng :
        Optional NumPy Generator.

    Returns
    -------
    np.ndarray
        Countdown minutes, dtype float64.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if t_max <= t_min:
        raise ValueError(f"require t_max > t_min, got [{t_min}, {t_max}]")
    gen = rng if rng is not None else np.random.default_rng()
    return gen.uniform(t_min, t_max, size=n).astype(np.float64)


def sample_price_sensitivity(
    n: int,
    *,
    beta_min: float = 0.5,
    beta_max: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample price sensitivity ``β ~ Uniform(beta_min, beta_max)``."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    gen = rng if rng is not None else np.random.default_rng()
    return gen.uniform(beta_min, beta_max, size=n).astype(np.float64)


def load_or_sample_passenger_features(
    n: int,
    *,
    csv_path: Optional[Path | str] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load passenger feature table from CSV, or sample the modelling priors.

    Expected CSV columns (if present):
    ``battery_level``, ``time_to_boarding``, ``price_sensitivity``.

    Parameters
    ----------
    n :
        Row count when sampling. Ignored if CSV has enough rows (then head(n)).
    csv_path :
        Optional path under ``data/raw/``. Missing / incomplete → sample.
    rng, seed :
        Reproducibility controls. ``seed`` builds a Generator if ``rng`` is None.

    Returns
    -------
    pd.DataFrame
        Columns ``battery_level``, ``time_to_boarding``, ``price_sensitivity``.
    """
    gen = rng
    if gen is None:
        gen = np.random.default_rng(seed)

    if csv_path is not None:
        path = Path(csv_path)
        if path.is_file():
            frame = pd.read_csv(path)
            required = {"battery_level", "time_to_boarding", "price_sensitivity"}
            if required.issubset(frame.columns):
                if len(frame) < n:
                    raise ValueError(
                        f"CSV {path} has {len(frame)} rows; need at least {n}"
                    )
                return frame.loc[: n - 1, list(required)].reset_index(drop=True)

    return pd.DataFrame(
        {
            "battery_level": sample_battery_levels(n, rng=gen),
            "time_to_boarding": sample_boarding_countdowns(n, rng=gen),
            "price_sensitivity": sample_price_sensitivity(n, rng=gen),
        }
    )


def estimate_wait_time_min(
    observed_queue_len: int,
    *,
    service_min_per_person: float = SERVICE_MIN_PER_PERSON,
) -> float:
    """
    Map observed queue length to expected wait (minutes).

    Parameters
    ----------
    observed_queue_len :
        Non-negative headcount ahead of the agent (physical DES observation).
    service_min_per_person :
        Mean service time per queued person (minutes).

    Returns
    -------
    float
        Estimated wait in minutes. Never negative.
    """
    if observed_queue_len < 0:
        raise ValueError(
            f"queue length must be non-negative (DES invariant): {observed_queue_len}"
        )
    if service_min_per_person < 0.0:
        raise ValueError("service_min_per_person must be non-negative")
    return float(observed_queue_len) * float(service_min_per_person)


@dataclass
class BDIPassenger:
    """
    Cognitive BDI passenger: beliefs, desire utility, intention automaton.

    The same specification is intended to run against a SimPy DES environment
    API later (arXiv:2602.13878 portable BDI–DES mapping). Physical clocks and
    resource holds live outside this class.
    """

    passenger_id: int
    beliefs: PassengerBeliefs
    intention: IntentionState = IntentionState.IDLE
    w1: float = DEFAULT_W1
    w2: float = DEFAULT_W2
    w3: float = DEFAULT_W3
    kiosk_cost: float = DEFAULT_KIOSK_COST
    service_min_per_person: float = SERVICE_MIN_PER_PERSON
    affordable_price_base: float = AFFORDABLE_PRICE_BASE
    history: list[IntentionState] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.beliefs.validate()
        self.history.append(self.intention)

    @classmethod
    def from_row(
        cls,
        passenger_id: int,
        row: Mapping[str, float],
        **kwargs: float,
    ) -> "BDIPassenger":
        """Construct from a feature mapping / DataFrame row."""
        beliefs = PassengerBeliefs(
            battery_level=float(row["battery_level"]),
            time_to_boarding=float(row["time_to_boarding"]),
            price_sensitivity=float(row["price_sensitivity"]),
        )
        return cls(passenger_id=passenger_id, beliefs=beliefs, **kwargs)

    def perceive(
        self,
        *,
        battery_level: Optional[float] = None,
        time_to_boarding: Optional[float] = None,
    ) -> None:
        """Update beliefs from environment perceptions (external events)."""
        if battery_level is not None:
            self.beliefs.battery_level = float(battery_level)
        if time_to_boarding is not None:
            self.beliefs.time_to_boarding = float(time_to_boarding)
        self.beliefs.validate()

    def charge_utility(
        self,
        *,
        wait_time: float,
        cost: float,
    ) -> float:
        """
        Desire evaluation ``U_charge``.

        Parameters
        ----------
        wait_time :
            Estimated wait at the kiosk (minutes). Must be >= 0.
        cost :
            Perceived monetary cost (already scaled by β if desired).

        Returns
        -------
        float
            Utility; ``-inf`` if miss-flight hard constraint fires.
        """
        if wait_time < 0.0:
            raise ValueError(f"wait_time must be non-negative: {wait_time}")
        t_board = self.beliefs.time_to_boarding
        if wait_time > (t_board - BOARDING_BUFFER_MIN):
            return float("-inf")
        e = self.beliefs.battery_level
        deficit = max(0.0, BATTERY_TARGET_COMFORT - e)
        return float(self.w1 * deficit - self.w2 * cost - self.w3 * wait_time)

    def _set_intention(self, new_state: IntentionState) -> IntentionState:
        if new_state != self.intention:
            self.intention = new_state
            self.history.append(new_state)
        return self.intention

    def _queue_is_feasible(self, wait_time: float, cost: float) -> bool:
        """True iff charging desire is finite and not dominated by abort (0)."""
        u = self.charge_utility(wait_time=wait_time, cost=cost)
        return np.isfinite(u) and u > float("-inf")

    def _should_renege(self, observed_queue_len: int, wait_time: float) -> bool:
        """
        Reneging trigger: miss-flight hard constraint or excessive queue.

        Queue length > 5 is an operational reneging threshold used in stress
        tests (tight boarding window).
        """
        if wait_time > (self.beliefs.time_to_boarding - BOARDING_BUFFER_MIN):
            return True
        if observed_queue_len > 5:
            return True
        return False

    def _powerbank_affordable(self, powerbank_price: float) -> bool:
        """β-adjusted willingness-to-pay for a rented power bank."""
        ceiling = self.affordable_price_base / self.beliefs.price_sensitivity
        return float(powerbank_price) <= ceiling

    def evaluate_intentions(
        self,
        observed_queue_len: int,
        powerbank_price: float,
    ) -> IntentionState:
        """
        Deliberation step: map beliefs + observations → next intention.

        Parameters
        ----------
        observed_queue_len :
            Current kiosk queue length (non-negative integer from DES).
        powerbank_price :
            Posted rental price for a power bank (currency units).

        Returns
        -------
        IntentionState
            Updated intention after one cognitive cycle.
        """
        if observed_queue_len < 0:
            raise ValueError(
                f"observed_queue_len must be non-negative: {observed_queue_len}"
            )
        wait_time = estimate_wait_time_min(
            observed_queue_len,
            service_min_per_person=self.service_min_per_person,
        )
        beta = self.beliefs.price_sensitivity
        kiosk_perceived_cost = self.kiosk_cost * beta
        e = self.beliefs.battery_level

        # --- IDLE ----------------------------------------------------------
        if self.intention == IntentionState.IDLE:
            if e < BATTERY_ANXIETY_THRESHOLD:
                return self._set_intention(IntentionState.SEEK_POWER)
            return self.intention

        # --- SEEK_POWER ----------------------------------------------------
        if self.intention == IntentionState.SEEK_POWER:
            if self._should_renege(observed_queue_len, wait_time):
                # Cannot safely queue: jump into reneging branch.
                self._set_intention(IntentionState.RENEGING)
                return self._resolve_reneging(powerbank_price)
            if self._queue_is_feasible(wait_time, kiosk_perceived_cost):
                return self._set_intention(IntentionState.QUEUE_AT_KIOSK)
            # Infeasible desire → reneging / abort path.
            self._set_intention(IntentionState.RENEGING)
            return self._resolve_reneging(powerbank_price)

        # --- QUEUE_AT_KIOSK ------------------------------------------------
        if self.intention == IntentionState.QUEUE_AT_KIOSK:
            if self._should_renege(observed_queue_len, wait_time):
                self._set_intention(IntentionState.RENEGING)
                return self._resolve_reneging(powerbank_price)
            # Still feasible: remain in queue.
            return self.intention

        # --- RENEGING / terminals ------------------------------------------
        if self.intention == IntentionState.RENEGING:
            return self._resolve_reneging(powerbank_price)

        # Terminal intentions stay put unless re-evaluated from IDLE later.
        if self.intention in (
            IntentionState.RENT_POWERBANK,
            IntentionState.ABORT_CHARGING,
        ):
            return self.intention

        return self.intention

    def _resolve_reneging(self, powerbank_price: float) -> IntentionState:
        """RENEGING → RENT_POWERBANK if affordable, else ABORT_CHARGING."""
        if self._powerbank_affordable(powerbank_price):
            return self._set_intention(IntentionState.RENT_POWERBANK)
        return self._set_intention(IntentionState.ABORT_CHARGING)

    def did_renege(self) -> bool:
        """True if the intention history contains RENEGING."""
        return IntentionState.RENEGING in self.history


def build_population(
    n: int,
    *,
    csv_path: Optional[Path | str] = None,
    seed: int = 42,
) -> list[BDIPassenger]:
    """
    Build ``n`` BDI passenger agents from CSV or sampled priors.

    Returns
    -------
    list[BDIPassenger]
        Agents with ids ``0 .. n-1``.
    """
    features = load_or_sample_passenger_features(n, csv_path=csv_path, seed=seed)
    agents: list[BDIPassenger] = []
    for i, row in features.iterrows():
        agents.append(BDIPassenger.from_row(int(i), row))
    return agents


def _assert_tight_boarding_always_reneges(
    *,
    n_trials: int = 100,
    queue_len: int = 6,
    time_to_boarding: float = 10.0,
    seed: int = 0,
) -> None:
    """
    Self-check: T_board = 10 min and queue > 5 ⇒ 100% reneging leave.

    With boarding buffer 15 min, wait > T_board - 15 is automatic for any
    non-negative wait when T_board = 10. Queue length 6 also trips the
    operational reneging rule.
    """
    gen = np.random.default_rng(seed)
    reneged = 0
    for i in range(n_trials):
        beliefs = PassengerBeliefs(
            battery_level=float(gen.uniform(1.0, 14.9)),
            time_to_boarding=time_to_boarding,
            price_sensitivity=float(gen.uniform(0.5, 2.0)),
        )
        agent = BDIPassenger(
            passenger_id=i,
            beliefs=beliefs,
            intention=IntentionState.QUEUE_AT_KIOSK,
        )
        final = agent.evaluate_intentions(
            observed_queue_len=queue_len,
            powerbank_price=float(gen.uniform(5.0, 40.0)),
        )
        left_queue = final in (
            IntentionState.RENEGING,
            IntentionState.RENT_POWERBANK,
            IntentionState.ABORT_CHARGING,
        ) and agent.did_renege()
        if left_queue:
            reneged += 1
        else:
            raise AssertionError(
                f"trial {i}: expected reneging leave, got {final}, "
                f"history={agent.history}"
            )
    rate = reneged / n_trials
    assert rate == 1.0, f"reneging rate {rate} != 1.0 over {n_trials} trials"
    print(
        f"[OK] {n_trials}/{n_trials} agents reneged when "
        f"T_board={time_to_boarding} min and queue_len={queue_len}."
    )


def _demo_anxiety_mass(seed: int = 7) -> None:
    """Print empirical anxiety share for a sampled cohort (sanity check)."""
    levels = sample_battery_levels(10_000, rng=np.random.default_rng(seed))
    share = float(np.mean(levels < BATTERY_ANXIETY_THRESHOLD))
    print(f"[info] empirical anxiety share E<15%: {share:.4f} (target 0.25)")


if __name__ == "__main__":
    _demo_anxiety_mass()
    _assert_tight_boarding_always_reneges()

    # Extra path check: IDLE → SEEK_POWER when E < 15%.
    agent = BDIPassenger(
        passenger_id=999,
        beliefs=PassengerBeliefs(
            battery_level=10.0,
            time_to_boarding=90.0,
            price_sensitivity=1.0,
        ),
    )
    nxt = agent.evaluate_intentions(observed_queue_len=2, powerbank_price=20.0)
    assert nxt == IntentionState.SEEK_POWER, nxt
    nxt = agent.evaluate_intentions(observed_queue_len=2, powerbank_price=20.0)
    assert nxt == IntentionState.QUEUE_AT_KIOSK, nxt
    print(f"[OK] IDLE→SEEK_POWER→QUEUE_AT_KIOSK path: {agent.history}")
