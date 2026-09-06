"""Deterministic generic relay queue demonstration, not a radio model."""

from dataclasses import asdict, dataclass
import math
import time

import simpy


@dataclass(frozen=True)
class Scenario:
    messages: int = 12
    interval_s: float = 0.1
    transit_s: float = 0.02
    service_s: float = 0.15
    duration_s: float = 3.0

    def validate(self):
        if isinstance(self.messages, bool) or not isinstance(self.messages, int) or not 1 <= self.messages <= 1000:
            raise ValueError("消息数必须为 1～1000 的整数")
        for field in ("interval_s", "transit_s", "service_s", "duration_s"):
            value = getattr(self, field)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 3600:
                raise ValueError(f"{field} 必须是 (0,3600] 内的有限秒数")


def simulate(scenario):
    scenario.validate()
    started = time.perf_counter()
    env = simpy.Environment()
    relay = simpy.Resource(env, capacity=1)
    events = []
    latencies = []
    sent = []

    def record(message_id, node, stage):
        events.append({"event_id": len(events), "time_s": float(env.now),
                       "message_id": message_id, "node": node, "stage": stage})

    def transfer(message_id):
        begin = env.now
        sent.append(message_id)
        record(message_id, "A", "sent")
        yield env.timeout(scenario.transit_s)
        record(message_id, "relay", "queued")
        with relay.request() as request:
            yield request
            record(message_id, "relay", "processing")
            yield env.timeout(scenario.service_s)
            record(message_id, "relay", "forwarded")
        yield env.timeout(scenario.transit_s)
        record(message_id, "B", "received")
        latencies.append(float(env.now - begin))

    def source():
        for index in range(scenario.messages):
            env.process(transfer(index))
            yield env.timeout(scenario.interval_s)

    env.process(source())
    # Include events on the horizon; SimPy run(until=t) excludes some events at t.
    while env.peek() <= scenario.duration_s:
        env.step()
    return {
        "model": "generic_relay_queue_v1", "abstraction_level": "event_demo",
        "scenario": asdict(scenario), "events": events,
        "summary": {"planned": scenario.messages, "sent": len(sent),
                    "received": len(latencies), "pending": len(sent) - len(latencies),
                    "not_started": scenario.messages - len(sent),
                    "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
                    "simulated_duration_s": scenario.duration_s,
                    "wall_duration_s": time.perf_counter() - started},
        "unsupported": ["radio_waveform", "frequency_hopping", "synchronization", "TDMA", "coding"],
    }
