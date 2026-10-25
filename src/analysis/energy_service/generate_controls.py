"""Regenerate the locked controller commands from anonymous inputs and probes.

This step chooses thermostat commands. It does not simulate building physics.
All input locations are relative to --input-root; no historical checkout is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def distributions(temperature, weights):
    order = np.argsort(temperature, axis=1)
    temperature = np.take_along_axis(temperature, order, axis=1)
    weights = np.take_along_axis(np.broadcast_to(weights, temperature.shape), order, axis=1)
    zero = np.zeros((len(temperature), 1))
    return (temperature, np.c_[zero, np.cumsum(weights, axis=1)],
            np.c_[zero, np.cumsum(weights * temperature, axis=1)],
            np.c_[zero, np.cumsum(weights * temperature ** 2, axis=1)])


def interval_mse(cdf, low, high):
    result = []
    for temperature, cw, ct, ct2 in zip(*cdf):
        left = np.searchsorted(temperature, low, side="left")
        right = np.searchsorted(temperature, high, side="right")
        value = low ** 2 * cw[left] - 2 * low * ct[left] + ct2[left]
        value += (ct2[-1] - ct2[right] - 2 * high * (ct[-1] - ct[right])
                  + high ** 2 * (cw[-1] - cw[right]))
        result.append(float(np.mean(value) / cw[-1]))
    return np.maximum(result, 0.)


def generate(input_root, *, profile_tables=None,
             permissions=("building", "floor", "native_zone"), expected_commands=1296):
    contexts = json.loads((input_root / "controller/CONTEXTS.json").read_text(encoding="utf-8"))
    designs = []
    for city, cfg in contexts.items():
        with np.load(input_root / cfg["probes"], allow_pickle=False) as saved:
            probes = dict(saved)
        if profile_tables is None:
            with np.load(input_root / cfg["profiles"], allow_pickle=False) as saved:
                data = pd.DataFrame(dict(saved))
        else:
            data = profile_tables[city]
        x = probes["uniform_target_c"]
        grid = np.round(np.arange(x.min(), x.max() + .00001, .1), 8)
        energy_proxy = np.interp(grid, x, probes["purchased_site_kwh"])
        zones = cfg["zones"]
        prediction, weights, cdfs, mean_t, mean_t2 = {}, {}, {}, {}, {}
        for zone_index, zone in enumerate(zones):
            series = probes["operative_c"][:, :, zone_index]
            prediction[zone] = np.array([np.interp(grid, x, value) for value in series.T]).T
            weights[zone] = probes["native_people"][:, zone_index]
            cdfs[zone] = distributions(prediction[zone], weights[zone])
            mean_t[zone] = np.average(prediction[zone], weights=weights[zone], axis=1)
            mean_t2[zone] = np.average(prediction[zone] ** 2, weights=weights[zone], axis=1)
        for (composition, layout), group in data.groupby(["composition_id", "layout"]):
            global_mean = float(group.centre_c.mean())
            uniform_mse = sum(np.sum((prediction[z] - global_mean) ** 2 * weights[z], axis=1)
                              for z in zones) / sum(w.sum() for w in weights.values())
            losses = {}
            for zone, local in group.groupby("zone_id"):
                mu = float(local.centre_c.mean())
                losses[zone] = {
                    "uniform_average": uniform_mse,
                    "personal_center": (mean_t2[zone] - 2 * mu * mean_t[zone] + mu ** 2
                                        + local.centre_c.var(ddof=0)),
                    "ltpref_interval": interval_mse(cdfs[zone], local.controller_lower_c.to_numpy(),
                                                     local.controller_upper_c.to_numpy()),
                }
            for permission in permissions:
                bindings = {z: ("building" if permission == "building" else
                                cfg["floor_groups"][z] if permission == "floor" else z)
                            for z in zones}
                for control in ("uniform_average", "personal_center", "ltpref_interval"):
                    for variant, (slack, amplitude) in enumerate(((0., 1.), (.25, 1.),
                                                                  (.25, .5), (.25, 0.))):
                        targets = {}
                        for binding in dict.fromkeys(bindings.values()):
                            members = [z for z in zones if bindings[z] == binding]
                            loss = (uniform_mse if control == "uniform_average" else
                                    sum(weights[z].sum() * losses[z][control] for z in members)
                                    / sum(weights[z].sum() for z in members))
                            rms = np.sqrt(np.maximum(loss, 0))
                            eligible = np.flatnonzero(rms <= rms.min() + slack + 1e-10)
                            desired = float(grid[eligible[np.argmin(energy_proxy[eligible])]])
                            target = (desired if amplitude == 1 else
                                      float(np.round(25 + amplitude * (desired - 25), 1)))
                            targets.update(dict.fromkeys(members, target))
                        designs.append({"context": city, "composition_id": int(composition),
                                        "layout": str(layout), "permission": permission,
                                        "control": control, "variant": variant,
                                        "targets_c": targets,
                                        "permission_alias": "building" if permission == "floor"
                                        and cfg["floor_binding_equals_building"] else ""})
    if len(designs) != expected_commands:
        raise ValueError(f"Expected exactly {expected_commands} logical commands")
    return designs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    designs = generate(args.input_root.resolve())
    path = args.output_root / "CONTROL_COMMANDS.json"
    path.write_text(json.dumps(designs, indent=2) + "\n", encoding="utf-8")
    qa = {"logical_commands": len(designs), "native_commands": sum(
        row["permission"] == "native_zone" for row in designs),
        "physical_simulations_run": 0, "historical_checkout_required": False,
        "command_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output_root / "CONTROL_REGENERATION.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
