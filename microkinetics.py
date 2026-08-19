#!/usr/bin/env python3
"""Ideal single-site mean-field steady-state microkinetics."""

import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

KB_EV=8.617333262e-5; KB_J=1.380649e-23; PLANCK=6.62607015e-34


def main():
    p=argparse.ArgumentParser(); p.add_argument("network", type=Path); p.add_argument("--output", type=Path, required=True); a=p.parse_args()
    plan=json.loads(a.network.read_text()); temperature=float(plan.get("temperature_K",298.15)); species=plan["surface_species"]
    if "*" not in species: raise ValueError("surface_species must include vacant site '*'")
    unknowns=[x for x in species if x!="*"]; reactions=plan["reactions"]; activities={k:float(v) for k,v in plan.get("activities",{}).items()}
    prefactor=KB_J*temperature/PLANCK
    def unpack(x):
        z=np.exp(np.r_[x,0.0]-np.max(np.r_[x,0.0])); z/=z.sum(); return dict(zip(unknowns+["*"],map(float,z)))
    def rate(r,theta):
        kf=prefactor*np.exp(-float(r["forward_barrier_eV"])/(KB_EV*temperature)); kr=prefactor*np.exp(-float(r["reverse_barrier_eV"])/(KB_EV*temperature)); f,b=kf,kr
        for name,nu in r["stoichiometry"].items():
            value=theta[name] if name in theta else activities.get(name)
            if value is None: raise ValueError(f"Missing activity for {name}")
            if nu<0: f*=value**(-float(nu))
            elif nu>0: b*=value**float(nu)
        return f-b,kf,kr
    def residual(x):
        theta=unpack(x); balances={name:0.0 for name in unknowns}
        for r in reactions:
            net,_,_=rate(r,theta)
            for name,nu in r["stoichiometry"].items():
                if name in balances: balances[name]+=float(nu)*net
        return np.array([balances[name] for name in unknowns])
    solution=least_squares(residual,np.zeros(len(unknowns)),xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=10000); theta=unpack(solution.x); rows=[]
    for r in reactions:
        net,kf,kr=rate(r,theta); rows.append({"reaction":r["id"],"net_rate_s-1":net,"kf_s-1":kf,"kr_s-1":kr})
    tof_id=plan.get("tof_reaction"); tof=next((x["net_rate_s-1"] for x in rows if x["reaction"]==tof_id),None)
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    with (out/"coverages.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=["species","coverage"]); w.writeheader(); w.writerows({"species":k,"coverage":v} for k,v in theta.items())
    with (out/"rates.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    result={"status":"converged" if solution.success else "failed","temperature_K":temperature,"steady_state_residual_norm":float(np.linalg.norm(residual(solution.x))),"coverages":theta,"rates":rows,"tof_reaction":tof_id,"tof_s-1":tof,"rate_model":"k=(kBT/h) exp(-DeltaG_dagger/kBT)","warning":"Ideal single-site mean-field model."}
    (out/"microkinetics.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
