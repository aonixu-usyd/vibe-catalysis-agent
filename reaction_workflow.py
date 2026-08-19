#!/usr/bin/env python3
"""Natural-workflow backend for adsorption states, barriers, kinetics, and figures."""

import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent; MECHANISMS={"pcet","coupled","decoupled-proton","decoupled-electron","chemical"}


def validate(plan):
    ids=[s.get("id") for s in plan.get("states",[])];
    if not ids or len(ids)!=len(set(ids)): raise ValueError("States need unique IDs")
    for step in plan.get("steps",[]):
        if step.get("reactant") not in ids or step.get("product") not in ids: raise ValueError("Step references unknown state")
        if step.get("mechanism") not in MECHANISMS: raise ValueError("Unknown elementary mechanism")
        h,e=int(step.get("protons",0)),int(step.get("electrons",0))
        if abs(h)>1 or abs(e)>1: raise ValueError("Split multi-pair transfer into elementary steps")
        if step["mechanism"]=="pcet" and (h==0 or h!=e): raise ValueError("PCET requires one coupled proton/electron pair")
    return plan


def main():
    p=argparse.ArgumentParser(); p.add_argument("plan",type=Path); p.add_argument("--output",type=Path,required=True); p.add_argument("--dry-run",action="store_true"); a=p.parse_args(); plan=validate(json.loads(a.plan.read_text())); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False); (out/"states").mkdir(); commands=[]
    surface=plan["surface"]
    for state in plan["states"]:
        cmd=[sys.executable,str(ROOT/"predict_adsorption.py")]
        cmd += ["--structure",str(Path(surface["structure"]).expanduser().resolve())] if surface.get("structure") else ["--metal",surface["metal"],"--facet",str(surface.get("facet","111"))]
        cmd += (["--adsorbate-file",str(Path(state["adsorbate_file"]).expanduser().resolve())] if state.get("adsorbate_file") else ["--adsorbate",state["adsorbate"]]) + ["--output",str(out/"states"/state["id"])]
        for k,v in (plan.get("calculation",{})|state.get("calculation",{})).items(): cmd += (["--"+k.replace("_","-"),*map(str,v)] if isinstance(v,list) else (["--"+k.replace("_","-")] if v is True else ["--"+k.replace("_","-"),str(v)]))
        commands.append(cmd)
        if not a.dry_run: subprocess.run(cmd,check=True)
    barriers={}; (out/"barriers").mkdir(exist_ok=True)
    for step in plan["steps"]:
        b=step.get("barrier");
        if not b: continue
        initial=b.get("initial") or out/"states"/step["reactant"]/"best_structure.extxyz"; final=b.get("final") or out/"states"/step["product"]/"best_structure.extxyz"; target=out/"barriers"/step["id"]
        cmd=[sys.executable,str(ROOT/"calculate_barrier.py"),"--initial",str(Path(initial).resolve()),"--final",str(Path(final).resolve()),"--output",str(target)]
        for k,v in b.items():
            if k in {"initial","final"}: continue
            if v is True: cmd.append("--"+k.replace("_","-"))
            elif v is not False and v is not None: cmd += ["--"+k.replace("_","-"),str(v)]
        commands.append(cmd)
        if not a.dry_run: subprocess.run(cmd,check=True); barriers[step["id"]]=json.loads((target/"barrier.json").read_text())
    (out/"workflow_manifest.json").write_text(json.dumps({"status":"validated" if a.dry_run else "calculated","commands":commands,"plan":plan},indent=2,default=str))
    if a.dry_run: print(out); return
    rows=[]
    for state in plan["states"]:
        s=json.loads((out/"states"/state["id"]/"summary.json").read_text()); rows.append({"id":state["id"],"label":state.get("label",state["id"]),"total_energy_eV":s["best_candidate"]["relaxed_total_eV"]})
    by={x["id"]:x for x in rows}; steps=[]
    for step in plan["steps"]:
        dg=by[step["product"]]["total_energy_eV"]-by[step["reactant"]]["total_energy_eV"]+float(step.get("reservoir_energy_eV",0))+float(step.get("energy_correction_eV",0)); item={**step,"delta_G_approx_eV":dg}
        if step["id"] in barriers: item.update({k:barriers[step["id"]][k] for k in ("forward_barrier_eV","reverse_barrier_eV","status")})
        steps.append(item)
    result={"title":plan.get("title","Catalytic reaction pathway"),"states":rows,"steps":steps,"conditions":plan.get("conditions",{}),"energy_level":"UMA electronic energies"}; (out/"pathway_results.json").write_text(json.dumps(result,indent=2))
    if plan.get("microkinetics"):
        network=plan["microkinetics"]; calculated={s["id"]:s for s in steps}
        for reaction in network.get("reactions",[]):
            source=reaction.pop("barrier_step",None)
            if source:
                reaction["forward_barrier_eV"]=calculated[source]["forward_barrier_eV"]
                reaction["reverse_barrier_eV"]=calculated[source]["reverse_barrier_eV"]
        network.setdefault("temperature_K",float(plan.get("conditions",{}).get("temperature_k",298.15))); network_path=out/"microkinetic_plan.json"; network_path.write_text(json.dumps(network,indent=2))
        subprocess.run([sys.executable,str(ROOT/"microkinetics.py"),str(network_path),"--output",str(out/"microkinetics")],check=True)
    subprocess.run([sys.executable,str(ROOT/"plot_reaction_path.py"),str(out/"pathway_results.json"),"--output",str(out/"figure")],check=True)


if __name__=="__main__": main()
