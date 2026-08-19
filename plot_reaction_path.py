#!/usr/bin/env python3
"""Nature/CatMAP-style pathway plotting from structured Vibe results."""

import argparse, csv, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument("results",type=Path); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    data=json.loads(a.results.read_text()); states={x["id"]:x for x in data["states"]}; steps=data["steps"]
    if not steps: raise ValueError("Path requires at least one step")
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":7,"axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"svg.fonttype":"none","pdf.fonttype":42})
    linear=all(steps[i]["reactant"]==steps[i-1]["product"] for i in range(1,len(steps))); colors={"pcet":"#3266A8","coupled":"#25897C","decoupled-proton":"#B56A3B","decoupled-electron":"#7655A6","chemical":"#586174"}
    fig,ax=plt.subplots(figsize=(7.2,4.0),constrained_layout=True); source=[]
    if linear:
        energies=[0.0]; labels=[states[steps[0]["reactant"]].get("label",steps[0]["reactant"])]
        for step in steps: energies.append(energies[-1]+float(step["delta_G_approx_eV"])); labels.append(states[step["product"]].get("label",step["product"]))
        for i,e in enumerate(energies):
            ax.hlines(e,i-.28,i+.28,color="#162033",lw=2.4); ax.text(i,e+.05,f"{e:+.2f}",ha="center",fontsize=7); source.append({"order":i,"state":labels[i],"relative_energy_eV":e})
            if i<len(steps):
                step=steps[i]; c=colors[step["mechanism"]]
                if "forward_barrier_eV" in step:
                    ts=e+float(step["forward_barrier_eV"]); x1=np.linspace(i+.28,i+.5,30); x2=np.linspace(i+.5,i+1-.28,30)
                    ax.plot(x1,e+(ts-e)*np.sin(np.linspace(0,np.pi/2,30))**2,color=c); ax.plot(x2,energies[i+1]+(ts-energies[i+1])*np.cos(np.linspace(0,np.pi/2,30))**2,color=c); ax.scatter(i+.5,ts,s=12,color=c); ax.text(i+.5,ts+.05,f"$E_a$={step['forward_barrier_eV']:.2f}",ha="center",fontsize=6)
                else: ax.plot([i+.28,i+1-.28],[e,energies[i+1]],color=c)
        ax.set_xticks(range(len(labels)),labels)
    else:
        labels=[f"{states[s['reactant']].get('label',s['reactant'])} → {states[s['product']].get('label',s['product'])}" for s in steps]; dg=[float(s["delta_G_approx_eV"]) for s in steps]; x=np.arange(len(steps)); width=.34
        ax.bar(x-width/2,dg,width,color=[colors[s["mechanism"]] for s in steps],label="$\\Delta G$"); ea=[float(s.get("forward_barrier_eV",np.nan)) for s in steps]; ax.bar(x+width/2,ea,width,color="#C8CDD5",edgecolor="#596273",label="$E_a$"); ax.set_xticks(x,labels); ax.legend()
        source=[{"order":i,"state":label,"relative_energy_eV":value} for i,(label,value) in enumerate(zip(labels,dg))]
    ax.axhline(0,color="#A9AFB8",lw=.7); ax.set_ylabel("Relative energy (eV)"); ax.set_title(data.get("title","Catalytic reaction pathway"),loc="left",fontsize=10,weight="bold")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for ext,kwargs in (("svg",{}),("pdf",{}),("png",{"dpi":300}),("tiff",{"dpi":600})): fig.savefig(a.output.with_suffix("."+ext),bbox_inches="tight",**kwargs)
    plt.close(fig)
    with a.output.with_name(a.output.name+"_source_data.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(source[0])); w.writeheader(); w.writerows(source)
    a.output.with_name(a.output.name+"_caption.txt").write_text("UMA electronic-energy pathway. Validate important states and barriers with consistent DFT, vibrational free energies, and the relevant reaction environment.\n")
    print(a.output)


if __name__=="__main__": main()
