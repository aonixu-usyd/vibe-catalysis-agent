#!/usr/bin/env python3
"""CatMAP-style pathways with aligned state and TS top views."""
import argparse, csv, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.visualize.plot import plot_atoms

COLORS={"pcet":"#3266A8","coupled":"#25897C","decoupled-proton":"#B56A3B","decoupled-electron":"#7655A6","chemical":"#586174"}

def curve(ax,x0,xts,x1,e0,ets,e1,color):
    t=np.linspace(0,1,40); smooth=.5-.5*np.cos(np.pi*t)
    ax.plot(x0+(xts-x0)*t,e0+(ets-e0)*smooth,color=color,lw=2)
    ax.plot(xts+(x1-xts)*t,ets+(e1-ets)*smooth,color=color,lw=2)

def top_view(ax,path,title):
    if not path or not Path(path).is_file(): ax.axis("off"); ax.set_title(title,fontsize=7); return
    plot_atoms(read(path,-1),ax=ax,rotation="0x,0y,0z",show_unit_cell=1,radii=.68)
    ax.set_title(title,fontsize=7,pad=4); ax.set_facecolor("#F8FAFC")

def main():
    p=argparse.ArgumentParser(); p.add_argument("results",type=Path); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    data=json.loads(a.results.read_text()); states={x["id"]:x for x in data["states"]}; steps=data["steps"]
    if not steps: raise ValueError("Path requires at least one step")
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":7,"axes.linewidth":.8,"svg.fonttype":"none","pdf.fonttype":42})
    linear=all(steps[i]["reactant"]==steps[i-1]["product"] for i in range(1,len(steps))); source=[]
    if linear:
        state_ids=[steps[0]["reactant"]]+[s["product"] for s in steps]; energies=[0.0]
        for step in steps: energies.append(energies[-1]+float(step["delta_G_approx_eV"]))
        positions=np.arange(0,2*len(state_ids),2); panels=2*len(steps)+1
        fig=plt.figure(figsize=(max(8.0,2.05*panels),6.4),constrained_layout=True); grid=fig.add_gridspec(2,panels,height_ratios=(1.65,1)); ax=fig.add_subplot(grid[0,:])
        for i,(sid,e,x) in enumerate(zip(state_ids,energies,positions)):
            ax.hlines(e,x-.42,x+.42,color="#162033",lw=2.4); ax.text(x,e+.05,f"{e:+.2f}",ha="center",fontsize=7)
            source.append({"order":2*i,"kind":"state","label":states[sid].get("label",sid),"relative_energy_eV":e})
            top_view(fig.add_subplot(grid[1,2*i]),states[sid].get("structure"),states[sid].get("label",sid))
            if i<len(steps):
                step=steps[i]; e1=energies[i+1]; color=COLORS[step["mechanism"]]
                if "forward_barrier_eV" in step:
                    ets=e+float(step["forward_barrier_eV"]); curve(ax,x+.42,x+1,x+1.58,e,ets,e1,color)
                    ax.text(x+1,ets+.06,f"$E_a$={float(step['forward_barrier_eV']):.2f}",ha="center",fontsize=6)
                    ax.text(x+1,(e+e1)/2-.08,f"$\\Delta E$={float(step['delta_G_approx_eV']):+.2f}",ha="center",fontsize=6)
                    source.append({"order":2*i+1,"kind":"TS candidate","label":step.get("label",step["id"]),"relative_energy_eV":ets})
                    top_view(fig.add_subplot(grid[1,2*i+1]),step.get("transition_state_structure"),"TS "+step.get("label",step["id"]))
                else: ax.plot([x+.42,x+1.58],[e,e1],color=color,lw=1.5); fig.add_subplot(grid[1,2*i+1]).axis("off")
        ax.set_xticks(positions,[states[s].get("label",s) for s in state_ids]); ax.set_ylabel("Relative electronic energy (eV)"); ax.spines[["top","right"]].set_visible(False); ax.axhline(0,color="#A9AFB8",lw=.7)
        ax.set_title(data.get("title","Catalytic reaction pathway"),loc="left",fontsize=11,weight="bold")
    else:
        labels=[f"{states[s['reactant']].get('label',s['reactant'])} → {states[s['product']].get('label',s['product'])}" for s in steps]; dg=[float(s["delta_G_approx_eV"]) for s in steps]; ea=[float(s.get("forward_barrier_eV",np.nan)) for s in steps]; x=np.arange(len(steps)); width=.34
        fig,ax=plt.subplots(figsize=(max(7.2,1.5*len(steps)),4.2),constrained_layout=True); ax.bar(x-width/2,dg,width,color=[COLORS[s["mechanism"]] for s in steps],label="$\\Delta E$"); ax.bar(x+width/2,ea,width,color="#C8CDD5",edgecolor="#596273",label="$E_a$")
        ax.set_xticks(x,labels); ax.legend(); ax.set_ylabel("Energy (eV)"); ax.set_title(data.get("title","Competing elementary steps"),loc="left",fontsize=11,weight="bold")
        source=[{"order":i,"kind":"branch","label":label,"relative_energy_eV":value} for i,(label,value) in enumerate(zip(labels,dg))]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for ext,kwargs in (("svg",{}),("pdf",{}),("png",{"dpi":300}),("tiff",{"dpi":600})): fig.savefig(a.output.with_suffix("."+ext),bbox_inches="tight",**kwargs)
    plt.close(fig)
    with a.output.with_name(a.output.name+"_source_data.csv").open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=list(source[0])); w.writeheader(); w.writerows(source)
    a.output.with_name(a.output.name+"_caption.txt").write_text("UMA electronic-energy pathway with aligned ASE top views for sequential states and available TS candidates. Validate barriers with consistent DFT and frequencies; ZPE, entropy, solvent, potential/field and coverage corrections are absent.\n")
    print(a.output)

if __name__=="__main__": main()
