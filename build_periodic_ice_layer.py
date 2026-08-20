#!/usr/bin/env python3
"""Match and build a periodic Ice-Ih(0001)-like layer on a catalyst slab."""

import argparse, json
from itertools import product
from pathlib import Path
import networkx as nx
import numpy as np
from ase import Atom, Atoms
from ase.build import fcc100, fcc110, fcc111, make_supercell
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, reference_states
from ase.io import read, write


def unit(v):
    v=np.asarray(v,float); return v/np.linalg.norm(v)


def inplane_area(cell): return float(np.linalg.norm(np.cross(cell[0],cell[1])))


def integer_matrices(max_det, limit=3):
    result=[]
    for values in product(range(-limit,limit+1),repeat=4):
        matrix=np.array(values,int).reshape(2,2); determinant=int(round(np.linalg.det(matrix)))
        if 1 <= determinant <= max_det: result.append(matrix)
    return result


def ice_primitive_cell(oo_distance=2.76):
    a=np.sqrt(3.)*float(oo_distance)
    return np.array([[a,0.],[.5*a,np.sqrt(3.)*.5*a]])


def match_periodic_ice(cell,oo_distance=2.76,max_strain=.08,max_substrate_area=4,
                       allow_substrate_expansion=True):
    """Match 2D integer cells; water count is 2*det(ice matrix), never fixed."""
    substrate=np.asarray(cell,float)[:2,:2]
    if abs(np.linalg.det(substrate)) < 1e-8: raise ValueError("Slab needs two independent in-plane vectors")
    water=ice_primitive_cell(oo_distance); ratio=abs(np.linalg.det(substrate))/abs(np.linalg.det(water))
    smats=integer_matrices(max_substrate_area if allow_substrate_expansion else 1)
    wmats=integer_matrices(max(2,int(np.ceil(ratio*max_substrate_area))+2))
    candidates=[]
    for sm in smats:
        ds=int(round(np.linalg.det(sm))); ss=sm@substrate
        for wm in wmats:
            dw=int(round(np.linalg.det(wm)))
            if abs(dw-ratio*ds)>1.5: continue
            ws=wm@water
            if abs(np.linalg.det(ws))<1e-8: continue
            deformation=np.linalg.solve(ws,ss)
            strain=float(np.max(np.abs(np.linalg.svd(deformation,compute_uv=False)-1.)))
            if strain<=max_strain:
                candidates.append((strain+.002*(ds-1),strain,sm,wm,ds,dw,deformation))
    if not candidates:
        scope="fixed catalyst cell" if not allow_substrate_expansion else f"expansions up to {max_substrate_area}x area"
        raise ValueError(f"No ice coincidence cell for {scope} within {100*max_strain:.1f}% strain; expand the catalyst cell or revise the reported strain threshold, never insert a fixed water count.")
    _,strain,sm,wm,ds,dw,deformation=min(candidates,key=lambda x:x[0])
    return {"max_principal_strain":strain,"substrate_matrix":sm,"water_matrix":wm,
            "substrate_area_multiplier":ds,"ice_primitive_cells":dw,"water_count":2*dw,
            "deformation":deformation}


def embed(matrix):
    result=np.eye(3,dtype=int);result[:2,:2]=matrix;return result


def water_oxygen_net(matrix,target_cell,oo_distance):
    primitive=ice_primitive_cell(oo_distance);cell=np.zeros((3,3));cell[:2,:2]=primitive;cell[2,2]=20.
    oxygen=Atoms("O2",scaled_positions=[[0.,0.,.5],[1/3,1/3,.5]],cell=cell,pbc=[True,True,False]);oxygen.set_tags([1,2])
    oxygen=make_supercell(oxygen,embed(matrix),wrap=True);scaled=oxygen.get_scaled_positions(wrap=True)
    oxygen.set_cell(target_cell,scale_atoms=False);oxygen.set_scaled_positions(scaled);return oxygen


def build_periodic_ice_layer_on_slab(slab,oo_distance=2.76,max_strain=.08,max_substrate_area=4,
                                     allow_substrate_expansion=True,fixed_layers=2):
    requested_area=inplane_area(slab.cell);requested_angle=float(slab.cell.angles()[2])
    match=match_periodic_ice(slab.cell,oo_distance,max_strain,max_substrate_area,allow_substrate_expansion)
    slab=make_supercell(slab,embed(match["substrate_matrix"]),wrap=True)
    levels=sorted(set(np.round(slab.positions[:,2],6)))
    fixed=[i for i,z in enumerate(slab.positions[:,2]) if any(abs(z-level)<.05 for level in levels[:fixed_layers])]
    top_z=max(slab.positions[:,2]);oxygen=water_oxygen_net(match["water_matrix"],slab.cell,oo_distance)
    oxygen.positions[:,2]=[top_z+(2.40 if tag==1 else 2.82) for tag in oxygen.get_tags()]
    nwater=len(oxygen);graph=nx.Graph();graph.add_nodes_from(range(nwater));edges=[]
    cutoff=oo_distance*(1+max_strain)+.25
    for left in range(nwater):
        for right in range(left+1,nwater):
            displacement=oxygen.get_distance(left,right,mic=True,vector=True)
            if np.linalg.norm(displacement)<=cutoff: graph.add_edge(left,right);edges.append((left,right,displacement))
    if any(graph.degree[n]!=3 for n in graph): raise RuntimeError(f"O net is not a three-connected periodic honeycomb: {dict(graph.degree)}")
    matching={frozenset(e) for e in nx.max_weight_matching(graph,maxcardinality=True)}
    if 2*len(matching)!=nwater: raise RuntimeError("Water net has no perfect proton-ordering matching")
    tags=oxygen.get_tags();donors={n:[] for n in graph}
    for left,right,displacement in edges:
        donor=(left if tags[left]==2 else right) if frozenset((left,right)) in matching else (left if tags[left]==1 else right)
        donors[donor].append(unit(displacement if donor==left else -displacement))
    atoms=slab.copy();water_indices=[];angle=np.deg2rad(104.5)
    for node in range(nwater):
        position=oxygen.positions[node];directions=donors[node]
        if tags[node]==1:
            bisector=unit(directions[0]+directions[1]);transverse=unit(directions[0]-directions[1])
            hdirs=[np.cos(angle/2)*bisector+np.sin(angle/2)*transverse,np.cos(angle/2)*bisector-np.sin(angle/2)*transverse]
        else:
            network=directions[0];down=unit(np.array([0.,0.,-1.])-np.dot([0.,0.,-1.],network)*network)
            hdirs=[network,np.cos(angle)*network+np.sin(angle)*down]
        start=len(atoms);atoms+=Atom("O",position);atoms+=Atom("H",position+.97*unit(hdirs[0]));atoms+=Atom("H",position+.97*unit(hdirs[1]));water_indices.append([start,start+1,start+2])
    atoms.set_constraint(FixAtoms(indices=fixed));atoms.set_tags([1]*len(slab)+[2]*(3*nwater))
    metadata={"status":"constructed_only_no_UMA_no_relaxation","water_layer":"commensurate periodic Ice-Ih(0001)-like H-down honeycomb",
              "water_count_rule":"derived from catalyst/ice 2D coincidence supercell; never fixed","n_water":nwater,
              "ice_oo_distance_A":float(oo_distance),"max_principal_strain":match["max_principal_strain"],
              "substrate_supercell_matrix":match["substrate_matrix"].tolist(),"ice_supercell_matrix":match["water_matrix"].tolist(),
              "substrate_area_multiplier":match["substrate_area_multiplier"],"surface_area_A2":inplane_area(slab.cell),
              "surface_angle_degree":float(slab.cell.angles()[2]),"requested_surface_area_A2":requested_area,
              "requested_surface_angle_degree":requested_angle,"periodic_oxygen_coordination":[graph.degree[n] for n in graph],
              "fixed_atom_indices_zero_based":fixed,"water_atom_indices_zero_based":water_indices,
              "warning":"Constructed matched ice-like initial model; relax and inspect before interpreting energies."}
    return atoms,metadata


def build_periodic_ice_layer(metal="Pt",facet="111",size=None,lattice_a=None,layers=4,vacuum=10.,fixed_layers=2,**options):
    reference=reference_states[atomic_numbers[metal]]
    if not reference or reference.get("symmetry")!="fcc": raise ValueError("Automatic elemental generation supports fcc metals; upload other prepared slabs")
    lattice_a=float(lattice_a or reference["a"]);facet=str(facet).replace("(","").replace(")","");xy=tuple(size or ((3,3) if facet=="111" else (2,2)))
    builder={"111":fcc111,"100":fcc100,"110":fcc110}.get(facet)
    if builder is None: raise ValueError("Automatic matched-ice generation supports fcc(111/100/110); upload other slabs")
    kwargs={"size":(*xy,layers),"a":lattice_a,"vacuum":vacuum}
    if facet=="111":kwargs["orthogonal"]=False
    if facet=="110":kwargs["orthogonal"]=True
    atoms,metadata=build_periodic_ice_layer_on_slab(builder(metal,**kwargs),fixed_layers=fixed_layers,**options)
    metadata.update({"surface":f"{metal}({facet})","requested_surface_size":[*xy,layers],"lattice_a_A":lattice_a});return atoms,metadata


def main():
    p=argparse.ArgumentParser();p.add_argument("--metal",default="Pt");p.add_argument("--facet",default="111");p.add_argument("--structure",type=Path);p.add_argument("--size",nargs=2,type=int);p.add_argument("--lattice-a",type=float);p.add_argument("--layers",type=int,default=4);p.add_argument("--vacuum",type=float,default=10.);p.add_argument("--fixed-layers",type=int,default=2);p.add_argument("--ice-oo-distance",type=float,default=2.76);p.add_argument("--max-strain",type=float,default=.08);p.add_argument("--max-substrate-area",type=int,default=4);p.add_argument("--fixed-catalyst-cell",action="store_true");p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    options={"oo_distance":a.ice_oo_distance,"max_strain":a.max_strain,"max_substrate_area":a.max_substrate_area,"allow_substrate_expansion":not a.fixed_catalyst_cell,"fixed_layers":a.fixed_layers}
    if a.structure: atoms,metadata=build_periodic_ice_layer_on_slab(read(a.structure.resolve(),-1),**options);metadata["surface_source"]=str(a.structure.resolve())
    else: atoms,metadata=build_periodic_ice_layer(a.metal,a.facet,a.size,a.lattice_a,a.layers,a.vacuum,**options)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);write(out/"periodic_ice_interface.extxyz",atoms);write(out/"periodic_ice_interface.traj",atoms);write(out/"periodic_ice_interface.cif",atoms);write(out/"POSCAR",atoms,format="vasp",direct=True,sort=False);(out/"structure_manifest.json").write_text(json.dumps(metadata,indent=2));print(out)


if __name__=="__main__":main()
