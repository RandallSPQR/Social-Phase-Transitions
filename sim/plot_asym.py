import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open(_os.path.join(ROOT, "sim", "plots.py")).read().split("OUT =")[0])  # reuse Tufte rc + colors
OUT=_os.path.join(ROOT, "figures")
def note(fig, t, y=-0.02):
    fig.text(0.06, y, t, fontsize=8.5, color="#999999", ha="left", va="top", family="sans-serif")
rows=np.load(_os.path.join(ROOT, "data", "asym", "asym.npy"),allow_pickle=True)
def series(N,rho,T,k=4):
    xs,ys,es=[],[],[]
    for n,c,r,e,t,res in rows:
        if n==N and r==rho and t==T:
            xs.append(e); ys.append(res[:,k].mean()); es.append(1.96*res[:,k].std(ddof=1)/np.sqrt(len(res)))
    return np.array(xs),np.array(ys),np.array(es)
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5.2),sharex=True); fig.subplots_adjust(wspace=0.28)
for T,sh in [(1.0,"#111111"),(1.3,"#555555"),(1.6,"#999999")]:
    for N,alpha,lw in [(1000,0.35,1.0),(4000,1,1.6)]:
        x,y,e=series(N,0.5,T)
        a1.plot(x,y,color=sh,alpha=alpha,lw=lw)
        if N==4000:
            a1.errorbar(x,y,yerr=e,fmt="none",ecolor=sh,elinewidth=0.8)
            a1.text(-0.02,y[0],f"T = {T}",color=sh,fontsize=10.5,va="center",ha="right")
a1.text(0.45,0.03,"T = 2.4 (above transition): ≈0",fontsize=9.5,style="italic",color=C["text3"])
a1.text(0.36,0.30,"faint lines: N = 1,000\nsolid: N = 4,000",fontsize=9.5,style="italic",color=C["text3"])
a1.set_title("Frozen camps (glass, ρ = 0.5)",loc="left",fontsize=13,color=C["text"])
a1.set_ylabel("persistence: agents still holding\ntheir position 2,000 sweeps later")
a1.set_xlabel("non-reciprocity ε"); a1.set_ylim(-0.02,0.6); a1.set_xlim(-0.17,1.02)
a1.spines["left"].set_bounds(0,0.6); a1.spines["bottom"].set_bounds(0,1)
# right: two-party polarization (balanced network), T = 2.0, N = 2000, 20 graphs
eps=np.array([0,0.2,0.5,1.0])
asy=np.array([0.827,0.713,0.243,0.036]); asy_e=1.96*np.array([0.002,0.003,0.015,0.000])
ctl=np.array([0.827,0.741,0.542,0.054]); ctl_e=1.96*np.array([0.002,0.003,0.006,0.002])
a2.plot(eps,ctl,color="#999999",lw=1.4); a2.errorbar(eps,ctl,yerr=ctl_e,fmt="o",ms=3.5,color="#999999")
a2.plot(eps,asy,color=C["hl"],lw=1.6); a2.errorbar(eps,asy,yerr=asy_e,fmt="o",ms=3.5,color=C["hl"])
a2.text(0.52,0.58,"symmetric control,\nsame average loyalty",color="#777777",fontsize=10)
a2.text(0.36,0.20,"non-reciprocal",color=C["hl"],fontsize=10.5)
a2.set_title("Two-party polarization (balanced network, T = 2.0)",loc="left",fontsize=13,color=C["text"])
a2.set_ylabel("polarization: alignment with own party, |m|")
a2.set_xlabel("non-reciprocity ε"); a2.set_ylim(-0.02,0.9)
a2.spines["left"].set_bounds(0,0.9); a2.spines["bottom"].set_bounds(0,1)
fig.text(0.06,1.06,"Non-reciprocal ties melt frozen camps fast; two-party polarization is sturdier, not immune",fontsize=16,color=C["text"])
fig.text(0.06,1.005,"ε = share of ties where the reverse direction is redrawn independently (0 = fully reciprocal). Mean degree 4. Error bars: 95% CI over graphs.",fontsize=11.5,color=C["text2"])
note(fig,"Plain heat-bath dynamics (no energy function exists for ε > 0): 2,000-sweep warm-up, persistence averaged over the next 2,000 sweeps; two replicas per graph. Left: 40 / 16 graphs at N = 1,000 / 4,000.\n"
     "Right: N = 2,000, 20 graphs per point; control uses symmetric ties with ρ = ε/4, matching the average directed coupling. Persistence is window-relative: glass states at ε = 0 also age on longer timescales.",y=-0.03)
fig.savefig(f"{OUT}/fig4_asymmetry.png"); print("ok")
