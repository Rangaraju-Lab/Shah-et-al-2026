from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent / "assets"
INK, BLUE, TEAL, ORANGE, GREY = "#172C45", "#3266AA", "#178D8D", "#D58B22", "#6B7785"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":12, "text.color":INK, "axes.labelcolor":INK, "axes.edgecolor":"#B7C3CF", "svg.fonttype":"none"})

def card(ax, x, y, w, h, title, text, color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.015",ec="#D9E1EA",fc="#F4F7FB",lw=1.2))
    ax.text(x+0.02,y+h-0.035,title,fontsize=13,fontweight="bold",color=color,va="top")
    ax.text(x+0.02,y+h-0.18,text,fontsize=11.2,va="top",linespacing=1.65)

def save(fig, name):
    ROOT.mkdir(exist_ok=True)
    fig.savefig(ROOT / f"{name}.png",dpi=220,facecolor="white")
    fig.savefig(ROOT / f"{name}.svg",facecolor="white")
    plt.close(fig)

def spine_figure():
    fig=plt.figure(figsize=(15,10),facecolor="white")
    fig.text(.045,.95,"SPINE-BASE FINDING",fontsize=24,fontweight="bold")
    fig.text(.045,.913,"Surface geometry → dendritic reference → radius-change knee → anchored path",fontsize=13,color=GREY)
    ax=fig.add_axes([.06,.47,.41,.37])
    ax.set_title("A  Synapse-to-shaft skeleton trace",loc="left",fontweight="bold",pad=16)
    ax.plot([-3,-1.5,0,1.5,3],[0]*5,color=TEAL,lw=10,solid_capstyle="round",alpha=.3)
    ax.plot([-3,-1.5,0,1.5,3],[0]*5,"o-",color=TEAL,lw=3,ms=7)
    trace=np.array([[0,2.8],[.15,2.3],[.15,1.8],[.12,1.3],[.03,.8],[0,.35],[0,0]])
    ax.plot(trace[:,0],trace[:,1],"o-",color=BLUE,lw=3,ms=6)
    ax.scatter([0],[2.8],s=160,color="#B34A75",zorder=5)
    ax.scatter([.03],[.8],s=150,color=ORANGE,zorder=5)
    ax.scatter([0],[0],s=140,facecolor="white",edgecolor=TEAL,lw=2,zorder=5)
    ax.annotate("snapped synapse",(0,2.8),(.65,2.65),arrowprops={"arrowstyle":"-","color":GREY})
    ax.annotate("refined base",(.03,.8),(.9,1.1),arrowprops={"arrowstyle":"-","color":GREY},color=ORANGE)
    ax.text(-2.9,-.52,"Longest leaf-to-leaf path after radius filtering",fontsize=11,color=TEAL)
    ax.annotate("meeting point on shaft",(0,0),(.8,-.25),arrowprops={"arrowstyle":"-","color":GREY},fontsize=11)
    ax.set_xlim(-3.3,3.8);ax.set_ylim(-.9,3.15);ax.axis("off")
    ax.text(-3.2,-.82,"Illustrative geometry; markers represent skeleton nodes.",fontsize=10,color=GREY)
    ax=fig.add_axes([.56,.53,.37,.29])
    x=np.arange(15);y=80+20*np.maximum(0,x-8)
    ax.plot(x,y,color=BLUE,lw=2.5,label="bounded ReLU fit")
    ax.scatter(x,y,s=40,facecolor="white",edgecolor=BLUE,zorder=3,label="radius samples")
    ax.axvline(8,color=ORANGE,ls="--",lw=2)
    ax.text(8.3,225,"knee → nearest\nobserved node",color=ORANGE,fontsize=11,va="top")
    ax.set_xlim(-.5,14.5);ax.set_ylim(0,250)
    ax.set_xlabel("Node index i on the candidate-to-shaft trace")
    ax.set_ylabel("Wavefront radius r (nm)")
    ax.set_title("B  Local radius-ramp refinement",loc="left",fontweight="bold",pad=16)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=.15)
    fig.text(.56,.44,r"$r(i)=b+m\,\max(0,i-i_0),\quad m\geq0$",fontsize=17)
    fig.text(.56,.406,"Fit uses node index, not cumulative physical distance.\nCurve is illustrative; it is not experimental data.",fontsize=10.5,color=GREY)
    ax=fig.add_axes([.045,.09,.91,.27]);ax.axis("off")
    card(ax,.0,.02,.30,.91,"1  Local surface and shaft",r"Mesh-edge weight: $w_{uv}=\|v_u-v_v\|_2$"+"\nCrop: edge-geodesic distance ≤ 7,000 nm\nExact wavefront spacing: 30 nm\nKeep shaft candidates with r ≥ 100 nm\nSelect longest weighted leaf-to-leaf path.")
    card(ax,.345,.02,.30,.91,"2  Trace and initialize", "Shortest raw-skeleton path to the shaft.\nAt meeting point q with radius R:\n"+r"$j^*=\arg\min_j |\|p_j-q\|_2-R|$"+"\nThis is Euclidean distance to q,\nnot a backward arclength traversal.")
    card(ax,.69,.02,.30,.91,"3  Fit and relax", "Fit the candidate-to-shaft radius tail.\nRequire ≥ 4 tail nodes for refinement.\nChoose the observed node nearest i₀.\n"+r"$p_i' = p_i+0.5(\Delta_{\perp}+0.5\Delta_{\parallel})$"+"\nFix synapse/base endpoints; 100 iterations.")
    for x in [.315,.66]:ax.add_patch(FancyArrowPatch((x,.47),(x+.023,.47),arrowstyle="-|>",mutation_scale=16,color=GREY))
    fig.text(.045,.04,"Path length is the sum of relaxed edge lengths. The source algorithm's numerical decisions are retained.",fontsize=10.5,color=GREY)
    save(fig,"spine_base_math")

def mitochondrial_figure():
    fig=plt.figure(figsize=(15,10),facecolor="white")
    fig.text(.045,.95,"MITOCHONDRIAL WIDTH & MORPHOLOGY",fontsize=24,fontweight="bold")
    fig.text(.045,.913,"Retained centerline → component-aware surface mapping → local measurements → operational rules",fontsize=13,color=GREY)
    ax=fig.add_axes([.055,.56,.38,.27])
    ax.set_title("A  Vertex-to-centerline diameter estimate",loc="left",fontweight="bold",pad=16)
    nodes=np.array([[0,0],[2,0],[4,0]])
    ax.plot(nodes[:,0],nodes[:,1],"o-",color=TEAL,lw=3,ms=9)
    for j,radius in enumerate([.6,1.1,.8]):
        points=nodes[j]+np.array([[0,radius],[0,-radius],[.3,radius*.8],[-.3,-radius*.8]])
        ax.scatter(points[:,0],points[:,1],color=BLUE,s=25)
        for point in points:ax.plot([nodes[j,0],point[0]],[0,point[1]],color=BLUE,lw=1,alpha=.4)
    ax.text(1.6,1.45,"Mean assigned radial distance × 2",fontsize=11,color=BLUE)
    ax.set_xlim(-.8,4.8);ax.set_ylim(-1.6,1.8);ax.axis("off")
    fig.text(.055,.51,r"$A_j=\{v:\ j=\operatorname{argmin}_{k\in c(v)}\|v-s_k\|_2\}$",fontsize=15)
    fig.text(.055,.463,r"$D_j=\frac{2}{|A_j|}\sum_{v\in A_j}\|v-s_j\|_2;\quad D(v)=D_{j(v)}$",fontsize=16)
    fig.text(.055,.415,r"$z(v)=\frac{D(v)-\mu_{c(v)}}{\sigma_{c(v)}}$",fontsize=17)
    fig.text(.055,.345,"Means and standard deviations are over mesh vertices within\neach component. Widths are estimates, not cross-sectional fits.\nA constant-width component receives z = 0.",fontsize=10.5,color=GREY)
    ax=fig.add_axes([.49,.43,.46,.40]);ax.axis("off")
    card(ax,0,.02,.96,.94,"B  Base and synapse → surface", "Closest point q on a mitochondrial triangle:\n"+r"$d_{\mathrm{abs}}=\|b-q\|_2$"+"\n"+r"$d_{\mathrm{signed}}=-d_{\mathrm{abs}}$ inside; $+d_{\mathrm{abs}}$ outside."+"\nMap q to its nearest triangle vertex; read D and z.\nAlso query the synapse-nearest surface/component.\n"+r"$V_c=\left|\sum_t a_t\cdot(b_t\times c_t)/6\right|/10^9$"+"\nVolume is in μm³; coordinates are in nm.",TEAL)
    ax=fig.add_axes([.045,.07,.91,.24]);ax.axis("off")
    card(ax,0,.03,.30,.92,"C  Protrusion", "Base-to-mito distance ≤ 2,000 nm\nNormalized width z > 1.5\nSigned base-to-surface distance < 200 nm\n\nNegative signed distances also qualify.",ORANGE)
    card(ax,.345,.03,.30,.92,"D  Recruitment", "Base-to-mito distance ≤ 2,000 nm\nNormalized width z > 1.0\nDoes not meet the protrusion rule.\n\nOperational label from static geometry.",TEAL)
    card(ax,.69,.03,.30,.92,"E  Budding", "Synapse-to-mito distance ≤ 500 nm\nSynapse-nearest volume ≥ 0.01 μm³\nBase-nearest and synapse-nearest\nvolumes are unequal.\nIndependent rule: categories can overlap.",BLUE)
    fig.text(.045,.03,"Unequal volume is the supplied code's proxy for different objects; these labels do not establish dynamic remodeling.",fontsize=10.5,color=GREY)
    save(fig,"mitochondrial_mapping_logic")

if __name__ == "__main__":
    spine_figure()
    mitochondrial_figure()
    print(f"Schematics saved to {ROOT}")
