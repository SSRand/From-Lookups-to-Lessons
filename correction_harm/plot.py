"""Render Fig. 4 from the full-answer transition counts and frozen dev selection.

    python -m correction_harm.plot --data runs/fig4/data/correction_harm.json --output figs/correction_harm
"""
import argparse
import json
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

COLORS={'mlpmemory':'#D55E00','resmem':'#0072B2','memory_decoder':'#009E73'}
LABELS={'mlpmemory':'MLP Memory','resmem':'ResMem','memory_decoder':'Memory Decoder'}
MARKERS={'mlpmemory':'s','resmem':'o','memory_decoder':'^'}
SYMBOLS={'mlpmemory':r'\lambda','resmem':r'\gamma','memory_decoder':r'\lambda'}
LINESTYLES={'mlpmemory':'--','resmem':'-','memory_decoder':'-.'}
METHODS=('mlpmemory','resmem','memory_decoder')
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
 'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':.65,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#E6E6E6',linewidth=.5,zorder=0)
    ax.tick_params(width=.65,length=3)
    ax.set_axisbelow(True)

def err(ax,x,y,ci,color='black',**kw):
    ax.errorbar(x,y,yerr=np.array([[y-ci[0]],[ci[1]-y]]),color=color,
                capsize=2.4,elinewidth=.8,capthick=.8,**kw)

UNITS='percent'
BASE_CORRECT=BASE_WRONG=1

def hx_(v):
    # Harm on the x axis: percent of Base-correct answers, or a count.
    return 100*v if UNITS=='percent' else v*BASE_CORRECT

def cy_(v):
    # Correction on the y axis: percent of Base-wrong answers, or a count.
    return 100*v if UNITS=='percent' else v*BASE_WRONG

def draw_curve(ax,rows,selection,small=False,labels=False,label_xmax=12,zoomed=True):
    for method in METHODS:
        rr=sorted((r for r in rows if r['method']==method),key=lambda r:r['lambda_'])
        x=[hx_(r['harm_rate']) for r in rr];y=[cy_(r['correction_rate']) for r in rr]
        ax.plot(x,y,color=COLORS[method],marker=MARKERS[method],markersize=3 if small else 4.5,
                linewidth=1 if small else 1.55,linestyle=LINESTYLES[method],
                markerfacecolor='white' if method=='mlpmemory' else COLORS[method],
                markeredgewidth=.85,zorder=3,label=LABELS[method])
        chosen=selection.get(method)
        if chosen is not None:
            r=next(r for r in rr if r['lambda_']==chosen)
            sx,sy=hx_(r['harm_rate']),cy_(r['correction_rate'])
            ax.scatter([sx],[sy],marker='*',s=45 if small else 145,facecolors=COLORS[method],
                       edgecolors='white',linewidths=.8,zorder=6)
            if not small:
                hx=[hx_(v) for v in r['ci95']['harm_rate']];cy=[cy_(v) for v in r['ci95']['correction_rate']]
                ax.errorbar(sx,sy,xerr=[[sx-hx[0]],[hx[1]-sx]],yerr=[[sy-cy[0]],[cy[1]-sy]],
                            fmt='none',color=COLORS[method],elinewidth=.8,capsize=2,zorder=4)
        if labels and method!='memory_decoder':
            for r in rr:
                l=r['lambda_'];xx,yy=hx_(r['harm_rate']),cy_(r['correction_rate'])
                if l==0 or xx>label_xmax:continue
                selected=l==chosen
                if not zoomed and not selected:continue
                text='$'+SYMBOLS[method]+(r'^*' if selected else '')+'='+f'{l:g}'+'$'
                if method=='mlpmemory': offset=(6,1) if l==.1 else (5,-12) if l in [.2,.3] else (4,-11)
                else: offset=(-4,9) if l<=.4 else (4,7)
                ax.annotate(text,(xx,yy),xytext=offset,textcoords='offset points',
                            ha='left' if offset[0]>=0 else 'right',fontsize=7.5 if not selected else 8,
                            color=COLORS[method],fontweight='bold' if selected else 'normal')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True,help='output path without extension; writes .pdf and .png')
    ap.add_argument('--units',choices=('percent','count'),default='percent')
    ap.add_argument('--aspect',choices=('auto','equal'),default='auto');a=ap.parse_args()
    global UNITS,BASE_CORRECT,BASE_WRONG
    data=json.loads(a.data.read_text());selected=dict(data['selected']);n=data['n']
    UNITS,BASE_CORRECT,BASE_WRONG=a.units,data['base_correct'],data['base_wrong']
    for r in data['pooled']:
        assert r['correction']+r['harm']+r['preserved']+r['still_wrong']==n
        assert abs(r['net_em']-(r['correction']-r['harm'])/n)<1e-12
    fig=plt.figure(figsize=(6.1,3.35))
    # (a) trade-off curve on the left, (b) decomposition at the selected coefficients on the right.
    gs=fig.add_gridspec(1,2,width_ratios=[1.50,.90],left=.075 if UNITS=='percent' else .095,right=.985,bottom=.20,top=.89,wspace=.30 if UNITS=='percent' else .34)
    right=fig.add_subplot(gs[0]);left=fig.add_subplot(gs[1]);style(left);style(right)
    left.axhline(0,color='#333333',linewidth=.8,zorder=2)
    gain=[];loss=[]
    for i,method in enumerate(METHODS):
        r=next(r for r in data['pooled'] if r['method']==method and r['lambda_']==selected[method])
        if UNITS=='percent': c=100*r['correction_contribution'];h=100*r['harm_contribution']
        else: c=r['correction'];h=r['harm']
        gain.append(c);loss.append(h)
        left.bar(i,c,width=.46,color=COLORS[method],alpha=.88,zorder=2)
        left.bar(i,-h,width=.46,facecolor='white',edgecolor=COLORS[method],hatch='////',linewidth=.9,zorder=2)
        fmt=(lambda v:f'{v:.2f}') if UNITS=='percent' else (lambda v:f'{v:d}')
        left.annotate('+'+fmt(c),(i,c),xytext=(0,4),textcoords='offset points',ha='center',va='bottom',fontsize=9,color=COLORS[method])
        left.annotate('−'+fmt(h),(i,-h),xytext=(0,-4),textcoords='offset points',ha='center',va='top',fontsize=9,color=COLORS[method])
    pad=.18*max(gain)
    left.set_xlim(-.55,2.55);left.set_ylim(-max(loss)-pad,max(gain)+pad)
    short={'mlpmemory':'MLP\nMemory','resmem':'ResMem\n','memory_decoder':'Memory\nDecoder'}
    left.set_xticks([0,1,2],[short[m]+'\n$'+SYMBOLS[m]+r'^*='+f'{selected[m]:g}'+'$' for m in METHODS],fontsize=6.8)
    left.set_ylabel('Change in EM (points)' if UNITS=='percent' else 'Answers')
    left.legend(handles=[Patch(facecolor='#888888',label='Corrected'),
                         Patch(facecolor='white',edgecolor='#888888',hatch='////',label='Broken')],
                loc='lower center',bbox_to_anchor=(.5,1.005),frameon=False,ncol=2,
                fontsize=7,handlelength=.9,handletextpad=.35,columnspacing=.7)
    xmax=12 if UNITS=='percent' else 1400
    ymax=4.55 if UNITS=='percent' else 1250
    draw_curve(right,data['pooled'],selected,labels=True,label_xmax=xmax)
    right.set_xlim(-.015*xmax,xmax);right.set_ylim(-.015*ymax,ymax)
    if UNITS=='percent':right.set_xticks([0,2,4,6,8,10,12]);right.set_yticks([0,1,2,3,4])
    if a.aspect=='equal':right.set_aspect('equal',adjustable='box')
    suffix=' (%)' if UNITS=='percent' else ''
    right.set_xlabel('Correct Base answers broken'+suffix)
    right.set_ylabel('Wrong Base answers corrected'+suffix)
    slope=BASE_CORRECT/BASE_WRONG if UNITS=='percent' else 1.0
    right.plot([0,xmax],[0,xmax*slope],color='#9A9A9A',linewidth=1.55,linestyle=':',zorder=1)
    fig.canvas.draw()
    lx=min(.97*xmax,.965*ymax/slope);p0=right.transData.transform((0,0));p1=right.transData.transform((lx,lx*slope))
    right.annotate('ΔEM = 0',(lx,lx*slope),xytext=(-3,3),textcoords='offset points',color='#777777',fontsize=7,ha='right',va='bottom',
               rotation=np.degrees(np.arctan2(p1[1]-p0[1],p1[0]-p0[0])),rotation_mode='anchor')
    inset=right.inset_axes([.59,.06,.38,.34]);style(inset)
    draw_curve(inset,data['pooled'],selected,small=True)
    fx=100 if UNITS=='percent' else BASE_CORRECT;fy=4.2 if UNITS=='percent' else 1150
    inset.set_xlim(-.02*fx,1.03*fx);inset.set_ylim(-.02*fy,1.02*fy)
    inset.tick_params(labelsize=6,length=2,pad=1);inset.locator_params(nbins=3)
    inset.set_title('Full grid',fontsize=6.4,pad=3)
    inset.set_facecolor('#FAFAFA');inset.patch.set_alpha(.98)
    for method in ('mlpmemory','resmem'):
        last=next(r for r in data['pooled'] if r['method']==method and r['lambda_']==1)
        inset.annotate('$'+SYMBOLS[method]+'=1$',(hx_(last['harm_rate']),cy_(last['correction_rate'])),xytext=(-4,5),textcoords='offset points',fontsize=5.5,color=COLORS[method])
    right.legend(handles=[Line2D([],[],color=COLORS[m],marker=MARKERS[m],markerfacecolor='white' if m=='mlpmemory' else COLORS[m],
                                  linestyle=LINESTYLES[m],markersize=4,label=LABELS[m]) for m in METHODS],
                 loc='lower center',bbox_to_anchor=(.47,1.005),frameon=False,ncol=3,fontsize=7.2,handlelength=1.8,columnspacing=.9)
    fig.text(.035,.955,'(a)',fontweight='bold',fontsize=10)
    fig.text(.085,.955,'Correction–harm trade-off',fontsize=9)
    fig.text(.655,.955,'(b)',fontweight='bold',fontsize=10)
    fig.text(.705,.955,'Net gain decomposition',fontsize=9)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png'):fig.savefig(a.output.with_suffix('.'+ext),dpi=260)
    plt.close(fig)
    print(a.output.with_suffix('.pdf'))

if __name__=='__main__':main()
