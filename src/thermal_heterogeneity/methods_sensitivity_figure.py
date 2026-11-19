"""Draw population and command-domain sensitivity from numerical source tables."""
from pathlib import Path
import argparse
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties, findfont
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FONT_SIZE = 11
FILES = ['POPULATION_COMMAND_NESTED_SUMMARY.csv', 'POPULATION_COMMAND_SELECTIONS.csv',
         'UNIFORM_COMMAND_ENERGY_CURVES.csv', 'COMMAND_DOMAIN_SUMMARY.csv']
RESULT_FILES = ['population_command_sensitivity/NESTED_SAMPLE_SUMMARY.csv',
                'population_command_sensitivity/ALL_SELECTIONS.csv',
                'command_domain_sensitivity/UNIFORM_COMMAND_ENERGY_CURVES.csv',
                'command_domain_sensitivity/DOMAIN_SUMMARY.csv']
WEATHER = ['Beijing','Brasilia','Delhi','Malmo','Miami','Montreal','Singapore','Sydney','Tunis']
NAMES = {'Brasilia':'Brasília','Malmo':'Malmö'}
COLORS = ['#6F9EBC','#AF94AD','#C8A36A','#7FA995','#C78470','#718C92','#A99F62','#9D809A','#7F7F7F']


def render(data_root, output_root, analysis_root=None):
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        return _draw(data_root, output_root, analysis_root)


def _draw(data_root, output_root, analysis_root):
    data_root, output_root = Path(data_root), Path(output_root)
    paths = ([Path(analysis_root) / name for name in RESULT_FILES] if analysis_root
             else [data_root / name for name in FILES])
    nested, choices, curves, domains = [pd.read_csv(p) for p in paths]
    findfont(FontProperties(family='Times New Roman'), fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':FONT_SIZE,'axes.titlesize':12,
                         'axes.titleweight':'bold','axes.labelsize':FONT_SIZE,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.linewidth':0.7,'legend.frameon':False,
                         'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    fig, axes = plt.subplots(2,2,figsize=(8.2,7.0),gridspec_kw={'hspace':0.58,'wspace':0.55})
    fig.subplots_adjust(left=0.11,right=0.97,top=0.94,bottom=0.17)
    a,b,c,d = axes.flat
    selected = nested.set_index('sample').loc[[f'new_{n}_per_source' for n in [3,10,30]]]
    x = np.arange(3)
    a.bar(x,selected.choice_difference_percent,color=COLORS[0],width=0.58)
    for i,row in enumerate(selected.itertuples()):
        a.text(i,row.choice_difference_percent+2,f'{row.different_choices}/{row.settings}',ha='center',va='bottom')
    a.set(xticks=x,xticklabels=['3','10','30'],ylim=(0,65),yticks=[0,20,40,60],
          xlabel='Compositions per source',ylabel='Different selections (%)')
    a.set_title('(a) Population resampling',loc='left',pad=10)

    new = choices[choices.sample_kind.eq('new')]
    counts = new.groupby(['climate','source']).choices_differ.agg(['sum','count'])
    z = np.asarray([[counts.loc[(w,s),'sum'] for s in ['ASHRAE','CTCD']] for w in WEATHER])
    cmap = LinearSegmentedColormap.from_list('counts',['#FAFAF7','#D7E4EA','#6F9EBC'])
    b.imshow(z,vmin=0,vmax=30,cmap=cmap,aspect='auto',interpolation='nearest')
    for i,w in enumerate(WEATHER):
        for j,s in enumerate(['ASHRAE','CTCD']):
            b.text(j,i,f'{z[i,j]}/30',ha='center',va='center',fontsize=10.5)
    b.set(xticks=[0,1],xticklabels=['ASHRAE','CTCD'],yticks=np.arange(9),
          yticklabels=[NAMES.get(w,w) for w in WEATHER])
    b.tick_params(length=0)
    for spine in b.spines.values():
        spine.set_visible(False)
    b.set_title('(b) Differences by weather',loc='left',pad=10)

    c.axvspan(21,26,color='#F0EFE7',zorder=0)
    c.axvline(21,color='#B0AEA2',ls=':',lw=0.8)
    c.axvline(26,color='#B0AEA2',ls=':',lw=0.8)
    for weather,color in zip(WEATHER,COLORS):
        g = curves[curves.climate.eq(weather)].sort_values('target_c')
        c.plot(g.target_c,g.facility_purchased_kwh/g.facility_purchased_kwh.min(),color=color,lw=1.3,label=NAMES.get(weather,weather))
    c.set(xlim=(20,27),xticks=np.arange(20,28),xlabel='Shared command midpoint (°C)',
          ylabel='Annual purchased energy\n(relative to minimum)')
    c.set_title('(c) Shared-command energy',loc='left',pad=10)
    c.text(23.5,0.97,'21–26 °C',transform=c.get_xaxis_transform(),ha='center',va='top',fontsize=10,color='#5D5D57')
    handles,labels = c.get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='lower left',bbox_to_anchor=(0.105,0.01),fontsize=10,
               columnspacing=1.4,handlelength=1.5,handletextpad=0.45)

    width = 0.32
    for offset,row,color in zip([-width/2,width/2],domains.itertuples(),[COLORS[0],COLORS[1]]):
        values = [row.qualifying_set_differences,row.different_choices]
        d.bar(np.arange(2)+offset,values,width=width,color=color,label=row.command_domain)
        for pos,value in zip(np.arange(2)+offset,values):
            d.text(pos,value+1.4,f'{value}/54',ha='center',va='bottom',fontsize=10)
    d.set(xticks=[0,1],xticklabels=['Qualifying\nsets','Lowest-energy\nselections'],ylim=(0,68),yticks=[0,20,40,60],
          ylabel='Settings with differences')
    d.set_title('(d) Command-domain restriction',loc='left',pad=10)
    d.legend(loc='upper center',bbox_to_anchor=(0.5,-0.21),ncol=2,fontsize=10,handlelength=1.2,columnspacing=1.0)
    for axis in [a,c,d]:
        axis.tick_params(width=0.7,length=3,pad=3)
    output_root.mkdir(parents=True,exist_ok=True)
    base = output_root / 'Figure_S03'
    for suffix in ['png','pdf','svg']:
        fig.savefig(base.with_suffix('.'+suffix),dpi=600,bbox_inches='tight',pad_inches=0.035)
    receipt = {'figure':'Figure_S03','panels':'abcd','input_sha256':{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               'formats':['PNG','PDF','SVG'],'png_dpi':600,'font':'Times New Roman'}
    base.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    plt.close(fig)
    return base.with_suffix('.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/figure_data/Tables')
    parser.add_argument('--analysis-root',type=Path,help='Completed analysis results instead of packaged figure tables')
    parser.add_argument('--output-root',type=Path,default=ROOT/'output/publication/supplementary')
    args=parser.parse_args()
    print(render(args.data_root,args.output_root,args.analysis_root))


if __name__=='__main__':
    main()
