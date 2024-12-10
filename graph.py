import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def get_color(value, cmap_name = 'hot', vmin = 0, vmax = 1):
    value = np.clip(value, vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgb = cmap(norm(abs(value)))[:3]
    color = matplotlib.colors.rgb2hex(rgb)
    return color


def plot_cluster(df_cluster, eqp_1, eqp_2):
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot()

    title = ''

    x_c = []
    y_c = []

    for i, row in df_cluster.iterrows():
        eqp = row['EQP_NM']
        warp = row['WARP']

        if eqp in [eqp_1, eqp_2]:
            weight = 'bold'
            color = 'b' if eqp == eqp_1 else 'r'
            fontsize = 15
            title += f'{eqp}: {warp:.2f} um, '
            x_c.append(row['X'])
            y_c.append(row['Y'])

        else:
            weight = None
            color = 'gray'
            fontsize = 10

        ax.text(
            x = row['X'], 
            y = row['Y'], 
            s = row['EQP_NM'], 
            color = color, 
            fontsize = fontsize, 
            ha = 'center',
            va = 'center',
            weight = weight
        )

    ax.add_patch(plt.Circle(
        xy = (np.mean(x_c), np.mean(y_c)), 
        radius = 0.15,
        fc = 'w',
        ec = '#E1002A',
        linestyle = '--',
        linewidth = 2
    ))

    ax.set_title(title.strip(), fontsize = 15)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    return fig


def plot_pie_chart(x, y, title = None):
    fig = plt.figure(figsize = (6, 6))
    plt.title(title)
    plt.pie(y, labels = x, autopct = lambda p : '{:.2f}%'.format(p))
    return fig

def plot_shape(data, eqp_nm, list_lot_id):
    plot_data = data[(data.LOT_ID.isin(list_lot_id)) & (data.EQP_NM == eqp_nm)]

    sub_col_list = ['WARP_BF'] + sorted(list(set(plot_data.columns) & set(map(lambda x: str(float(x))  ,list(range(-145,146))))), key=float)
    
    plot_data_grp = plot_data.groupby(['EQP_NM', 'LOT_ID', 'POS_TYPE','BASE_DT']).agg(dict(zip(sub_col_list, ['mean'] * len(sub_col_list))))
    plot_data_grp.reset_index(inplace=True)
    plot_data_grp['ALPHA'] = plot_data_grp.groupby("POS_TYPE")["BASE_DT"].rank(method="first", ascending=True)
    plot_data_grp['ALPHA'] = plot_data_grp.ALPHA / plot_data_grp.ALPHA.max()

    
    # 색 지정
    color_dict = dict(zip(['SEED', 'MID', 'TAIL'],['royalblue','seagreen','sienna']))
    
    fig, ax = plt.subplots(1, 3, figsize=(10,3), sharex=True, sharey=True)
    warp_data = plot_data_grp.groupby(['POS_TYPE']).WARP_BF.mean().reset_index()
    i = -1
    for POS_TYPE in ['SEED','MID','TAIL']:
        i += 1
        sub_plot_data_grp = plot_data_grp[(plot_data_grp.EQP_NM == eqp_nm) & (plot_data_grp.POS_TYPE == POS_TYPE)]
        
        POS_warp_rank_data = sub_plot_data_grp.WARP_BF.sort_values()
    
        melt_sub_plot_data_grp = pd.melt(sub_plot_data_grp,
                        id_vars=['EQP_NM','LOT_ID','POS_TYPE', 'WARP_BF', 'ALPHA'],
                        value_vars=sub_plot_data_grp.columns[4:].tolist()).sort_values(['EQP_NM','LOT_ID','POS_TYPE'])
        melt_sub_plot_data_grp.columns = ['EQP_NM', 'LOT_ID', 'POS_TYPE', 'WARP_BF', 'ALPHA', 'WF_POSITION', 'value']
        
        melt_sub_plot_data_grp['WF_POSITION'] = melt_sub_plot_data_grp.WF_POSITION.map(lambda x: int(float(x)))
        melt_sub_plot_data_grp_plot = melt_sub_plot_data_grp.groupby('LOT_ID')
        for name, group in melt_sub_plot_data_grp_plot:
            ax[i].plot(group.WF_POSITION,
                group['value'],
                marker='',
                linestyle='solid',
                color=color_dict[POS_TYPE],
                alpha= group.ALPHA.iloc[0],
                label=name 
                )
        ax[i].set_xticks([-144, -100, -50, 0, 50, 100, 144])
        ax[i].grid()
        ax[i].set_ylim([-15, 15])
        ax[i].set_xlabel('WF_POSITION (mm)', fontsize=10)
        ax[i].set_ylabel('SHAPE (um)', fontsize=10)
        if warp_data[warp_data.POS_TYPE == POS_TYPE].WARP_BF.values:
            ax[i].set_title(f"{POS_TYPE} / WARP = {round(warp_data[warp_data.POS_TYPE == POS_TYPE].WARP_BF.values[0],2)} um", position=(0.5, 5), fontsize= 12)

            
    plt.tight_layout()
    return fig
