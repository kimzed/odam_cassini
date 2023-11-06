
import pandas as pd
from plotly.subplots import make_subplots

def analytics_fig():
    precipitation_csv = 'frontend/data/precipitation.csv'
    evatransporation_csv = 'frontend/data/evapotranspiration.csv'
    temperature_csv = 'frontend/data/temperature.csv'
    downstream_csv = 'frontend/data/downstream.csv'
    upstream_csv = 'frontend/data/upstream.csv'

    downstream_df = pd.read_csv(downstream_csv)
    upstream_df = pd.read_csv(upstream_csv)
    precipitation_df = pd.read_csv(precipitation_csv)
    temperature_df = pd.read_csv(temperature_csv)
    evatransporation_df = pd.read_csv(evatransporation_csv)

    import plotly.graph_objects as go

    results_HBV = temperature_df
    results_GR6J = temperature_df

    # plot
    font_axis  = dict(size = 16)
    font_l     = dict(size = 13)
    font_ticks = dict(size = 14)

    col0 = "#000000"
    col1 = "#c51b7d"
    col2 = "#de77ae"
    col3 = "#7fbc41"
    col4 = "#4d9221"
    col5 = "#8dd3c7"
    col6 = "#fed98e"
    col7 = "#bebada"
    col8 = "#fb8072"

    temp_col = '#ca0020'
    et_col = '#f4a582'
    precip_col = '#92c5de'
    inflow_col = '#0571b0'
    outflow_col = '#bababa'

    l_width = 1.5

    
    fig = make_subplots(rows=4, cols=1)

    fig.add_trace(
        go.Scatter(x=precipitation_df['Date'], y=precipitation_df['Obs'], mode='lines', name='Precipitation', line=dict(color=precip_col, width=2)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=evatransporation_df['Date'], y=evatransporation_df['Obs'], mode='lines', name='Evatransporation', line=dict(color=et_col, width=2)),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=temperature_df['Date'], y=temperature_df['Obs'], mode='lines', name='Temperature', line=dict(color=temp_col, width=2)),
        row=3, col=1,
    )

    fig.add_trace(
        go.Scatter(x=upstream_df['Date'], y=upstream_df['Obs'], mode='lines', name='Inflow', line=dict(color=inflow_col, width=2)),
        row=4, col=1
    )

    fig.add_trace(
        go.Scatter(x=downstream_df['Date'], y=downstream_df['Obs'], mode='lines', name='Outflow', line=dict(color=outflow_col, width=2)),
        row=4, col=1
    )

    # Add vertical line at the year 1976 for each subplot
    datas = [precipitation_df['Obs'], evatransporation_df['Obs'], temperature_df['Obs'], upstream_df['Obs'], downstream_df['Obs']]
    for i in range(1, 5):
        fig.add_shape(
            type="line",
            x0="1976-01-01", y0=min(datas[i-1]),
            x1="1976-01-01", y1=max(datas[i-1]),
            line=dict(
                color="Black",
                width=3,
            ),
            xref="x"+str(i), yref="paper",
            row=i, col=1
        )

    # Update layout and axes
    fig.update_layout(
        title_text="Temperature, Precipitation and Evatransporation",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        title=dict(
            x=0.5,
        ),
    )

    # Update x-axes for all subplots
    fig.update_xaxes(
        zeroline=True,
        showline=True,
        mirror=True,
        ticks='outside',
        tickmode='linear',
        dtick='M12',
        tickformat='%Y',
        showticklabels=True,
        linecolor='black',
        linewidth=2,
        tickfont=dict(
            size=10,
        ),
        row='all', col='all'  # Apply settings to all rows and columns
    )

    # Update y-axes for all subplots
    fig.update_yaxes(
        showticklabels=True,
        zeroline=True,
        showline=True,
        mirror=True,
        ticks='outside',
        linecolor='black',
        linewidth=2,
        row='all', col='all'  # Apply settings to all rows and columns
    )

    # Add bounding box to each subplot
    y_titles = ['', '', '', '', 'Volume', '']
    for i in range(1, 5):
        fig['layout']['xaxis{}'.format(i)].update(
            showline=True, linewidth=2, linecolor='black', mirror=True
        )
        #fig['layout']['xaxis{}'.format(i)].update(title='Time-X')

        fig['layout']['yaxis{}'.format(i)].update(
            showline=True, linewidth=2, linecolor='black', mirror=True
        )
        fig['layout']['yaxis{}'.format(i)].update(title=y_titles[i])

    fig.update_layout(
        showlegend=True,
    )
    return fig