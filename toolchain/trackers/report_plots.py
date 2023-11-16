import plotly.graph_objects as go
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np

df = pd.read_csv(r'C:\Users\danie\onnxmodelzoo\toolchain\models\timm\2023-08-30.csv')

colors = {
    "blue": "#5470c6",
    "orange": "#FF7F0E",
    "green": "#94cc74",
    "saffron_mango": "#fac858",
    "red": "#ee6666",
    "light_blue": "#73c0de",
    "ocean_green": "#3ba272",
}


def throughput_acceleration(df):
    vitisep_results = df[df["runtime"] == "vitisep"]
    ort_results = df[df["runtime"] == "ort"]
    assert len(vitisep_results) == len(ort_results)
    on_ipu = vitisep_results.ipu_compilation_successful.to_numpy()
    ratio = vitisep_results.throughput.to_numpy()/ort_results.throughput.to_numpy()

    y0 = [ratio[idx] for idx in range(len(ratio)) if on_ipu[idx] == 'True']
    y1 = [ratio[idx] for idx in range(len(ratio)) if on_ipu[idx] == 'False']
    y2 = np.concatenate ([y0,y1])

    y0_label = ["Yes"]*len(y0)
    y1_label = ["No"]*len(y1)
    y2_label = y0_label+y1_label   

    df = pd.DataFrame({'graph_name':['Running on IPU']*len(y0)+['Fallback to CPU']*len(y1)+['All models']*len(y2),
                    'value': np.concatenate([y0,y1,y2],0),
                    'Actually running on the IPU?':y0_label+y1_label+y2_label}
                    )

    fig = px.strip(df,
            x='graph_name',
            y='value',
            color='Actually running on the IPU?',
            stripmode='overlay')

    fig.add_trace(go.Box(y=df.query('graph_name == "Running on IPU"')['value'], name='Running on IPU',marker=dict(opacity=0.1)))
    fig.add_trace(go.Box(y=df.query('graph_name == "Fallback to CPU"')['value'], name='Fallback to CPU'))
    fig.add_trace(go.Box(y=df.query('graph_name == "All models"')['value'], name='All models'))

    fig.update_layout(autosize=False,
                    legend={'traceorder':'normal'},
                    )
    fig.update_yaxes(title_text="Acceleration compared to OnnxRuntime CPU EP")
    fig.update_xaxes(title_text="")
    fig.show()

def parameter_histogram(df: pd.DataFrame) -> None:
    # Add parameters histogram
    all_models = [float(x) / 1000000 for x in df[df["runtime"] == "vitisep"]["parameters"] if x != "-"]

    hist_data = []
    group_labels = []

    if all_models != []:
        hist_data.append(all_models)
        group_labels.append("All models")


    if hist_data:
        fig = ff.create_distplot(
            hist_data,
            group_labels,
            bin_size=5,
            histnorm="",
            colors=list(colors.values()),
            curve_type="normal",
        )
        fig.update_layout(showlegend=False)
        fig.layout.update(xaxis_title="Parameters in millions")
        fig.layout.update(yaxis_title="Models inside bin")
        fig.update_xaxes(range=[1, 200])

        fig.show()

def throughput_plot(df):
    vitisep_results = df[df["runtime"] == "vitisep"]
    ort_results = df[df["runtime"] == "ort"]

    fig = go.Figure(data=[
        go.Bar(name='VitisEP', x=vitisep_results.model_name, y=vitisep_results.throughput),
        go.Bar(name='OnnxRuntime CPU EP', x=ort_results.model_name, y=ort_results.throughput)
    ])

    # Set x and y axis labels
    fig.update_layout(
        barmode='group',
        xaxis_title="",
        yaxis_title="Throughput"
    )
    fig.show()



def compilation_time(df):
    # Add compilation time histogram
    all_models = [float(x) for x in df[df["runtime"] == "vitisep"]["ipu_compilation_seconds"] if x != "-"]

    hist_data = []
    group_labels = []

    hist_data.append(all_models)
    group_labels.append("All models")

    if hist_data:
        fig = ff.create_distplot(
            hist_data,
            group_labels,
            bin_size=5,
            histnorm="",
            colors=list(colors.values()),
            curve_type="normal",
        )
        fig.update_layout(showlegend=False)
        fig.layout.update(xaxis_title="Compilation time in seconds")
        fig.layout.update(yaxis_title="Models inside bin")

        fig.show()

parameter_histogram(df)
throughput_plot(df)
throughput_acceleration(df)
compilation_time(df)