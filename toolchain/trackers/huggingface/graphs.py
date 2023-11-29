from collections import Counter
from streamlit_echarts import st_echarts  # pylint: disable=import-error
import numpy as np
import pandas as pd
import streamlit as st  # pylint: disable=import-error
import plotly.figure_factory as ff
from plotly import graph_objs as go
import plotly.express as px
from statistics import median

colors = {
    "blue": "#5470c6",
    "orange": "#FF7F0E",
    "green": "#94cc74",
    "saffron_mango": "#fac858",
    "red": "#ee6666",
    "light_blue": "#73c0de",
    "ocean_green": "#3ba272",
}
device_colors = {
    "x86": colors["blue"],
    "nvidia": colors["green"],
}


class StageCount:
    def __init__(self, df: pd.DataFrame) -> None:
        self.all_models = len(df)
        self.base_onnx = int(np.sum(df["base_onnx"]))
        self.optimize_fp32 = int(np.sum(df["optimize_fp32"]))
        self.all_ops_supported = int(np.sum(df["all_ops_supported"]))
        self.fp16_onnx = int(np.sum(df["fp16_onnx"]))
        self.compiles = int(np.sum(df["compiles"]))
        self.assembles = int(np.sum(df["assembles"]))


class DeviceStageCount:
    def __init__(self, df: pd.DataFrame) -> None:
        self.all_models = len(df)
        self.base_onnx = int(np.sum(df["onnx_exported"]))
        self.optimize_fp32 = int(np.sum(df["onnx_optimized"]))
        self.fp16_onnx = int(np.sum(df["onnx_converted"]))
        self.x86 = df.loc[df.x86_latency != "-", "x86_latency"].count()
        self.nvidia = df.loc[df.nvidia_latency != "-", "nvidia_latency"].count()


def stages_count_summary(current_df: pd.DataFrame, prev_df: pd.DataFrame) -> None:
    """
    Show count of how many models compile, assemble, etc
    """
    current = StageCount(current_df)
    prev = StageCount(prev_df)

    kpi = st.columns(7)

    kpi[0].metric(
        label="All models",
        value=current.all_models,
        delta=current.all_models - prev.all_models,
    )

    kpi[1].metric(
        label="Converts to ONNX",
        value=current.base_onnx,
        delta=current.base_onnx - prev.base_onnx,
    )

    kpi[2].metric(
        label="Optimizes ONNX file",
        value=current.optimize_fp32,
        delta=current.optimize_fp32 - prev.optimize_fp32,
    )

    kpi[3].metric(
        label="Supports all ops",
        value=current.all_ops_supported,
        delta=current.all_ops_supported - prev.all_ops_supported,
    )

    kpi[4].metric(
        label="Converts to FP16",
        value=current.fp16_onnx,
        delta=current.fp16_onnx - prev.fp16_onnx,
    )

    kpi[5].metric(
        label="Compiles",
        value=current.compiles,
        delta=current.compiles - prev.compiles,
    )

    kpi[6].metric(
        label="Assembles",
        value=current.assembles,
        delta=current.assembles - prev.assembles,
    )

    # Show Sankey graph with percentages
    sk_val = {
        "All models": "100%",
        "Converts to ONNX": str(int(100 * current.base_onnx / current.all_models))
        + "%",
        "Optimizes ONNX file": str(
            int(100 * current.optimize_fp32 / current.all_models)
        )
        + "%",
        "Supports all ops": str(
            int(100 * current.all_ops_supported / current.all_models)
        )
        + "%",
        "Converts to FP16": str(int(100 * current.fp16_onnx / current.all_models))
        + "%",
        "Compiles": str(int(100 * current.compiles / current.all_models)) + "%",
        "Assembles": str(int(100 * current.assembles / current.all_models)) + "%",
    }
    option = {
        "series": {
            "type": "sankey",
            "animationDuration": 1,
            "top": "0%",
            "bottom": "20%",
            "left": "0%",
            "right": "13.5%",
            "darkMode": "true",
            "nodeWidth": 2,
            "textStyle": {"fontSize": 16},
            "lineStyle": {"curveness": 0},
            "layoutIterations": 0,
            "layout": "none",
            "emphasis": {"focus": "adjacency"},
            "data": [
                {
                    "name": "All models",
                    "value": sk_val["All models"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Converts to ONNX",
                    "value": sk_val["Converts to ONNX"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Optimizes ONNX file",
                    "value": sk_val["Optimizes ONNX file"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Supports all ops",
                    "value": sk_val["Supports all ops"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Converts to FP16",
                    "value": sk_val["Converts to FP16"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Compiles",
                    "value": sk_val["Compiles"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Assembles",
                    "value": sk_val["Assembles"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
            ],
            "label": {
                "position": "insideTopLeft",
                "borderWidth": 0,
                "fontSize": 16,
                "color": "white",
                "textBorderWidth": 0,
                "formatter": "{c}",
            },
            "links": [
                {
                    "source": "All models",
                    "target": "Converts to ONNX",
                    "value": current.base_onnx,
                },
                {
                    "source": "Converts to ONNX",
                    "target": "Optimizes ONNX file",
                    "value": current.optimize_fp32,
                },
                {
                    "source": "Optimizes ONNX file",
                    "target": "Supports all ops",
                    "value": current.all_ops_supported,
                },
                {
                    "source": "Supports all ops",
                    "target": "Converts to FP16",
                    "value": current.fp16_onnx,
                },
                {
                    "source": "Converts to FP16",
                    "target": "Compiles",
                    "value": current.compiles,
                },
                {
                    "source": "Compiles",
                    "target": "Assembles",
                    "value": current.assembles,
                },
            ],
        }
    }
    st_echarts(
        options=option,
        height="50px",
    )


def workload_origin(df: pd.DataFrame) -> None:
    """
    Show pie chart that groups models by author
    """
    all_authors = list(df.loc[:, "author"])
    author_count = {i: all_authors.count(i) for i in all_authors}
    all_models = len(df)

    options = {
        "darkMode": "true",
        "textStyle": {"fontSize": 16},
        "tooltip": {"trigger": "item"},
        "series": [
            {  # "Invisible" chart, used to show author labels
                "name": "Name of corpus:",
                "type": "pie",
                "radius": ["70%", "70%"],
                "data": [
                    {"value": author_count[k], "name": k} for k in author_count.keys()
                ],
                "label": {
                    "formatter": "{b}\n{d}%",
                },
            },
            {
                # Actual graph where data is shown
                "name": "Name of corpus:",
                "type": "pie",
                "radius": ["50%", "70%"],
                "data": [
                    {"value": author_count[k], "name": k} for k in author_count.keys()
                ],
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0, 0, 0, 0.5)",
                    }
                },
                "label": {
                    "position": "inner",
                    "formatter": "{c}",
                    "color": "black",
                    "textBorderWidth": 0,
                },
            },
            {
                # Show total number of models inside
                "name": "Total number of models:",
                "type": "pie",
                "radius": ["0%", "0%"],
                "data": [{"value": all_models, "name": "Total"}],
                "silent": "true",
                "label": {
                    "position": "inner",
                    "formatter": "{c}",
                    "color": "white",
                    "fontSize": 30,
                    "textBorderWidth": 0,
                },
            },
        ],
    }
    st_echarts(
        options=options,
        height="400px",
    )


def parameter_histogram(df: pd.DataFrame, show_assembled=True) -> None:
    # Add parameters histogram
    all_models = [float(x) / 1000000 for x in df["params"] if x != "-"]

    hist_data = []
    group_labels = []

    if all_models != []:
        hist_data.append(all_models)
        if show_assembled:
            group_labels.append("Models we tried compiling")
        else:
            group_labels.append("All models")

    if show_assembled:
        assembled_models = df[
            df["assembles"] == True  # pylint: disable=singleton-comparison
        ]
        assembled_models = [
            float(x) / 1000000 for x in assembled_models["params"] if x != "-"
        ]
        if assembled_models != []:
            hist_data.append(assembled_models)
            group_labels.append("Assembled models")

    if hist_data:
        fig = ff.create_distplot(
            hist_data,
            group_labels,
            bin_size=25,
            histnorm="",
            colors=list(colors.values()),
            curve_type="normal",
        )
        fig.layout.update(xaxis_title="Parameters in millions")
        fig.layout.update(yaxis_title="count")
        fig.update_xaxes(range=[1, 1000])

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown(
            """At least one model needs to reach the compiler to show this graph 😅"""
        )


def process_latency_data(df, baseline):
    df = df[["model_name", "nvidia_latency", "x86_latency"]]
    df = df.sort_values(by=["model_name"])

    df.x86_latency.replace(["-"], [float("inf")], inplace=True)
    df.nvidia_latency.replace(["-"], [float("inf")], inplace=True)

    df["nvidia_latency"] = df["nvidia_latency"].astype(float)
    df["x86_latency"] = df["x86_latency"].astype(float)

    df["nvidia_compute_ratio"] = df[f"{baseline}_latency"] / df["nvidia_latency"]
    df["x86_compute_ratio"] = df[f"{baseline}_latency"] / df["x86_latency"]

    return df


def speedup_bar_chart(df: pd.DataFrame, baseline) -> None:

    if len(df) == 0:
        st.markdown(
            ("Nothing to show here since no models have been successfully benchmarked.")
        )
    else:
        df = process_latency_data(df, baseline)
        bar_chart = {}
        bar_chart["nvidia"] = go.Bar(
            x=df["model_name"],
            y=df["nvidia_compute_ratio"],
            name="NVIDIA A100",
        )
        bar_chart["x86"] = go.Bar(
            x=df["model_name"],
            y=df["x86_compute_ratio"],
            name="Intel(R) Xeon(R)",
        )

        # Move baseline to the back of the plot
        plot_sequence = list(bar_chart.keys())
        plot_sequence.insert(0, plot_sequence.pop(plot_sequence.index(baseline)))

        # Ensure that the baseline is the last bar
        data = [bar_chart[device_type] for device_type in plot_sequence]
        color_sequence = [device_colors[device_type] for device_type in plot_sequence]

        layout = go.Layout(
            barmode="overlay",  # group
            legend={
                "orientation": "h",
                "xanchor": "center",
                "x": 0.5,
                "y": 1.2,
            },
            yaxis_title="Latency Speedup",
            colorway=color_sequence,
            height=500,
        )

        fig = dict(data=data, layout=layout)
        st.plotly_chart(fig, use_container_width=True)


def kpi_to_markdown(
    compute_ratio, device, num_baseline_models, is_baseline=False, color="blue"
):

    if is_baseline:
        title = f"""<br><br>
        <p style="font-family:sans-serif; font-size: 20px;text-align: center;">Median {device} Acceleration ({len(compute_ratio)} models):</p>"""
        return (
            title
            + f"""<p style="font-family:sans-serif; color:{colors[color]}; font-size: 26px;text-align: center;"> {1}x (Baseline)</p>"""
        )

    title = f"""<br><br>
    <p style="font-family:sans-serif; font-size: 20px;text-align: center;">Median {device} Acceleration ({len(compute_ratio)}/{num_baseline_models} models):</p>"""

    if len(compute_ratio) > 0:
        kpi_min, kpi_median, kpi_max = (
            round(compute_ratio.min(), 2),
            round(median(compute_ratio), 2),
            round(compute_ratio.max(), 2),
        )
    else:
        kpi_min, kpi_median, kpi_max = 0, 0, 0

    return (
        title
        + f"""<p style="font-family:sans-serif; color:{colors[color]}; font-size: 26px;text-align: center;"> {kpi_median}x</p>
    <p style="font-family:sans-serif; color:{colors[color]}; font-size: 20px;text-align: center;"> min {kpi_min}x; max {kpi_max}x</p>
    """
    )


def speedup_text_summary(df: pd.DataFrame, baseline) -> None:

    df = process_latency_data(df, baseline)

    # Some latencies are "infinite" because they could not be calculated
    # To calculate statistics, we remove all elements of df where the baseline latency is inf
    df = df[(df[baseline + "_latency"] != float("inf"))]

    # Setting latencies that could not be calculated to infinity also causes some compute ratios to be zero
    # We remove those to avoid doing any calculations with infinite latencies
    x86_compute_ratio = df["x86_compute_ratio"].to_numpy()
    nvidia_compute_ratio = df["nvidia_compute_ratio"].to_numpy()
    x86_compute_ratio = x86_compute_ratio[x86_compute_ratio != 0]
    nvidia_compute_ratio = nvidia_compute_ratio[nvidia_compute_ratio != 0]

    num_baseline_models = len(df[f"{baseline}_compute_ratio"])
    x86_text = kpi_to_markdown(
        x86_compute_ratio,
        device="Intel(R) Xeon(R) X40 CPU @ 2.00GHz",
        num_baseline_models=num_baseline_models,
        color="blue",
        is_baseline=baseline == "x86",
    )
    nvidia_text = kpi_to_markdown(
        nvidia_compute_ratio,
        device="NVIDIA A100-PCIE-40GB",
        num_baseline_models=num_baseline_models,
        color="green",
        is_baseline=baseline == "nvidia",
    )

    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"""{x86_text}""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""{nvidia_text}""", unsafe_allow_html=True)


def compiler_errors(df: pd.DataFrame) -> None:
    compiler_errors = df[df["compiler_error"] != "-"]["compiler_error"]
    compiler_errors = Counter(compiler_errors)
    if len(compiler_errors) > 0:
        compiler_errors = pd.DataFrame.from_dict(
            compiler_errors, orient="index"
        ).reset_index()
        compiler_errors = compiler_errors.set_axis(
            ["error", "count"], axis=1, inplace=False
        )
        compiler_errors["error"] = [ce[:80] for ce in compiler_errors["error"]]
        fig = px.bar(
            compiler_errors,
            x="count",
            y="error",
            orientation="h",
            height=400,
        )
        fig.update_traces(marker_color=colors["blue"])

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""No compiler errors found :tada:""")


def results_table(df: pd.DataFrame):
    model_name = st.text_input("", placeholder="Filter model by name")
    if model_name != "":
        df = df[[model_name in x for x in df["Model Name"]]]

    st.dataframe(df, height=min((len(df) + 1) * 35, 35 * 21))


def device_funnel_metrics(num_models: int, num_total_models: int) -> str:
    """
    Calculates the percentage between models and total_models
    Avoids ZeroDivisionError when dividend is zero
    """
    models_message = f"{num_models} model"
    models_message = models_message + "s" if num_models != 1 else models_message
    percentage_message = ""
    if num_total_models > 0:
        model_ratio = num_models / num_total_models
        if model_ratio < 0.01 and model_ratio != 0:
            percentage_message = " - < 1%"
        else:
            percentage_message = f" - {int(100*num_models / num_total_models)}%"
    return f"{models_message}{percentage_message}"


def device_funnel(df: pd.DataFrame) -> None:
    """
    Show count of how many models compile, assemble, etc
    """
    summ = DeviceStageCount(df)

    stages = [
        "All models",
        "Export to ONNX",
        "Optimize ONNX file",
        "Convert to FP16",
        "Acquire Performance",
    ]
    cols = st.columns(len(stages))

    for idx, stage in enumerate(stages):
        with cols[idx]:
            st.markdown(stage)

    # Show Sankey graph with percentages
    sk_val = {
        "All models": device_funnel_metrics(summ.all_models, summ.all_models),
        "Converts to ONNX": device_funnel_metrics(summ.base_onnx, summ.all_models),
        "Optimizes ONNX file": device_funnel_metrics(
            summ.optimize_fp32, summ.all_models
        ),
        "Converts to FP16": device_funnel_metrics(summ.fp16_onnx, summ.all_models),
        "Acquires Nvidia Perf": device_funnel_metrics(summ.nvidia, summ.all_models)
        + " (Nvidia)",
        "Acquires x86 Perf": device_funnel_metrics(summ.x86, summ.all_models)
        + " (x86)",
    }

    # Calculate bar heights for each of the devices
    # Bar height is proportional to the number of models benchmarked by each device
    default_bar_size = 1
    target_combined_height = max(default_bar_size, summ.fp16_onnx)
    device_bar_size = target_combined_height / 3

    option = {
        "series": {
            "type": "sankey",
            "animationDuration": 1,
            "top": "0%",
            "bottom": "20%",
            "left": "0%",
            "right": "19%",
            "darkMode": "true",
            "nodeWidth": 2,
            "textStyle": {"fontSize": 16},
            "nodeAlign": "left",
            "lineStyle": {"curveness": 0},
            "layoutIterations": 0,
            "nodeGap": 12,
            "layout": "none",
            "emphasis": {"focus": "adjacency"},
            "data": [
                {
                    "name": "All models",
                    "value": sk_val["All models"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Converts to ONNX",
                    "value": sk_val["Converts to ONNX"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Optimizes ONNX file",
                    "value": sk_val["Optimizes ONNX file"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Converts to FP16",
                    "value": sk_val["Converts to FP16"],
                    "itemStyle": {"color": "white", "borderColor": "white"},
                },
                {
                    "name": "Acquires Nvidia Perf",
                    "value": sk_val["Acquires Nvidia Perf"],
                    "itemStyle": {
                        "color": device_colors["nvidia"],
                        "borderColor": device_colors["nvidia"],
                    },
                },
                {
                    "name": "Acquires x86 Perf",
                    "value": sk_val["Acquires x86 Perf"],
                    "itemStyle": {
                        "color": device_colors["x86"],
                        "borderColor": device_colors["x86"],
                    },
                },
            ],
            "label": {
                "position": "insideTopLeft",
                "borderWidth": 0,
                "fontSize": 16,
                "color": "white",
                "textBorderWidth": 0,
                "formatter": "{c}",
            },
            "links": [
                {
                    "source": "All models",
                    "target": "Converts to ONNX",
                    "value": max(default_bar_size, summ.all_models),
                },
                {
                    "source": "Converts to ONNX",
                    "target": "Optimizes ONNX file",
                    "value": max(default_bar_size, summ.optimize_fp32),
                },
                {
                    "source": "Optimizes ONNX file",
                    "target": "Converts to FP16",
                    "value": max(default_bar_size, summ.fp16_onnx),
                },
                {
                    "source": "Converts to FP16",
                    "target": "Acquires Nvidia Perf",
                    "value": device_bar_size,
                },
                {
                    "source": "Converts to FP16",
                    "target": "Acquires x86 Perf",
                    "value": device_bar_size,
                },
            ],
        }
    }
    st_echarts(
        options=option,
        height="70px",
    )
