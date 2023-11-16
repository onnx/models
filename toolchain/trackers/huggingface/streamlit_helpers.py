from collections import Counter
from typing import List
import numpy as np
import streamlit as st  # pylint: disable=import-error
import pandas as pd


class Collapsable:
    """
    Creates a collapsable text composed of a preamble (clickable section of text)
    and epilogue (collapsable text).
    """

    def __init__(self, preamble="", epilogue=""):
        self.preamble = preamble
        self.epilogue = epilogue
        self.small_font = 18
        self.large_font = 18
        self.sections = []

    def add_section(self, heading, text):
        # Convert text to bullet points if it is a list
        if isinstance(text, list):
            text = (
                "<ul>"
                + "".join(
                    [
                        f'<li style="font-size:{self.small_font}px;" align="justify">{x}</li>'
                        for x in text
                    ]
                )
                + "</ul>"
            )

        # Append section
        self.sections.append((heading, text))

    def deploy(self):

        secs = "".join(
            [
                (
                    "<details>"
                    f"<summary style='font-size:{self.large_font}px;'>{heading}</summary>"
                    f"<blockquote style='font-size:{self.small_font}px;max-width: 80%;'"
                    f"align='justify'>{text}</details>"
                )
                for heading, text in self.sections
            ]
        )
        collapsable_sec = f"""
        <ol>
        {self.preamble}
        {secs}
        {self.epilogue}
        </ol>
        """
        st.markdown(collapsable_sec, unsafe_allow_html=True)


def add_filter(
    data_frame_list: List[pd.DataFrame],
    name: str,
    label: str,
    options: List[str] = None,
    num_cols: int = 1,
    last_is_others: bool = True,
):
    """
    Creates a filter on the side bar using checkboxes
    """

    # Get list of all options and return if no options are available
    all_options = set(data_frame_list[-1][label])
    if "-" in all_options:
        all_options.remove("-")
    if len(all_options) == 0:
        return data_frame_list

    st.markdown(f"#### {name}")

    # Create list of options if selectable options are not provided
    if options is None:
        options_dict = Counter(data_frame_list[-1][label])
        sorted_options = sorted(options_dict, key=options_dict.get, reverse=True)
        if "-" in sorted_options:
            sorted_options.remove("-")
        if len(sorted_options) > 8:
            options = list(sorted_options[:7]) + ["others"]
            last_is_others = True
        else:
            options = list(sorted_options)
            last_is_others = False

    cols = st.columns(num_cols)
    instantiated_checkbox = []
    for idx in range(len(options)):
        with cols[idx % num_cols]:
            instantiated_checkbox.append(
                st.checkbox(options[idx], False, key=f"{label}_{options[idx]}")
            )

    selected_options = [
        options[idx] for idx, checked in enumerate(instantiated_checkbox) if checked
    ]

    # The last checkbox will always correspond to "other"
    if instantiated_checkbox[-1] and last_is_others:
        selected_options = selected_options[:-1]
        other_options = [x for x in all_options if x not in options]
        selected_options = set(selected_options + other_options)

    if len(selected_options) > 0:
        for idx, _ in enumerate(data_frame_list):
            data_frame_list[idx] = data_frame_list[idx][
                [
                    any([x == model_entry for x in selected_options])
                    for model_entry in data_frame_list[idx][label]
                ]
            ]
    return data_frame_list


def slider_filter(
    data_frame_list: List[pd.DataFrame],
    title: str,
    filter_by: str,
    max_val: int = 1000,
):
    """
    Creates slider to filter dataframes according to a given label.
    label must be numeric. Values are in millions.
    """

    start_val, end_val = st.select_slider(
        title,
        options=[str(x) for x in np.arange(0, max_val + 1, 10, dtype=int)],
        value=("0", str(max_val)),
    )

    for idx in range(len(data_frame_list)):
        data_frame_list[idx] = data_frame_list[idx][
            [
                int(model_entry) >= int(start_val) * 1000000
                and int(model_entry) <= int(end_val) * 1000000
                for model_entry in data_frame_list[idx][filter_by]
            ]
        ]

    return data_frame_list
