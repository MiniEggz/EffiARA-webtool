import ast
import json
import os
import zipfile
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from effiara.annotator_reliability import Annotations
from effiara.data_generator import concat_annotations
from effiara.label_generators import DefaultLabelGenerator
from effiara.label_generators.effi_label_generator import EffiLabelGenerator
from effiara.label_generators.topic_label_generator import TopicLabelGenerator
from effiara.preparation import SampleDistributor


def display_annotator_graph_3d(G):
    """Display the annotation graph as an interactive 3D plot in Streamlit."""

    # create 2D networkX crcular layout
    layout_2d = nx.circular_layout(G)
    pos_3d = {}
    for node, (x, y) in layout_2d.items():
        z = G.nodes[node].get("reliability", 0)
        pos_3d[node] = np.array([x, y, z])

    # add edges
    edge_x, edge_y, edge_z = [], [], []
    edge_text = []

    for u, v, d in G.edges(data=True):
        x0, y0, z0 = pos_3d[u]
        x1, y1, z1 = pos_3d[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]
        edge_text.append(f"{u}–{v}: {d.get('agreement', 0):.3f}")

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="black", width=3),
        hoverinfo="text",
        text=edge_text,
    )
    mid_x, mid_y, mid_z, mid_text = [], [], [], []

    for u, v, d in G.edges(data=True):
        x0, y0, z0 = pos_3d[u]
        x1, y1, z1 = pos_3d[v]
        mx, my, mz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
        mid_x.append(mx)
        mid_y.append(my)
        mid_z.append(mz)
        mid_text.append(f"{u} – {v}<br>Agreement: {d.get('agreement', 0):.3f}")

    edge_hover_trace = go.Scatter3d(
        x=mid_x,
        y=mid_y,
        z=mid_z,
        mode="markers",
        marker=dict(
            size=5,
            color="rgba(255,0,0,0.5)",  # invisible marker
        ),
        hoverinfo="text",
        hovertext=mid_text,
        showlegend=False,
    )

    # nodes
    node_x, node_y, node_z = [], [], []
    node_text = []
    hover_text = []

    for node, (x, y, z) in pos_3d.items():
        intra = G.nodes[node].get("intra_agreement", 0)
        reliability = G.nodes[node].get("reliability")
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        hover_text.append(
            f"{node}<br>Intra-agreement: {intra:.3f}<br>Reliability: {reliability:.3f}"
        )

    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers+text",
        text=node_text,
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(
            size=20,
            color=node_z,  # color based on reliability
            colorscale="Blues",
            cmin=min(node_z),
            cmax=max(node_z),
            colorbar=dict(title="Reliability"),
            line=dict(width=2, color="black"),
        ),
        textposition="bottom center",
    )

    layout = go.Layout(
        title="3D Annotator Graph (Reliability as height)",
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(visible=False, showbackground=False, title="X"),
            yaxis=dict(visible=False, showbackground=False, title="Y"),
            zaxis=dict(showbackground=False, title="Reliability (Z)"),
        ),
    )

    fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace], layout=layout)
    st.plotly_chart(fig, use_container_width=True)


def display_annotator_graph(G, text_display="Number"):
    """Display the annotation graph."""
    plt.figure(figsize=(12, 12))
    pos = nx.circular_layout(G, scale=0.9)

    node_size = 3000
    nx.draw_networkx_nodes(G, pos, node_size=node_size)
    nx.draw_networkx_edges(G, pos)
    if text_display == "First Character":
        labels = {node: node[0] for node in G.nodes()}
    elif text_display == "Last Character":
        labels = {node: node[-1] for node in G.nodes()}
    else:
        labels = {node: i + 1 for i, node in enumerate(G.nodes())}
    nx.draw_networkx_labels(G, pos, labels=labels, font_color="white", font_size=24)

    # add inter-annotator agreement to edges
    edge_labels = {(u, v): f"{d['agreement']:.3f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=24)

    # adjust text pos for intra-annotator agreement
    for node, (x, y) in pos.items():
        if x == 0:
            align = "center"
            if y > 0:
                y_offset = 0.15
            else:
                y_offset = -0.15
        elif y == 0:
            align = "center"
            y_offset = 0 if x > 0 else -0.15
        elif x > 0:
            align = "left"
            y_offset = 0.15 if y > 0 else -0.15
        else:
            align = "right"
            y_offset = 0.15 if y > 0 else -0.15

        plt.text(
            x,
            y + y_offset,
            s=f"{G.nodes[node]['intra_agreement']:.3f}",
            horizontalalignment=align,
            verticalalignment="center",
            fontdict={"color": "black", "size": 24},
        )

    # plot
    plt.axis("off")
    st.pyplot(plt.gcf())


def display_agreement_heatmap(
    annotations,
    annotators: Optional[list] = None,
    other_annotators: Optional[list] = None,
    display_upper=False,
):
    """Plot a heatmap of agreement metric values for the annotators.

    If both annotators and other_annotators are specifed, compares
    users in annotators to those in other_annotators. Otherwise,
    compare all project annotators to each other.

    Args:
        annotators (list): Optional.
        other_annotators (list): Optional.

    Returns:
        np.ndarray: A matrix of the data displayed on the graph.
        List[str]: List of annotators in the order of the matrix rows.
    """
    mat = nx.to_numpy_array(annotations.G, weight="agreement")
    # Put intra-agreements on the diagonal
    intras = nx.get_node_attributes(annotations.G, "intra_agreement")
    intras = np.array(list(intras.values()))
    mat[np.diag_indices(mat.shape[0])] = intras
    agreements = annotations.G.nodes(data="avg_inter_agreement")
    if annotators is not None and other_annotators is not None:
        matrows = [
            i for (i, user) in enumerate(annotations.annotators) if user in annotators
        ]
        matcols = [
            i
            for (i, user) in enumerate(annotations.annotators)
            if user in other_annotators
        ]
        # If we're comparing two sets of annotators,
        # slice the agreement matrix.
        mat = mat[matrows][:, matcols]
        agreements = zip(annotators, np.mean(mat, axis=1))

    sorted_by_agreement = sorted(
        enumerate(agreements), key=lambda n: n[1][1], reverse=True
    )
    ordered_row_idxs = [i for (i, _) in sorted_by_agreement]
    mat = mat[ordered_row_idxs]

    # We now have two possible cases.
    #  1) annotators and other_annotators == None: We're comparing
    #     each annotator to each other. In this case we'll display
    #     only the lower triangle of the agreement heatmap as the
    #     the upper triangle will be identical to the lower.
    #  2) otherwise, we're comparing two possibly distinct sets of
    #     annotators, so we display the full matrix, with rows and
    #     columns sliced according to the annotators specified.
    sorted_users = [user for (i, (user, agree)) in sorted_by_agreement]
    if other_annotators is None:
        mat = mat[:, ordered_row_idxs]
        # Don't display upper triangle, since its redundant.
        if not display_upper:
            mat[np.triu_indices(mat.shape[0], k=1)] = np.nan
        xlabs = ylabs = sorted_users
    else:
        xlabs = [user for user in annotations.annotators if user in other_annotators]
        ylabs = sorted_users
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(mat, annot=True, fmt=".3f", xticklabels=xlabs, yticklabels=ylabs, ax=ax)
    st.pyplot(fig)
    return mat, sorted_users


def get_distribution():
    unknown_var = None
    st.write(
        "Input the known parameters from the EffiARA sample distribution equation to create your Sample Distributor."
    )
    st.divider()

    # get annotators
    annotator_knowledge = st.radio(
        "What do you currently know about your annotators?",
        [
            "Number of annotators AND their names/usernames",
            "Number of annotators but NOT their names/usernames",
            "Do not know the number of annotators",
        ],
        index=None,
    )

    num_annotators = None
    if annotator_knowledge == "Number of annotators AND their names/usernames":
        st.divider()
        # allow them to keep adding annotators
        st.write("Enter the names/usernames of annotators:")
        annotators_input = st.text_area(
            "List of annotators - please separate names using a comma (',')"
        )
        if annotators_input:
            annotators = list(
                set([name.strip() for name in annotators_input.split(",")])
            )  # ensure only one instance of each name
            annotators_text = ", ".join(annotators)
            num_annotators = len(annotators)
            st.write(f"Number of Annotators: {num_annotators}")
            st.write(f"Annotators: {annotators_text}")
    elif annotator_knowledge == "Number of annotators but NOT their names/usernames":
        # get number of annotators and generate
        num_annotators = st.number_input(
            "Enter number of annotators:", min_value=1, step=1
        )
        annotators = [f"user_{i}" for i in range(1, num_annotators + 1)]
        annotators_text = ", ".join(annotators)
        st.write(f"Annotators: {annotators_text}")
    elif annotator_knowledge == "Do not know the number of annotators":
        unknown_var = "Number of Annotators"

    if annotator_knowledge is not None and unknown_var is None:
        st.divider()
        # get which variable is unknown
        unknown_var = st.selectbox(
            "Which variable are you solving for?",
            [
                "",
                "Time Available (hours)",
                "Annotation Rate (annotations per hour)",
                "Number of Samples",
                "Proportion of Double Annotations",
                "Proportion of Reannotations",
            ],
        )

    if unknown_var is not None and unknown_var != "":
        # get input parameters
        if unknown_var != "Time Available (hours)":
            time_available = st.number_input(
                "Time Available (hours)", min_value=1.0, step=0.5
            )
        else:
            time_available = None

        if unknown_var != "Annotation Rate (annotations per hour)":
            annotation_rate = st.number_input(
                "Annotation Rate (annotations per hour)", min_value=1.0, step=1.0
            )
        else:
            annotation_rate = None

        if unknown_var != "Number of Samples":
            num_samples = st.number_input("Number of Samples", min_value=1.0, step=1.0)
        else:
            num_samples = None

        if unknown_var != "Proportion of Double Annotations":
            double_proportion = st.slider(
                "Proportion of Double Annotations",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            )
        else:
            double_proportion = None

        if unknown_var != "Proportion of Reannotations":
            reannotation_proportion = st.slider(
                "Proportion of Reannotations", min_value=0.0, max_value=1.0, step=0.01
            )
        else:
            reannotation_proportion = None

        if st.button("Calculate Distribution"):
            st.divider()
            if num_annotators is None:
                annotators = None

            sample_distributor = SampleDistributor(
                annotators=annotators,
                time_available=time_available,
                annotation_rate=annotation_rate,
                num_samples=num_samples,
                double_proportion=double_proportion,
                re_proportion=reannotation_proportion,
            )
            sample_distributor.set_project_distribution()

            st.write("Calculated Sample Distribution:")
            st.write(sample_distributor)

            # save sample_distributor for next step
            st.session_state.sample_distributor = sample_distributor


def distribute_samples():
    st.write(
        "Upload your samples in a CSV file to have them split into annotation projects."
    )
    st.divider()

    if "sample_distributor" not in st.session_state:
        st.write(
            "Sample Distributor not found in your session, please go back to Step 1 to create your sample distributor."
        )
        return

    st.write("Sample distribution in use:")
    st.write(st.session_state.sample_distributor)

    st.divider()

    # take uploaded csv
    uploaded_file = st.file_uploader("Upload CSV file of samples:", type="csv")

    if uploaded_file is not None:
        try:
            # read df
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded CSV:")
            st.dataframe(df.head())

            # check whether to use single only for reannotation
            all_reannotations = st.checkbox(
                "Use double samples for reannotation (this will not follow the timescale as shown in the sample distribution)",
                key="all_reannotations",
            )

            # distribute samples
            with TemporaryDirectory() as temp_dir:
                st.session_state.sample_distributor.distribute_samples(
                    df, temp_dir, all_reannotation=all_reannotations
                )

                all_files = [f"{temp_dir}/{file}" for file in os.listdir(temp_dir)]

                zip_path = os.path.join(temp_dir, "annotation_projects.zip")
                with zipfile.ZipFile(zip_path, "w") as zf:
                    for file in all_files:
                        zf.write(file, os.path.basename(file))

                # allow user to download zipped annotation projects
                with open(zip_path, "rb") as file:
                    st.download_button(
                        "Download Annotation Projects", file, "annotation_projects.zip"
                    )
        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")


def extract_zip_once(uploaded_zip):
    if "extracted_files" in st.session_state:
        return st.session_state.extracted_files, st.session_state.temp_dir

    # create temp directory and store so no extra creations
    temp_dir = TemporaryDirectory()
    temp_dir_path = temp_dir.name

    zip_path = os.path.join(temp_dir_path, "annotation_upload.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getvalue())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir_path)

    # recursively find all csvs
    csv_files = []
    for root, _, files in os.walk(temp_dir_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    st.session_state.extracted_files = csv_files
    st.session_state.temp_dir = temp_dir  # keep reference to prevent cleanup

    return csv_files, temp_dir


def prepare_data():

    # take the zip file
    # upload zip file containing all annotations
    uploaded_zip = st.file_uploader(
        "Upload ZIP containing annotation CSV files", type="zip"
    )

    if uploaded_zip is not None:
        csv_files, _ = extract_zip_once(uploaded_zip)

        st.write("CSV files detected:")
        include_flags = {}
        for full_path in csv_files:
            display_name = os.path.relpath(full_path, os.path.commonpath(csv_files))
            include_flags[full_path] = st.checkbox(
                f"Include {display_name}", value=True
            )

        # get selected files only
        selected_files = [path for path, checked in include_flags.items() if checked]
        st.write("CSV files to be used:")
        selected_display = [os.path.relpath(full_path, os.path.commonpath(csv_files)) for full_path in selected_files]
        st.write(", ".join(selected_display))

        included_files = [f for f, include in include_flags.items() if include]

        # annotator: df
        annotations_dict = {
            os.path.basename(f)[:-4]: pd.read_csv(f) for f in included_files
        }

        user_columns = {user: set(df.columns) for user, df in annotations_dict.items()}
        common_columns = list(set.intersection(*user_columns.values()))
        user_columns = {
            user: list(cols - set(common_columns))
            for user, cols in user_columns.items()
        }

        st.markdown("## Common Columns")
        common_column_actions = {}

        for col in common_columns:
            st.markdown(f"**Column: `{col}`**")

            action = st.selectbox(
                f"What to do with `{col}`?",
                options=["Keep Common", "Make User Specific", "Remove"],
                key=f"action_common_{col}",
            )
            new_name = st.text_input(
                f"Rename `{col}`", value=col, key=f"rename_common_{col}"
            )

            common_column_actions[col] = {"new_name": new_name, "action": action}
            st.divider()

        if st.button("Save Common Column Changes"):
            st.markdown("### Preview: Common Column Changes")
            for col, config in common_column_actions.items():
                action = config["action"]
                new_name = config["new_name"]
                if action == "Keep Common":
                    st.write(
                        f"`{col}` will be kept as **common** and renamed to `{new_name}`."
                    )
                elif action == "Make User Specific":
                    st.write(
                        f"`{col}` will become user-specific: each user will have `[user]_{new_name}`."
                    )
                elif action == "Remove":
                    st.write(f"`{col}` will be **removed**.")

        st.markdown("## User-Specific Columns")
        user_specific_actions = {}

        renaming_mode = st.radio(
            "Column renaming mode",
            options=["Use Template", "Edit Individually"],
            index=0,
        )

        if renaming_mode == "Use Template":
            # build list of user-specific column sets
            user_to_generic = {}
            for user, cols in user_columns.items():
                generic_cols = []
                for col in cols:
                    if user in col:
                        generic_cols.append(col.replace(user, "ANNOTATOR"))
                user_to_generic[user] = generic_cols

            # check if all generic columns are identical across users
            all_generic_sets = [set(gen_cols) for gen_cols in user_to_generic.values()]
            if (
                all(gen_set == all_generic_sets[0] for gen_set in all_generic_sets)
                and all_generic_sets[0]
            ):
                templated_actions = {}
                st.markdown("### Shared Template Columns")

                for templated_col in sorted(all_generic_sets[0]):
                    new_name = st.text_input(
                        f"Rename `{templated_col}`",
                        value=templated_col,
                        key=f"rename_template_{templated_col}",
                    )
                    keep = st.checkbox(
                        f"Keep `{templated_col}`?",
                        value=True,
                        key=f"keep_template_{templated_col}",
                    )
                    templated_actions[templated_col] = {
                        "new_name": new_name,
                        "keep": keep,
                    }

                if st.button("Save Template Column Changes"):
                    st.markdown(
                        "### Preview: User-Specific Column Changes (via template)"
                    )
                    for user, gen_cols in user_to_generic.items():
                        st.write(f"User: `{user}`")
                        for gen_col in gen_cols:
                            if not gen_col:
                                continue  # skip non-template columns
                            action = templated_actions.get(gen_col)
                            if action:
                                if action["keep"]:
                                    final_col = action["new_name"].replace(
                                        "ANNOTATOR", user
                                    )
                                    st.write(
                                        f"`{gen_col.replace('ANNOTATOR', user)}` will be renamed to `{final_col}`."
                                    )
                                else:
                                    st.write(
                                        f"`{gen_col.replace('ANNOTATOR', user)}` will be removed."
                                    )
            else:
                st.warning(
                    "Not all users share the same templated columns. Switching to individual editing."
                )
                renaming_mode = "Edit Individually"

        if renaming_mode == "Edit Individually":
            st.markdown("### User-Specific Columns")
            for user, cols in user_columns.items():
                st.markdown(f"#### User: `{user}`")
                user_specific_actions[user] = {}

                for col in cols:
                    new_name = st.text_input(
                        f"Rename `{col}` (User: {user})",
                        value=col,
                        key=f"rename_{user}_{col}",
                    )
                    keep = st.checkbox(
                        f"Keep `{col}`?", value=True, key=f"keep_{user}_{col}"
                    )
                    user_specific_actions[user][col] = {
                        "new_name": new_name,
                        "keep": keep,
                    }

                st.divider()

            if st.button("Save User-Specific Column Changes"):
                st.markdown("### Preview: User-Specific Column Changes")
                for user, col_dict in user_specific_actions.items():
                    st.write(f"User: `{user}`")
                    for col, config in col_dict.items():
                        if config["keep"]:
                            st.write(
                                f"`{col}` will be renamed to `{config['new_name']}`."
                            )
                        else:
                            st.write(f"`{col}` will be removed.")

        if st.button("Rename Columns"):
            st.session_state.rename_button_press = True

            # rename columns in each df
            for user, df in annotations_dict.items():
                rename_map = {}

                # common columns
                for col, config in common_column_actions.items():
                    if config["action"] == "Keep Common":
                        rename_map[col] = config["new_name"]
                    elif config["action"] == "Make User Specific":
                        new_col = f"{user}_{config['new_name']}"
                        rename_map[col] = new_col
                    elif config["action"] == "Remove":
                        df.drop(columns=[col], inplace=True, errors="ignore")

                # user-specific columns
                if renaming_mode == "Use Template":
                    assert isinstance(templated_actions, dict)
                    for col, action in templated_actions.items():
                        col_name = col.replace("ANNOTATOR", user)
                        if action["keep"]:
                            rename_map[col_name] = action["new_name"].replace(
                                "ANNOTATOR", user
                            )
                        else:
                            df.drop(columns=[col_name], inplace=True)
                else:
                    if user in user_specific_actions:
                        for col, config in user_specific_actions[user].items():
                            if not config["keep"]:
                                df.drop(columns=[col], inplace=True, errors="ignore")
                            else:
                                rename_map[col] = config["new_name"]

                df.rename(columns=rename_map, inplace=True)
            st.success("Columns renamed!")
        if (
            "rename_button_press" in st.session_state
            and st.session_state.rename_button_press
        ):

            # move reannotations to re_ prefix cols
            def move_to_re_cols(df, exclude_cols):
                if "is_reannotation" not in df.columns:
                    return df
                cols_to_prefix = [c for c in df.columns if c not in exclude_cols]
                for col in cols_to_prefix:
                    re_col = f"re_{col}"
                    if re_col not in df.columns:
                        df[re_col] = None
                    df.loc[df["is_reannotation"] == True, re_col] = df.loc[
                        df["is_reannotation"] == True, col
                    ]
                    df.loc[df["is_reannotation"] == True, col] = None
                return df

            rename_reannotations = st.radio(
                "Do you want to move all reannotations to separate columns?",
                ("Yes", "No"),
                index=0,
            )

            if st.button("GO"):
                st.session_state.reanno_button_press = True
                if rename_reannotations == "Yes":
                    exclude_cols = ["sample_id", "is_reannotation"] + [
                        config["new_name"]
                        for col, config in common_column_actions.items()
                        if config["action"] == "Keep Common"
                    ]
                    annotations_dict = {
                        user: move_to_re_cols(df.copy(), exclude_cols)
                        for user, df in annotations_dict.items()
                    }
                    st.write("Reannotations prepended with `re_`!")
                else:
                    st.write("Skipping reannotation column separation.")

            if (
                "reanno_button_press" in st.session_state
                and st.session_state.reanno_button_press
            ):
                if st.button("Merge Dataset"):
                    # merge all dataframes by sample_id
                    st.markdown("## Merging DataFrames...")
                    st.warning("Please be patient, this may take some time.")

                    try:
                        merged_df = concat_annotations(annotations_dict)
                        st.success("Merged dataset created!")
                        st.dataframe(merged_df.head())
                        with NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                            merged_df.to_csv(tmp.name, index=False)
                            tmp_path = tmp.name
                        with open(tmp_path, "rb") as f:
                            st.download_button(
                                label="Download",
                                data=f,
                                file_name="effidataset.csv",
                                mime="text/csv",
                            )
                    except:
                        st.warning(
                            "Cannot merge. Check 'sample_id' column is available for each data point."
                        )


def merge_dataset():
    # take in data
    annotators = None
    if (
        "sample_distributor" in st.session_state
        and st.session_state.sample_distributor is not None
    ):
        annotators = st.session_state.sample_distributor.annotators
    else:
        st.write("Enter the names/usernames of annotators:")
        annotators_input = st.text_area(
            "List of annotators - please separate names using a comma (',')"
        )
        if annotators_input:
            annotators = list(
                set([name.strip() for name in annotators_input.split(",")])
            )  # ensure only one instance of each name
            annotators_text = ", ".join(annotators)
            num_annotators = len(annotators)

    if annotators is not None:
        st.write(f"Number of Annotators: {num_annotators}")
        st.write(f"Annotators: {annotators_text}")
        uploaded_zip = st.file_uploader("Upload zip containing all annotations")

        if uploaded_zip is not None:
            with TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "annotation_upload.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.getvalue())

                # TODO: cleanup
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    for member in zip_ref.namelist():
                        if not member.endswith("/"):
                            extracted_path = zip_ref.extract(member, temp_dir)
                            new_path = os.path.join(temp_dir, os.path.basename(member))
                            os.rename(extracted_path, new_path)

                print(os.listdir(temp_dir))

                annotations_dict = {
                    user: pd.read_csv(f"{temp_dir}/{user}.csv") for user in annotators
                }

                df = concat_annotations(annotations_dict)
                dataset_path = f"{temp_dir}/dataset.csv"
                df.to_csv(f"{temp_dir}/dataset.csv", index=False)

                with open(dataset_path, "rb") as f:
                    st.download_button(
                        label="Download Dataset",
                        data=f,
                        file_name="dataset.csv",
                        mime="text/csv",
                    )


def calculate_annotator_reliability():
    # upload the dataset
    st.divider()
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload Dataset (with columns as specified in the instructions)", type="csv"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data")
        st.dataframe(df.head())

        # TODO: check the dataset is in the correct form

        st.divider()
        st.subheader("Select Label Generator")

        # take the label generator
        label_generators = [
            ("Effi Label Generator", EffiLabelGenerator),
            ("Topic Label Generator", TopicLabelGenerator),
            ("Default Label Generator", DefaultLabelGenerator),
        ]

        selected_name = st.selectbox(
            "Label Generator", [name for name, _ in label_generators]
        )

        selected_generator = next(
            obj for name, obj in label_generators if name == selected_name
        )

        st.divider()
        # get label mapping
        st.subheader("Label Mapping")
        if "label_mapping" not in st.session_state:
            st.session_state.label_mapping = {}

        label_mapping_option = st.radio(
            "How would you like to create your label mapping?",
            [
                "Automatically from your dataset",
                "Add key and values one-by-one",
                "Python dictionary",
            ],
        )

        if label_mapping_option == "Automatically from your dataset":
            st.markdown("##### Set label mapping from your dataset")
            if st.button("Automatic Label Mapping"):
                try:
                    st.session_state.label_mapping = (
                        selected_generator.from_annotations(df).label_mapping
                    )
                    st.success("Label mapping updated!")
                except:
                    st.error(
                        "Failed to create label mapping from dataset, please enter manually."
                    )
        elif label_mapping_option == "Add key and values one-by-one":
            st.markdown("##### Set label mapping one-by-one")
            key_input = st.text_input("Enter key (type will be evaluated)")
            value_input = st.number_input("Enter integer value of the label", step=1)

            if st.button("Add to label mapping"):
                try:
                    if key_input.isnumeric():  # if input is numeric, convert it
                        key = ast.literal_eval(key_input)
                    else:
                        key = key_input.strip()
                    value = int(value_input)
                    st.session_state.label_mapping[key] = value
                    st.success(f"`{key}: {value}` added!")
                except (ValueError, SyntaxError):
                    st.error(
                        "Invalid key or value. Please ensure the key is valid and the value is an integer."
                    )
        else:
            st.markdown("##### Manually set label mapping with Python dictionary")
            dict_input = st.text_area(
                "Enter label mapping dictionary in plain text (e.g., {'key1': 1, 'key2': 2})"
            )
            if st.button("Set label mapping from text"):
                try:
                    new_dict = ast.literal_eval(dict_input)  # safely evaluate the input
                    if isinstance(new_dict, dict) and all(
                        isinstance(v, int) for v in new_dict.values()
                    ):
                        st.session_state.label_mapping = new_dict
                        st.success("Label mapping set successfully.")
                    else:
                        st.error(
                            "Invalid input. Ensure it's a dictionary with integer values."
                        )
                except (ValueError, SyntaxError):
                    st.error(
                        "Invalid dictionary format. Please enter a valid dictionary."
                    )

        st.markdown("### Label Mapping Display")
        if st.button("Clear label mapping"):
            st.session_state.label_mapping = {}
            st.success("Label mapping cleared!")

        st.write(st.session_state.label_mapping)

        st.divider()
        st.subheader("Desired Output")
        display_anno_reliability = st.checkbox("Annotator Reliability", value=True)
        display_anno_graph = st.checkbox("Annotator Agreement Graph", value=True)
        display_anno_heatmap = st.checkbox("Annotator Agreement Heatmap", value=True)

        # get annotators
        st.write("Enter the names/usernames of annotators:")
        try:
            annotators_input = st.text_area(
                "List of annotators - please separate names using a comma (',')",
                value=", ".join(selected_generator.from_annotations(df).annotators),
            )
        except:
            annotators_input = st.text_area(
                "List of annotators - please separate names using a comma (',')"
            )
        if annotators_input:
            annotators = list(
                dict.fromkeys([name.strip() for name in annotators_input.split(",")])
            )
            annotators_text = ", ".join(annotators)
            num_annotators = len(annotators)
            st.write(f"Number of Annotators: {num_annotators}")
            st.write(f"Annotators: {annotators_text}")

        # get agreement metric
        agreement_metrics = [
            "krippendorff",
            "multi_krippendorff",
            "cohen",
            "fleiss",
            "cosine",
        ]
        agreement_metric = st.selectbox("Agreement Metric", agreement_metrics)

        overlap_threshold = int(
            st.number_input("Overlap Threshold (default=15):", step=1, value=15)
        )

        reliability_alpha = float(
            st.number_input(
                "Reliability alpha:", step=0.1, min_value=0.0, max_value=1.0, value=0.5
            )
        )

        contains_reannotations = st.checkbox(
            "Do your annotations contain reannotations (columns begining `re_`)?"
        )

        graph_in_3d = False
        text_display = "Number"
        if display_anno_graph:
            st.markdown("##### Options specific to the agreement graph")
            graph_in_3d = st.checkbox("Display annotator graph in 3D?")
            if not graph_in_3d:
                text_display = st.radio(
                    "What should be displayed on each node? (2D graph only)",
                    ["Number", "First Character", "Last Character"],
                )

        heatmap_annotators = None
        heatmap_other_annotators = None
        display_upper_triangle = False
        if display_anno_heatmap:
            st.markdown("##### Options specific to the agreement heatmap")
            display_specific_annos = st.checkbox(
                "Display specific annotators (rather than all)?"
            )
            first_set_annotators = None
            second_set_annotators = None
            if display_specific_annos:
                first_annotators_input = st.text_area(
                    "List of first set of annotators annotators - please separate names using a comma (',')"
                )
                if first_annotators_input:
                    first_set_annotators = list(
                        dict.fromkeys(
                            [name.strip() for name in first_annotators_input.split(",")]
                        )
                    )
                    first_annotators_text = ", ".join(first_set_annotators)
                    first_num_annotators = len(first_set_annotators)
                    st.write(f"Number of Annotators: {first_num_annotators}")
                    st.write(f"Annotators: {first_annotators_text}")
                second_annotators_input = st.text_area(
                    "List of second set of annotators - please separate names using a comma (',')"
                )
                if second_annotators_input:
                    second_set_annotators = list(
                        dict.fromkeys(
                            [
                                name.strip()
                                for name in second_annotators_input.split(",")
                            ]
                        )
                    )
                    second_annotators_text = ", ".join(second_set_annotators)
                    second_num_annotators = len(second_set_annotators)
                    st.write(f"Number of Annotators: {second_num_annotators}")
                    st.write(f"Annotators: {second_annotators_text}")
                heatmap_annotators = first_set_annotators
                heatmap_other_annotators = second_set_annotators
            display_upper_triangle = st.checkbox(
                "Display redundant upper triangle (improves readability)?"
            )

        st.divider()
        st.subheader("Output")

        # set up the annotations after button pressed
        if st.button("Generate Graphics"):
            st.warning("Generating graphics...")
            try:
                label_generator = selected_generator(
                    annotators=annotators,
                    label_mapping=st.session_state.label_mapping,
                )
                annotations = Annotations(
                    df,
                    label_generator,
                    agreement_metric=agreement_metric,
                    overlap_threshold=overlap_threshold,
                    reliability_alpha=reliability_alpha,
                    reannotations=contains_reannotations,
                )
                annotations.calculate_annotator_reliability()
                st.success("SUCCESS!")

                if display_anno_reliability:
                    st.divider()
                    st.markdown("##### Annotator Reliability")
                    st.write(annotations.get_reliability_dict())

                if display_anno_graph:
                    st.divider()
                    st.markdown("##### Annotator Agreement Graph")
                    if graph_in_3d:
                        display_annotator_graph_3d(annotations.G)
                    else:
                        display_annotator_graph(
                            annotations.G, text_display=text_display
                        )

                if display_anno_heatmap:
                    st.divider()
                    st.markdown("##### Annotator Agreement Heatmap")
                    display_agreement_heatmap(
                        annotations,
                        annotators=heatmap_annotators,
                        other_annotators=heatmap_other_annotators,
                        display_upper=display_upper_triangle,
                    )

            except Exception as e:
                st.error(
                    f"Error found in annotations: {e}\nUnable to calculate reliablility; please check all settings and data."
                )


def gate_teamware_project():
    if "project_display" not in st.session_state:
        st.session_state.project_display = []

    st.write("Scroll to the bottom of this page to see the full project JSON.")
    if st.button("Clear Project"):
        st.session_state.project_display = []

    json_object = json.dumps(st.session_state.project_display, indent=4)

    label_type = st.selectbox("Add", ["", "Text Display", "Data Input"])

    if label_type == "Text Display":
        name = st.text_input(
            "Text display element ID (this just needs to be a unique ID)."
        )
        title = st.text_input("Title to display above your text (Optional)")
        output_column = st.text_input(
            "Name of the column you would like text to be displayed from"
        )
        option_list = ["bold", "border", "big", "bigger"]
        selected_options = st.multiselect(
            "Styling Options", options=option_list, default=[]
        )

        # add some styling to the div based on options

        div_style = "style='"
        if "bold" in selected_options:
            div_style += "font-weight: bold;"
        if "border" in selected_options:
            div_style += "border: 2px solid #000; padding: 10px; text-align: center;"

        # handle font size so they don't both add
        if "bigger" in selected_options:
            div_style += "font-size: 1.5em;"
        elif "big" in selected_options:
            div_style += "font-size: 1.3em;"

        div_style += "'"
        display_text = f"<div {div_style}>{{{{{{{output_column}}}}}}}</div>"
        object_dict = {
            "name": name,
            "title": title,
            "text": display_text,
            "type": "html",
        }

        st.divider()
        st.subheader("Current Object")
        st.write(object_dict)

        if st.button("Add to display"):
            st.session_state.project_display.append(object_dict)
            json_object = json.dumps(st.session_state.project_display, indent=4)
            st.write("Project display after changes")
            st.write(st.session_state.project_display)

        st.divider()
        st.subheader("Project JSON")
        st.text_area("Formatted JSON to Copy", json_object, height=300)
    elif label_type == "Data Input":
        data_type = st.selectbox("Type", ["text", "radio", "checkbox"])
        name = st.text_input(
            "Name of the input field (must be unique and will be used to identify the input value in your dataset)"
        )
        title = st.text_input("Prompt to go above your input")
        description = st.text_area("Any additional description (can leave blank)")
        optional = st.checkbox("Optional")

        if optional:
            title_append = " (Optional)"
        else:
            title_append = ""

        object_dict = {
            "type": data_type,
            "name": name,
            "title": title + title_append,
            "description": description,
            "optional": optional,
        }

        if data_type != "text":
            if "options" not in st.session_state:
                st.session_state.options = []

            object_dict["options"] = st.session_state.options.copy()

            st.divider()
            st.subheader("Add label-value pairs")
            st.write(
                "This allows you to add the label that you would like your annotator to see and the corresponding value you would like stored in your dataset."
            )
            # label
            label = st.text_input("Option label (annotator view)")

            # value
            value = st.text_input("Option value (in dataset)")

            # button to add
            if st.button("Add to options"):
                st.session_state.options.append({"label": label, "value": value})
                object_dict["options"] = st.session_state.options.copy()

            # button to remove
            if st.button("Remove last option"):
                try:
                    st.session_state.options.pop()
                    object_dict["options"] = st.session_state.options.copy()
                except:
                    st.error("Nothing to remove...")
                object_dict["options"] = st.session_state.options.copy()

            if st.button("Clear Options"):
                st.session_state.options = []
                object_dict["options"] = st.session_state.options.copy()

            st.subheader("Current Options")
            st.write(st.session_state.options)

        # display
        st.divider()
        st.subheader("Current Object")
        st.write(object_dict)

        # button to add to display
        if st.button("Add to display", key="Add to display2"):
            st.session_state.project_display.append(object_dict)
            json_object = json.dumps(st.session_state.project_display, indent=4)

        st.divider()
        st.subheader("Project JSON")
        st.text_area("Formatted JSON to Copy", json_object, height=300)


def main():
    st.sidebar.title("Choose Workflow")

    workflows = [
        "Sample Distribution",
        "Annotation Project",
        "Dataset Compilation",
        "Annotator Reliability",
    ]
    workflow = st.sidebar.selectbox("Select a workflow", workflows)

    if workflow == workflows[0]:
        st.sidebar.subheader("Steps of Sample Distribution")
        step = st.sidebar.radio(
            "Go to Step", ["Step 1: Get Distribution", "Step 2: Distribute Samples"]
        )

        st.title("Sample Distribution")

        if step == "Step 1: Get Distribution":
            get_distribution()
        elif step == "Step 2: Distribute Samples":
            distribute_samples()

    elif workflow == workflows[1]:
        st.sidebar.subheader("Platforms for Annotation")
        platform = st.sidebar.radio("Choose a platform", ["GATE Teamware"])

        if platform == "GATE Teamware":
            st.header("Setup Annotation Project Display")
            gate_teamware_project()

    elif workflow == workflows[2]:
        st.sidebar.subheader("Steps of Dataset Compilation")
        step = st.sidebar.radio(
            "Go to Step", ["Prepare Data"]
        )

        # TODO: add instructions

        if step == "Prepare Data":
            prepare_data()

    elif workflow == workflows[3]:
        st.sidebar.subheader("Calculate Annotator Reliability")
        # TODO: add instructions
        calculate_annotator_reliability()


if __name__ == "__main__":
    main()
