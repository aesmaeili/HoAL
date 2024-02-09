""" Some utility tools to use"""
import torch
import subprocess
from data_tools import prepare_sys_data
import holonic_ml_avg
import torch.utils.tensorboard as tb
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from hl_configs import Globals
from matplotlib.ticker import ScalarFormatter

sns.set_context("notebook")



def config_logger(logger, log_file, log_level):
    logger.remove()  # to prevent logging on terminal (stderr)
    subprocess.run(
        ["rm", "-rf", log_file]
    )  # removing the log file from previous experiments
    logger.add(
        log_file, level=log_level
    )  # to continue loggings debugs reset the level back to DEBUG
    return logger


def create_homogen_terminal_holons(
    num_of_holons,
    ml_model,
    starting_model=None,
    num_epochs=2,
    batch_size=32,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.SGD,
    optimizer_args={"lr": 0.01},
    dataset_name="mnist",
    samp_type="iid",
    unequal_splits=False,
    log_dir="",
    sys_name="HL",
    synced=False,
    device=torch.device("cpu"),
):
    t_holon_list = []
    trd, tsd, ddi = prepare_sys_data(
        list(range(num_of_holons)), dataset_name, samp_type, unequal_splits
    )
    for i in range(num_of_holons):
        h = holonic_ml_avg.TerminalMLAvgHolon(
            id=str(i),
            rank=i,
            neighbors={},
            superhs={},
            ml_model=ml_model,
            data=trd,
            num_epochs=num_epochs,
            batch_size=batch_size,
            starting_model=starting_model,
            data_idxs=ddi[i],
            global_test_data=tsd,
            log_dir=log_dir,
            sys_name=sys_name,
            synced=synced,
            device=device,
        )
        h.set_train_params(
            criterion(), optimizer(params=h.model.parameters(), **optimizer_args)
        )
        t_holon_list.append(h)
    return t_holon_list, tsd


def extract_number_from_string(s):
    mtch = re.search(r"\d+", s)
    if mtch:
        return int(mtch.group())
    return -1


def config_tensorboard(tb_log_directory):
    """Tensorboard Summary Writer configs"""
    subprocess.run(
        ["rm", "-rf", tb_log_directory]
    )  # cleaning the tensorpoard runs from previous experiments
    writer = tb.SummaryWriter(log_dir=tb_log_directory)
    return writer


def save_dict_to_pickle(dic, dir, f_name):
    file = os.path.join(dir, f_name + ".pkl")
    with open(file, "wb") as f:
        pickle.dump(dic, f)


def read_pickle_to_dict(f_path):
    with open(f_path, "rb") as f:
        dic = pickle.load(f)
        return dic


def read_pickle_to_run_df(f_path):
    i = 0
    dic = read_pickle_to_dict(f_path)
    df_data_dist = pd.DataFrame(
        {"class": list(range(len(dic["data_dist"]))), "counts": dic["data_dist"]}
    )
    df_comm_counts = pd.DataFrame(columns=["sys_name", "hor_comm", "ver_comm"])
    df_train_perf = pd.DataFrame(
        columns=[
            "sys_name",
            "round",
            "epoch",
            "training_accuracy",
            "training_loss",
            "hor_comm",
            "ver_comm",
        ]
    )
    df_test_perf = pd.DataFrame(
        columns=[
            "sys_name",
            "round",
            "epoch",
            "test_accuracy",
            "test_loss",
            "global_test_accuracy",
            "global_test_loss",
        ]
    )
    sys_name = dic["system"]
    for ri, rv in dic["train"].items():  # rounds
        for ei, ev in rv.items():  # episods
            for bv in ev.values():
                df_train_perf.loc[i] = [
                    sys_name,
                    int(ri),
                    int(ei),
                    float(bv["training_accuracy"]),
                    float(bv["training_loss"]),
                    float(bv["hor_comm"]),
                    float(bv["ver_comm"]),
                ]
                i += 1
    i = 0
    for ri, rv in dic["test"].items():  # rounds
        for ei, ev in rv.items():  # episods
            df_test_perf.loc[i] = [
                sys_name,
                int(ri),
                int(ei),
                float(ev["test_accuracy"]),
                float(ev["test_loss"]),
                float(ev["global_test_accuracy"]),
                float(ev["global_test_loss"]),
            ]
            i += 1

    i = 0
    for cc in dic["communication"]:
        df_comm_counts.loc[i] = [sys_name, int(cc["horizontal"]), int(cc["vertical"])]
        i += 1
    
    return df_data_dist, df_comm_counts, df_train_perf, df_test_perf


def merge_run_dfs(f_path):
    df_data_dist = None
    df_comm_counts = pd.DataFrame()
    df_train_perfs = pd.DataFrame()
    df_test_perfs = pd.DataFrame()
    _, _, pfiles = next(os.walk(f_path))
    # print(f"found {len(pfiles)} run files")
    for i, pf in enumerate(sorted(pfiles, key=extract_number_from_string)):
        path = os.path.join(f_path, pf)
        df_data_dist, ccd, trd, tsd = read_pickle_to_run_df(path)
        # print(f"run file {pf} detected. it has {len(trd.index)} rows")
        trd["run"] = i + 1
        tsd["run"] = i + 1
        df_comm_counts = pd.concat([df_comm_counts, ccd], ignore_index=True)
        df_train_perfs = pd.concat([df_train_perfs, trd], ignore_index=True)
        df_test_perfs = pd.concat([df_test_perfs, tsd], ignore_index=True)
    
    # Convert all number columns to numeric format
    df_data_dist = df_data_dist.apply(pd.to_numeric, errors='ignore')
    df_comm_counts = df_comm_counts.apply(pd.to_numeric, errors='ignore')
    df_train_perfs = df_train_perfs.apply(pd.to_numeric, errors='ignore')
    df_test_perfs = df_test_perfs.apply(pd.to_numeric, errors='ignore')
    
    return df_data_dist, df_comm_counts, df_train_perfs, df_test_perfs


def merge_holons_dfs(f_path):
    df_data_dist = pd.DataFrame()
    df_comm_counts = pd.DataFrame()
    df_train_perfs = pd.DataFrame()
    df_test_perfs = pd.DataFrame()
    _, hdir, _ = next(os.walk(f_path))
    for hd in sorted(hdir, key=extract_number_from_string):
        path = os.path.join(f_path, hd)
        dds, ccds, trds, tsds = merge_run_dfs(path)
        # print(f"holon {hd} has {len(trds.index)} rows with data size od {dds['counts'].sum()}")
        dds["holon"] = hd
        ccds["holon"] = hd
        trds["holon"] = hd
        tsds["holon"] = hd
        df_data_dist = pd.concat([df_data_dist, dds], ignore_index=True)
        df_comm_counts = pd.concat(
            [df_comm_counts, ccds],
        )
        df_train_perfs = pd.concat(
            [df_train_perfs, trds],
        )
        # print(f"for holon {hd}:\n")
        # print(trds.to_string())
        df_test_perfs = pd.concat(
            [df_test_perfs, tsds],
        )

    return df_data_dist, df_comm_counts, df_train_perfs, df_test_perfs


def merge_trial_dfs(t_path, which_holons="terminals"):
    df_data_dist = pd.DataFrame()
    df_comm_counts = pd.DataFrame()
    df_train_perfs = pd.DataFrame()
    df_test_perfs = pd.DataFrame()
    _, tdir, _ = next(os.walk(t_path))
    for td in sorted(tdir, key=extract_number_from_string):
        path = os.path.join(t_path, td, which_holons)
        dds, ccs, trds, tsds = merge_holons_dfs(path)
        dds["trial"] = td
        ccs["trial"] = td
        trds["trial"] = td
        tsds["trial"] = td
        df_data_dist = pd.concat([df_data_dist, dds], ignore_index=True)
        df_comm_counts = pd.concat(
            [df_comm_counts, ccs],
        )
        df_train_perfs = pd.concat(
            [df_train_perfs, trds],
        )
        df_test_perfs = pd.concat(
            [df_test_perfs, tsds],
        )
        # print(f"for trial {td}:\n")
        # print(trds.to_string())

    return df_data_dist, df_comm_counts, df_train_perfs, df_test_perfs


def merge_all_sys_defs(s_path, which_holons="terminals"):
    file_names = [which_holons+'_all_data_dist.pkl', which_holons+'_all_comm_counts.pkl', which_holons+'_all_train_perfs.pkl', which_holons+'_all_test_perfs.pkl']
    existing_files_count = 0
    for i in range(len(file_names)):
        file_path = os.path.join(s_path, file_names[i])
        if os.path.exists(file_path):
            print(f"{file_path} already exists. I will reuse them. Delete them if you want to regenerate them.")
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
            if i == 0:
                df_data_dist = df
            elif i == 1:
                df_comm_counts = df
            elif i == 2:
                df_train_perfs = df
            elif i == 3:
                df_test_perfs = df
            else:
                raise ValueError("i is out of range")
            existing_files_count += 1
            continue
        else:
            print(f"working on experiment {s_path}")
            break
    if existing_files_count == len(file_names):
        return df_data_dist, df_comm_counts, df_train_perfs, df_test_perfs
    elif existing_files_count > 0:
        raise ValueError("Some of the files exist and some don't. Delete them all if you want to regenerate them.")
    
    df_data_dist = pd.DataFrame()
    df_comm_counts = pd.DataFrame()
    df_train_perfs = pd.DataFrame()
    df_test_perfs = pd.DataFrame()
    _, sdir, _ = next(os.walk(s_path))
    for sd in sdir:
        # print(f"working on experiment {sd}")
        path = os.path.join(s_path, sd)
        dds, ccs, trds, tsds = merge_trial_dfs(path, which_holons)
        df_data_dist = pd.concat([df_data_dist, dds])
        df_comm_counts = pd.concat(
            [df_comm_counts, ccs],
        )
        df_train_perfs = pd.concat(
            [df_train_perfs, trds],
        )
        df_test_perfs = pd.concat(
            [df_test_perfs, tsds],
        )
    df_train_perfs["total_hor_comm"] = df_train_perfs.groupby(["sys_name", "holon"])[
        "hor_comm"
    ].transform(pd.Series.cumsum)
    df_train_perfs["total_ver_comm"] = df_train_perfs.groupby(["sys_name", "holon"])[
        "ver_comm"
    ].transform(pd.Series.cumsum)
    df_comm_counts["total_hor_comm"] = df_comm_counts.groupby(["sys_name", "holon"])[
        "hor_comm"
    ].transform(pd.Series.cumsum)
    df_comm_counts["total_ver_comm"] = df_comm_counts.groupby(["sys_name", "holon"])[
        "ver_comm"
    ].transform(pd.Series.cumsum)
    data_frames = [df_data_dist, df_comm_counts, df_train_perfs, df_test_perfs]
    for i in range(len(file_names)):
        file_path = os.path.join(s_path, file_names[i])
        with open(file_path, 'wb') as f:
            pickle.dump(data_frames[i], f)
    return df_data_dist, df_comm_counts, df_train_perfs, df_test_perfs



def plot_data_dist(dd_df, in_file=None):
    # sns.set_theme(style="white", palette="viridis")
    g = sns.FacetGrid(dd_df, col="holon", despine=False, hue="trial", col_wrap=4)
    g.map_dataframe(sns.barplot, x="class", y="counts")
    # sns.barplot(dd_df, x="class", y="counts", color="tab:blue")
    if in_file is not None:
        in_file = os.path.join(in_file, "data_dist.pdf")
        plt.savefig(in_file, bbox_inches="tight", format="pdf")
    else:
        plt.show()


def plot_comm_counts(
    cc_df, x="sys_name", y=["hor_comm", "ver_comm"], group_by=None, in_file=None
):
    """Plotting the communication counts for each system"""
    sns.set_theme(style="white", palette="colorblind")
    id_cols = list(set(cc_df.columns) - set(y))
    df = pd.melt(
        cc_df, id_vars=id_cols, var_name="communication_type", value_name="count"
    )
    sns.catplot(
        data=df, x=x, y="count", hue="communication_type", kind="bar", col=group_by
    )
    if in_file is not None:
        in_file = os.path.join(in_file, f"comm_counts_per_{x}.pdf")
        plt.savefig(in_file, bbox_inches="tight", format="pdf")
    else:
        plt.show()


def plot_perf(
    ac_df,
    metric,
    based_on="holon",
    group_by=None,
    in_file=None,
    ddf=None,
    seperate_files=False,
    **kwargs,
):
    # sns.set_theme(style="white", palette="colorblind")

    # sns.lineplot(ac_df, x= ac_df.index, y=metric, hue=based_on)
    rel = sns.relplot(
        data=ac_df,
        x=ac_df.index,
        y=metric,
        hue=based_on,
        kind="line",
        col=group_by,
        errorbar=None,
        style=based_on,
        markers=True,
        markevery=len(ac_df.index) // 10,
        **kwargs,
    )
    if "train" in metric:
        plt.gca().set_yscale("log")
    else:
        plt.gca().set_yscale("linear")
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    
    axin = None
    inset_loc = "upper center"
    if group_by is not None and ddf is not None:
        if "test" in metric:
            inset_loc = "lower center"
        for i, ax in enumerate(rel.axes.flat):
            axin = inset_axes(ax, width="35%", height="35%", loc=inset_loc, borderpad=4)
            axin.set_ylim(0, 1400)
            axin.set_title(f"data distribution")
            if "train" in metric:
                ax.set_yscale("log")
            else:
                ax.set_yscale("linear")
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
            sub_data_df = ddf[ddf["holon"] == str(i)]
            sns.barplot(
                data=sub_data_df, x="class", y="counts", ax=axin, color="tab:blue"
            )
            # axin.legend_.remove()

    if in_file is not None:
        if not os.path.exists(in_file) and in_file != "":
            os.makedirs(in_file)
        in_file = os.path.join(in_file, f"{metric}_per_{based_on}_group_by_{group_by}")
        # plt.tight_layout()
        rel.fig.tight_layout()
        rel.fig.savefig(in_file + ".pdf", format="pdf")
        if seperate_files and axin is not None:
            plt.close(rel.fig)
            plt.clf()
            for h in list(ddf["holon"].unique()):
                sub_ac_df = ac_df[ac_df["holon"] == h]
                ln = sns.lineplot(
                    data=sub_ac_df, x=sub_ac_df.index, y=metric, hue=based_on, errorbar=None, style=based_on, markers=True,markevery=len(sub_ac_df.index) // 10,
                )
                ln.set_title(f"holon={h}")
                if "train" in metric:
                    ln.set_yscale("log")
                else:
                    ln.set_yscale("linear")
                ln.yaxis.set_major_formatter(ScalarFormatter())
                ln.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
                axin = inset_axes(
                    ln, width="30%", height="30%", loc=inset_loc, borderpad=4
                )
                axin.set_ylim(0, 1400)
                axin.set_title(f"data distribution")
                sub_data_df = ddf[ddf["holon"] == h]
                # sns.countplot(data=sub_data_df, x="class", ax=axin)
                sns.barplot(
                    data=sub_data_df, x="class", y="counts", ax=axin, hue="trial"
                )
                axin.legend_.remove()
                ln.figure.tight_layout()
                ln.figure.savefig(in_file + f"_holon({h}).pdf", format="pdf")
                plt.close(ln.figure)
    else:
        plt.show()


def extract_dic_from_keys(dic, keys):
    """extract a sub dictionary from a dictionary based on the keys provided"""
    return {k: dic[k] for k in keys}


def log_print(*args, **kwargs):
    if Globals.DEBUG_MODE:
        print(*args, **kwargs)


if __name__ == "__main__":
    """For trying the defined methods"""
    
