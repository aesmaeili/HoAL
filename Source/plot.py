"""the python script to generated the required plots"""
import utilities as utils
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expname", type=str, default="test", help="name of the experiment")
    parser.add_argument("--plots", type=list, default=["data_dist", 
                                                       ("running_loss", "sys_name"), 
                                                       ("test_accuracy", "sys_name"), 
                                                       ("running_loss", "sys_name", "holon"), 
                                                       ("test_accuracy", "sys_name", "holon")
                                                       ], help="list of plots to generate")
    parser.add_argument("--expfolder", type=str, default="Experiments/", help="directory of the experiments performance data")
    parser.add_argument("--plotfolder", type=str, default="Plots/", help="directory of the plots")
    parser.add_argument("--holontype", type=str, default="terminals", help="type of holons to plot")
    parser.add_argument("--infile", type=bool, default=False, help="whether to save to file or not")

    args = parser.parse_args()
    expname = args.expname
    plots = args.plots
    expfolder = args.expfolder
    plotfolder = args.plotfolder
    holontype = args.holontype
    infile = args.infile

    dd,cc,trp,tsp=utils.merge_all_sys_defs(expfolder, holontype)

    if not infile:
        plot_file = None
    else:
        plot_file = os.path.join(plotfolder, expname)
        os.makedirs(plot_file, exist_ok=True)
    
    for p in plots:
        if p == "data_dist":
            utils.plot_data_dist(dd, in_file=plot_file)
        else:
            df = trp
            if p[0] == "test_accuracy" or p[0]=="test_loss":
                df = tsp
            print(p)
            if len(p) == 2:
                utils.plot_perf(df, p[0], p[1], in_file=plot_file)
            elif len(p) == 3:
                utils.plot_perf(df, p[0], p[1], p[2], in_file=plot_file, col_wrap=4)

if __name__ == "__main__":
    main()