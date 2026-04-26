from analysis.pipeline_group import run_all

if __name__ == "__main__":
    trial_df, reg_df, tail_df = run_all("data/expdata/index.csv")
    print("trial rows:", len(trial_df))
    print("regular event rows:", len(reg_df))
    print("tail event rows:", len(tail_df))