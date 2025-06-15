
def plot_wv_sessions():
    DATES = ['2024-03-04', '2024-03-05', '2024-03-08', '2024-03-11','2024-03-09', '2024-03-10', '2024-03-12', '2024-03-13', '2024-03-14', '2024-03-15', '2024-03-19', '2024-03-20', '2024-03-22', '2024-03-25', '2024-03-26', '2024-05-09', '2024-05-10', '2024-05-14', '2024-05-15', '2024-05-16', '2024-05-17', '2024-06-19', '2024-06-20', '2024-06-21', '2024-06-25', '2024-07-11', '2024-07-12', '2024-07-16']        
    SUBJECT_IDS = ['AV043', 'EB036', 'EB037', 'GB011', 'GB012']

    n_rows = 4  
    n_cols = 4
    fig = plt.figure(figsize = (24,24))

    valid_data = {}
    plot_count = 0

    for subject_id in tqdm(SUBJECT_IDS):

        fig = plt.figure(figsize = (24,24))
        
        for date in DATES:

            print(f'Processing {subject_id} on {date}')

            try:
                data = get_all_pinkrigs_data(subject_id, date)
                dlc_df, scorer = get_DLC_data(subject_id, date)
                folder = get_experiment_path(all_pinkrigs_data)
                exp_idx = data.index[data.expDef.isin(['spontaneousActivity'])]
                print(exp_idx)

                exp_folder = data.loc[exp_idx[0], 'expFolder']
                print(f"Exp folder is {exp_folder}")
                rotary_timestamps, rotary = get_rotary_metadata(exp_folder)
                print(f"Rotary metadata loaded for {subject_id} on {date}")
                position, wheel_velocity = calculate_wheel_velocity(rotary_timestamps, rotary)
                print(f"Rotary velocity calculated for {subject_id} on {date}")
                valid_data[(subject_id, date)] = wheel_velocity
                plot_count += 1


            except FileNotFoundError:
                    print(f"File not found for mouse {subject_id} on date {date}")

    if plot_count > n_rows * n_cols:
        n_rows = int(np.ceil(np.sqrt(plot_count)))
        n_cols = int(np.ceil(plot_count / n_rows))
        print(f"Adjusted grid to {n_rows} rows x {n_cols} columns to fit {plot_count} plots")

    plot_idx = 1
    for (subject_id, date), wheel_velocity in valid_data.items():
        ax1 = fig.add_subplot(n_rows, n_cols, plot_idx)
        ax1.plot(wheel_velocity)
        ax1.set_title(f"{subject_id} {date}")
        ax1.set_xlabel('Log Wheel Velocity')
        plot_idx += 1

    plt.tight_layout(pad=3.0)
    plt.show()