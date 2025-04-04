from pinkrigs_tools.dataset.query import load_data, queryCSV
import pandas as pd
from IPython.display import display, HTML

def load_specific_experiment_data(subject_id, date, data_components=None):
    """
    Load data for a single specific subject and date.
    
    Parameters:
    -----------
    subject_id : str
        The subject ID to load data for (e.g., 'GB012')
    date : str
        The specific experiment date (e.g., '2024-06-25')
    data_components : dict, optional
        Dictionary specifying which data components to load
        If None, loads all default data
        
    Returns:
    --------
    dict
        Dictionary containing the requested data
    """
    try:
        if data_components is None:
            # Load all available data
            data = load_data(
                subject=subject_id,
                expDate=date,
                data_name_dict='all-default'
            )
        else:
            # Load only specified components
            data = load_data(
                subject=subject_id,
                expDate=date,
                data_name_dict=data_components
            )
        
        return data
        
    except Exception as e:
        print(f"Error loading data for subject {subject_id} on date {date}: {e}")
        return None

def get_DLC_data (subject_id, date, delay):
        dlc_df = pd.read_hdf(fr'\\znas\Lab\Share\Maja\labelled_DLC_videos\{subject_id}_{date}DLC_resnet50_downsampled_trialJul11shuffle1_150000_filtered.h5')
        dlc_df = dlc_df.iloc[delay:]
        scorer = dlc_df.columns.get_level_values('scorer')

        return dlc_df, scorer


def get_subjectIDs_and_dates():
    experiments = queryCSV(expDef='spontaneousActivity')
    rigName = experiments[experiments['rigName']== 'poppy-stim']
    SUBJECT_IDS = rigName['subject'].unique()
    DATES = rigName['expDate'].unique()
    return SUBJECT_IDS, DATES


def get_experiment_path(data):
    exp_idx = data.index[data.expDef.isin(['spontaneousActivity'])][0]
    exp_folder = data.loc[exp_idx, 'expFolder']
    exp_num = data.loc[exp_idx, 'expNum']
    return exp_num, exp_folder

