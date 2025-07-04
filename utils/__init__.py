"""
CortexLab-Locomotion

A modular Python package for analyzing neural activity from chronic neuropixel recordings 
in mice during bouts of locomotion in different contexts.
"""


from .behavioral_analysis import (
    calculate_median_position,
    calculate_oa_speed,
    calculate_wh_speed,
    get_position_masks,
    temporal_buffer,
    get_ROI,
    preprocess_frame, 
)

from .correlation_analysis import (
    get_speed_correlations,
    get_cross_context_correlations,
    get_split_half_correlations,
    get_reliability_stability
)

from .decoding_analysis import(
    split_for_decoding,
    train_model,
    compute_leaveout_analysis,
)

from .data_loading_and_preprocessing import(
    load_ONE,
    get_experiment_identifiers,
    get_cam_timestamps,
    create_time_bins,
    temporally_align_variable,
    get_dlc_df,
    preprocess_dlc_data,
    load_probes,
    get_rotary_position,
    get_spike_hist,
    normalize_spike_counts, 
    filter_spike_counts
)

from .statistical_testing import (
    run_permutation_test,
    categorise_neurons,
)

from .visualization import (
    plot_correlation_histogram,
    plot_cross_context_correlation,
    plot_raster_pos_neg,
    plot_reliability,
    plot_masked_positions,
)




__version__ = "2.1.0"
__author__ = "Leonard Lazarevic"