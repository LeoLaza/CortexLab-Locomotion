"""
CortexLab-Locomotion

A modular Python package for analyzing neural activity from chronic neuropixel recordings 
in mice during bouts of locomotion in different contexts.
"""

from .neural_processing import (
    get_spike_hist,
    normalize_spike_counts, 
    filter_spike_counts
)

from .behavioral_analysis import (
    calculate_median_position,
    calculate_oa_speed,
    calculate_wh_speed,
    get_position_masks,
    temporal_buffer,
    get_speed_masks,
    detect_bouts
)

from .correlation_analysis import (
    get_correlations,
    get_cross_context_correlations
)

from .statistical_testing import (
    cross_validate_correlations,
    compute_null_distributions_for_session,
    compute_p_values_from_null,
    add_p_values_to_session,
    categorise_neurons
)

from .visualization import (
    plot_correlation_histogram,
    plot_sorted_spike_counts,
    plot_wheel_arena_corr,
    plot_cross_validation,
    plot_masked_positions,
    plot_single_session,
    plot_all_sessions,
    plot_categories,
    plot_likelihood_distributions,
    plot_position_changes,
    plot_dlc_pre_post,
    plot_dlc_analyses
)

from .data_io import (
    load_ONE,
    get_experiment_path,
    get_cam_timestamps,
    create_time_bins,
    temporally_align_variable,
    get_dlc_df,
    preprocess_dlc_data,
    load_probes,
    get_rotary_position,
)

from .roi_detection import (
    get_ROI,
    preprocess_frame,
    plot_ROI
)

from .pipeline import (
    load_and_process_session,
    analyze_single_session,
    analyze_multiple_sessions,
)

__version__ = "0.1.0"
__author__ = "Leonard Lazarevic"