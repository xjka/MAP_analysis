"""Reader for Abacus ``stepXXXX.time.bin`` timing log files.

A ``.time.bin`` file is written by ``TimingReport::WriteReport()``
(``src/Reporting/reporting.cpp``) in **PARALLEL builds only**.  Rank 0 gathers
every rank's ``PerformanceSummary`` struct and dumps them back-to-back with a
single ``fwrite`` -- no header.  So the file is just an array of ``MPI_size``
``PerformanceSummary`` structs.

Current sims self-describe this layout: rank 0 also drops a
``timing_schema.json`` (see ``src/Reporting/generate_timing_schema.py`` and
``py/abacus/post/history.py``) into the ``LogDirectory``, and that's the
preferred path once you have it -- gather with ``python -m abacus.post.history``
and then just ``asdf.open()`` the resulting ``timing_log.asdf``.

THIS FILE is for old sims that predate that schema drop and have no
``timing_schema.json`` to fall back on.  It hand-maintains the numpy dtype for
``PerformanceSummary`` (``src/Reporting/reporting_summary.hpp``) below, built
with nested ``np.dtype(..., align=True)`` so the byte offsets match the
compiler's struct layout (the same technique ``generate_timing_schema.py``
uses, just without parsing the header at runtime).

The dtype below mirrors the CURRENT header.  Old sims used an earlier,
usually smaller, version of the struct -- fields were added over time and
never removed/reordered, so an old ``.time.bin`` is normally a *prefit* of
this dtype missing some trailing members of one or more (sub)structs.  To
read one:

1. Find the version of ``reporting_summary.hpp`` the sim's binary was built
   from (git blame/log around the sim's run date, or ask -- whoever ran it).
2. Diff it against the current header and tell Claude what to remove/change
   below (e.g. "this sim predates ``state_compress_*``" or "``mapforest``
   didn't exist yet, and ``spin_breakers`` was only 25 long").
3. The dtypes are built bottom-up as named module constants specifically so
   a single struct can be edited/trimmed without touching the rest.

Usage::

    from read_timebin import load, fixedstring
    data = load("log/step0123/step0123.time.bin")
    data['singlestep_total_time']                     # per-rank wall time
    data['near_force']['gpu_gdirects']                 # nested sub-struct field
    fixedstring(data['sim_name'])                      # FixedString -> list[str]
"""

import sys
from pathlib import Path

import numpy as np

MULTIPLICITY_BINS = 24
MAX_SPIN_BREAKERS = 40


def fixed_string_dtype(max_len):
    """Mirrors FixedString<max_len> (src/Reporting/fixed_string.hpp): a
    size_t length prefix followed by a NUL-terminated char buffer."""
    return np.dtype([('size', '<u8'), ('buf', f'S{max_len}')], align=True)


def fixedstring(arr):
    """Decode a FixedString field (or array of them) to Python str(s)."""
    buf = np.asarray(arr['buf'])
    if buf.ndim == 0:
        return buf.item().split(b'\x00', 1)[0].decode('utf-8', 'replace')
    return [b.split(b'\x00', 1)[0].decode('utf-8', 'replace') for b in buf.ravel()]


multiplicity_stats_summary_dtype = np.dtype([
    ('ngroups', '<u8'),
    ('largest', '<u8'),
    ('tot', '<u8'),
    ('tot2', '<u8'),
    ('count', '<u8', (MULTIPLICITY_BINS,)),
    ('sumn', '<u8', (MULTIPLICITY_BINS,)),
    ('sumn2', '<u8', (MULTIPLICITY_BINS,)),
], align=True)

map_collision_stats_dtype = np.dtype([
    ('data', '<f4', (30,)),
], align=True)

io_thread_stats_dtype = np.dtype([
    ('total_read_time', '<f4'),
    ('total_read_bytes', '<f4'),
    ('total_write_time', '<f4'),
    ('total_write_bytes', '<f4'),
    ('checksum_time', '<f4'),
    ('checksum_bytes', '<f4'),
], align=True)

slab_pipeline_timings_dtype = np.dtype([
    ('fetch_slabs_time', '<f4'),
    ('transpose_positions_time', '<f4'),
    ('near_force_time', '<f4'),
    ('unpack_env_slab_time', '<f4'),
    ('accw_exchange_time', '<f4'),
    ('taylors_time', '<f4'),
    ('kick_time', '<f4'),
    ('group_finding_time', '<f4'),
    ('output_time', '<f4'),
    ('drift_time', '<f4'),
    ('neighbor_exchange_time', '<f4'),
    ('finish_time', '<f4'),
    ('check_multipoles_mpi_time', '<f4'),
    ('manifest_time', '<f4'),
    ('spinning_time', '<f4'),
    ('unaccounted_pipeline_time', '<f4'),
], align=True)

per_slab_stats_dtype = np.dtype([
    ('force_time_mean', '<f4'),
    ('force_time_sigma', '<f4'),
    ('force_time_min', '<f4'),
    ('force_time_max', '<f4'),
    ('force_latency_mean', '<f4'),
    ('force_latency_sigma', '<f4'),
    ('force_latency_min', '<f4'),
    ('force_latency_max', '<f4'),
], align=True)

near_force_timings_dtype = np.dtype([
    ('num_particles', '<i8'),
    ('blocking_total_time', '<f4'),
    ('precondition_time', '<f4'),
    ('calc_splits_time', '<f4'),
    ('construct_pencils_time', '<f4'),
    ('dispatch_interaction_time', '<f4'),
    ('cpu_fallback_time', '<f4'),
    ('cpu_gdirects', '<f4'),
    ('cpu_nsink_mpart_per_sec', '<f4'),
    ('device_thread_seconds', '<f4'),
    ('num_gpu_threads', '<i4'),
    ('fill_sinks_time', '<f4'),
    ('fill_sinks_bandwidth_mbps', '<f4'),
    ('fill_sinks_msink_per_sec', '<f4'),
    ('fill_sources_time', '<f4'),
    ('fill_sources_bandwidth_mbps', '<f4'),
    ('fill_sources_msource_per_sec', '<f4'),
    ('launch_kernels_time', '<f4'),
    ('wait_for_result_time', '<f4'),
    ('copy_accel_time', '<f4'),
    ('copy_accel_bandwidth_mbps', '<f4'),
    ('copy_accel_msink_per_sec', '<f4'),
    ('gpu_throughput_time', '<f4'),
    ('gpu_msink_per_sec', '<f4'),
    ('gpu_gdirects', '<f4'),
    ('gpu_padded_gdirects', '<f4'),
    ('directs_load_imbalance_percent', '<f4'),
    ('gb_to_device', '<f4'),
    ('gb_from_device', '<f4'),
    ('h2d_bytes_per_particle', '<f4'),
    ('source_replication', '<f4'),
    ('source_padding', '<f4'),
    ('sink_replication', '<f4'),
    ('sink_padding', '<f4'),
], align=True)

taylor_timings_dtype = np.dtype([
    ('num_particles', '<i8'),
    ('taylor_computation_time', '<f4'),
    ('precondition_time', '<f4'),
    ('compute_offsets_time', '<f4'),
    ('unpack_mpi_buffer_time', '<f4'),
    ('fft_taylor_time', '<f4'),
    ('taylor_kernel_time', '<f4'),
    ('taylor_redlack_time', '<f4'),
    ('taylor_transpose_time', '<f4'),
    ('transpose_precondition_time', '<f4'),
    ('fft_z_taylor_time', '<f4'),
    ('fill_mpi_bufs_time', '<f4'),
    ('all_to_all_time', '<f4'),
], align=True)

kick_timings_dtype = np.dtype([
    ('num_particles', '<i8'),
    ('kick_total_time', '<f4'),
    ('precondition_time', '<f4'),
    ('accumulate_pencil_stats_time', '<f4'),
    ('add_accel_time', '<f4'),
    ('add_accel_bandwidth_gbps', '<f4'),
    ('kick_cell_time', '<f4'),
    ('free_slabs_time', '<f4'),
], align=True)

group_finding_timings_dtype = np.dtype([
    ('num_particles', '<i8'),
    ('num_active_particles', '<i4'),
    ('group_finding_time', '<f4'),
    ('preconditions_time', '<f4'),
    ('make_cell_groups_time', '<f4'),
    ('find_cell_group_links_time', '<f4'),
    ('do_global_groups_time', '<f4'),
    ('finish_groups_time', '<f4'),
    ('scatter_groups_time', '<f4'),
    ('total_group_particles', '<i8'),
], align=True)

output_timings_dtype = np.dtype([
    ('num_particles', '<i8'),
    ('output_total_time', '<f4'),
    ('precondition_time', '<f4'),
    ('kick_ghost_cells_time', '<f4'),
    ('output_time_slice_time', '<f4'),
    ('output_light_cone_time', '<f4'),
    ('output_bin_time', '<f4'),
    ('lc_setup_time', '<f4'),
    ('lc_search_time', '<f4'),
    ('lc_gather_output_time', '<f4'),
    ('lc_sort_particles_time', '<f4'),
    ('lc_compress_particles_time', '<f4'),
    ('lc_create_sparse_map_time', '<f4'),
    ('lc_sort_healpix_time', '<f4'),
    ('lc_compress_healpix_time', '<f4'),
    ('num_lightcone_particles', '<i8'),
    ('lc_free_slab_accum_time', '<f4'),
], align=True)

drift_timings_dtype = np.dtype([
    ('num_particles', '<i8'),
    ('drift_total_time', '<f4'),
    ('precondition_time', '<f4'),
    ('move_time', '<f4'),
    ('rebin_time', '<f4'),
], align=True)

neighbor_exchange_timings_dtype = np.dtype([
    ('neighbor_send_time', '<f4'),
    ('neighbor_receive_time', '<f4'),
], align=True)

accw_exchange_timings_dtype = np.dtype([
    ('send_time', '<f4'),
    ('receive_time', '<f4'),
], align=True)

multipole_timings_dtype = np.dtype([
    ('compute_multipoles_time', '<f4'),
    ('compute_offsets_time', '<f4'),
    ('multipole_kernel_time', '<f4'),
    ('fft_multipole_time', '<f4'),
    ('fill_mpi_bufs_time', '<f4'),
    ('all_to_all_time', '<f4'),
], align=True)

post_timestep_timings_dtype = np.dtype([
    ('post_timestep_total_time', '<f4'),
    ('subsample_io_time', '<f4'),
    ('lc_start_finalize_time', '<f4'),
    ('reductions_time', '<f4'),
], align=True)

lc_complete_finalize_timings_dtype = np.dtype([
    ('total_time', '<f4'),
    ('barrier_time', '<f4'),
    ('gather_time', '<f4'),
    ('handoff_time', '<f4'),
], align=True)

finish_timings_dtype = np.dtype([
    ('finish_particles_time', '<f4'),
    ('particles_precondition_time', '<f4'),
    ('free_readstate_slabs_time', '<f4'),
    ('collect_il_gaps_time', '<f4'),
    ('partition_insert_list_time', '<f4'),
    ('sort_insert_list_time', '<f4'),
    ('mmsort_indexing_time', '<f4'),
    ('mmsort_sorting_time', '<f4'),
    ('mmsort_refine_partition_time', '<f4'),
    ('mmsort_merging_time', '<f4'),
    ('mmsort_sortpart_time', '<f4'),
    ('mmsort_bisect_time', '<f4'),
    ('mmsort_popping_time', '<f4'),
    ('num_sorted_items', '<i8'),
    ('index_cells_time', '<f4'),
    ('merge_time', '<f4'),
    ('compute_multipoles_finish_time', '<f4'),
    ('write_particles_time', '<f4'),
    ('queue_send_manifest_time', '<f4'),
    ('finish_multipoles_time', '<f4'),
    ('multipoles_precondition_time', '<f4'),
    ('unpack_mpi_buffer_time', '<f4'),
    ('fft_z_multipole_time', '<f4'),
    ('fft_z_transpose_time', '<f4'),
    ('queue_multipole_mpi_time', '<f4'),
    ('write_multipoles_time', '<f4'),
    ('release_free_memory_time', '<f4'),
    ('free_mpi_multipoles_time', '<f4'),
], align=True)

mpi_work_timings_dtype = np.dtype([
    ('check_manifest_mpi_time', '<f4'),
    ('queue_receive_manifest_time', '<f4'),
    ('queue_send_manifest_finish_time', '<f4'),
    ('queue_multipole_mpi_finish_time', '<f4'),
    ('check_mpi_multipoles_precon_time', '<f4'),
    ('free_mpi_multipoles_time', '<f4'),
    ('check_2d_multipole_mpi_time', '<f4'),
    ('check_2d_taylor_mpi_time', '<f4'),
], align=True)

spin_breaker_entry_dtype = np.dtype([
    ('name', fixed_string_dtype(32)),
    ('time', '<f4'),
], align=True)

spinning_timings_dtype = np.dtype([
    ('spinning_total_time', '<f4'),
    ('checking_preconditions_time', '<f4'),
    ('not_enough_ram_time', '<f4'),
    ('waiting_for_io_time', '<f4'),
    ('waiting_for_gpu_time', '<f4'),
    ('waiting_for_mpi_taylor_recv_time', '<f4'),
    ('waiting_for_mpi_taylor_send_time', '<f4'),
    ('waiting_for_mpi_taylor_2d_time', '<f4'),
    ('waiting_for_mpi_multipole_recv_time', '<f4'),
    ('waiting_for_mpi_multipole_send_time', '<f4'),
    ('waiting_for_mpi_multipole_2d_time', '<f4'),
    ('waiting_for_mpi_manifest_time', '<f4'),
    ('waiting_for_mpi_neighbor_recv_2d_time', '<f4'),
    ('waiting_for_mpi_accdens_recv_2d_time', '<f4'),
    ('spin_breakers', spin_breaker_entry_dtype, (MAX_SPIN_BREAKERS,)),
    ('num_spin_breakers', '<i4'),
], align=True)

convolution_timings_dtype = np.dtype([
    ('total_memory_allocated', '<i8'),
    ('cache_size_mb', '<f4'),
    ('compute_cores', '<i4'),
    ('blocksize', '<i4'),
    ('zwidth', '<i4'),
    ('order', '<i4'),
    ('convolve_wallclock', '<f4'),
    ('unaccounted_time', '<f4'),
    ('constructor_time', '<f4'),
    ('alloc_free_time', '<f4'),
    ('array_swizzle_time', '<f4'),
    ('load_derivs_time', '<f4'),
    ('read_derivatives_time', '<f4'),
    ('read_derivatives_bytes', '<i8'),
    ('expand_derivs_time', '<f4'),
    ('convolution_arithmetic_time', '<f4'),
    ('convolution_ops', '<i8'),
    ('fft_planning_time', '<f4'),
    ('forward_fft_multipoles_time', '<f4'),
    ('inverse_fft_taylor_time', '<f4'),
    ('queue_taylors_time', '<f4'),
    ('destructor_time', '<f4'),
    ('read_multipoles_time', '<f4'),
    ('read_multipoles_bytes', '<i8'),
    ('write_taylor_time', '<f4'),
    ('write_taylor_bytes', '<i8'),
    ('wait_for_io_time', '<f4'),
], align=True)

gfc_report_data_dtype = np.dtype([
    ('has_data', '|b1'),
    ('cg_active', '<u8'),
    ('cg_tot', '<u8'),
    ('pp_tot', '<u8'),
    ('fp_tot', '<u8'),
    ('fg_tot', '<u8'),
    ('link_tot', '<u8'),
    ('gg_tot', '<u8'),
    ('max_group_diameter', '<i4'),
    ('max_group_span', '<f8'),
    ('largest_gg', '<i4'),
    ('min_halo_size', '<i4'),
    ('np_gfc', '<u8'),
    ('longest_group_link_list', '<i8'),
    ('max_group_link_alloc', '<i8'),
    ('numdists1', '<i8'), ('numdists2', '<i8'),
    ('numsorts1', '<i8'), ('numsorts2', '<i8'),
    ('numcenters1', '<i8'), ('numcenters2', '<i8'),
    ('numcg1', '<i8'), ('numcg2', '<i8'),
    ('numgroups1', '<i8'), ('numgroups2', '<i8'),
    ('cell_group_time', '<f4'),
    ('create_face_time', '<f4'),
    ('find_link_time', '<f4'),
    ('sort_links_time', '<f4'),
    ('index_links_time', '<f4'),
    ('index_links_index_time', '<f4'),
    ('defer_groups_time', '<f4'),
    ('find_global_group_time', '<f4'),
    ('clear_defer_time', '<f4'),
    ('index_groups_time', '<f4'),
    ('gather_groups_time', '<f4'),
    ('scatter_aux_time', '<f4'),
    ('scatter_groups_time', '<f4'),
    ('process_level1_time', '<f4'),
    ('l1_fof_time', '<f4'),
    ('l2_fof_time', '<f4'),
    ('l1_tot_time', '<f4'),
    ('output_level1_time', '<f4'),
    ('collide_tracers_time', '<f4'),
    ('particle_sort_time', '<f4'),
    ('create_groups_formation_time', '<f4'),
    ('create_groups_merger_time', '<f4'),
    ('create_groups_accretion_time', '<f4'),
    ('process_groups_time', '<f4'),
    ('tracers_map_output_time', '<f4'),
    ('tracers_group_output_time', '<f4'),
    ('l0_stats', multiplicity_stats_summary_dtype),
    ('l1_stats', multiplicity_stats_summary_dtype),
    ('l2_stats', multiplicity_stats_summary_dtype),
    ('map_collision', map_collision_stats_dtype),
], align=True)

mapforest_stats_dtype = np.dtype([
    ('size', '<u8'),
    ('partition_size', '<u8'),
    ('min_free_margin', '<i8'),
    ('end_active', '<u8'),
    ('global_lowerbound', '<u8'),
    ('overflow_nodes', '<u8'),
    ('overflow_hits', '<u8'),
    ('overflow_laps', '<u8'),
    ('incoming_usage', '<u8'),
], align=True)

# ===== Top-level struct (src/Reporting/reporting_summary.hpp) =====
performance_summary_dtype = np.dtype([
    ('rank', '<i4'),
    ('rank_z', '<i4'),
    ('sim_name', fixed_string_dtype(128)),
    ('cpu_model', fixed_string_dtype(128)),
    ('gpu_name', fixed_string_dtype(128)),
    ('log_directory', fixed_string_dtype(512)),
    ('full_step_number', '<i4'),
    ('node_string', fixed_string_dtype(16)),
    ('singlestep_epilogue_time', '<f4'),
    ('multistep_prologue_time', '<f4'),
    ('multistep_epilogue_time', '<f4'),
    ('multistep_io_finish_time', '<f4'),
    ('multistep_total_time', '<f4'),
    ('with_multistep_proepi', '<i4'),
    ('singlestep_total_time', '<f4'),
    ('num_particles_node', '<i8'),
    ('num_particles_with_ghost', '<i8'),
    ('ppd', '<i4'),
    ('cpd', '<i4'),
    ('particles_per_cell', '<f4'),
    ('mpi_size_x', '<i4'),
    ('mpi_size_z', '<i4'),
    ('num_cpu_cores', '<i4'),
    ('num_gpus', '<i4'),
    ('singlestep_prologue_time', '<f4'),
    ('slab_pipeline_time', '<f4'),
    ('post_timestep_time', '<f4'),
    ('convolution_time', '<f4'),
    ('lc_complete_finalize_time', '<f4'),
    ('finish_io_time', '<f4'),
    ('unaccounted_time', '<f4'),
    ('io_stats', io_thread_stats_dtype),
    ('state_compress_time', '<f4'),
    ('state_compress_in_bytes', '<f4'),
    ('state_compress_out_bytes', '<f4'),
    ('manifest_check_time', '<f4'),
    ('receive_manifest_time', '<f4'),
    ('send_manifest_time', '<f4'),
    ('mpi_barrier_time', '<f4'),
    ('slab_pipeline', slab_pipeline_timings_dtype),
    ('per_slab_stats', per_slab_stats_dtype),
    ('near_force', near_force_timings_dtype),
    ('taylor', taylor_timings_dtype),
    ('kick', kick_timings_dtype),
    ('group_finding', group_finding_timings_dtype),
    ('output', output_timings_dtype),
    ('drift', drift_timings_dtype),
    ('neighbor_exchange', neighbor_exchange_timings_dtype),
    ('accw_exchange', accw_exchange_timings_dtype),
    ('multipole', multipole_timings_dtype),
    ('post_timestep', post_timestep_timings_dtype),
    ('lc_complete_finalize', lc_complete_finalize_timings_dtype),
    ('finish', finish_timings_dtype),
    ('mpi_work', mpi_work_timings_dtype),
    ('spinning', spinning_timings_dtype),
    ('arena_malloc_time', '<f4'),
    ('arena_free_time', '<f4'),
    ('slab_accum_free_time', '<f4'),
    ('fifo_write_time', '<f4'),
    ('log_write_time', '<f4'),
    ('max_rss_bytes', '<u8'),
    ('peak_arena_alloc_bytes', '<u8'),
    ('peak_numa_arena_alloc_bytes', '<u8'),
    ('tcmalloc_held_bytes', '<u8'),
    ('gpu_host_buffer_bytes', '<u8'),
    ('numaalloc_peak_bytes', '<u8'),
    ('mpi_buffer_pool_bytes', '<u8'),
    ('mpi_buffer_pool_inuse_peak', '<u8'),
    ('min_cell_size', '<i4'),
    ('max_cell_size', '<i4'),
    ('rms_fractional_overdensity', '<f4'),
    ('rms_velocity', '<f4'),
    ('convolution', convolution_timings_dtype),
    ('gfc_report', gfc_report_data_dtype),
    ('mapforest', mapforest_stats_dtype),
], align=True)


def load(path, dtype=performance_summary_dtype):
    """Load a ``stepXXXX.time.bin`` file into a structured array of shape
    (MPI_size,), one element per rank's PerformanceSummary.

    Pass a trimmed/adjusted ``dtype`` for sims whose binary predates the
    current struct layout (see the module docstring)."""
    path = Path(path)
    size = path.stat().st_size
    if size % dtype.itemsize != 0:
        raise ValueError(
            "%s size %d is not a multiple of itemsize %d; wrong (or "
            "out-of-date) dtype for this sim, or a truncated file?"
            % (path, size, dtype.itemsize))
    return np.fromfile(path, dtype=dtype)


if __name__ == '__main__':
    for path in sys.argv[1:]:
        d = load(path)
        print("\n%s: %d rank(s), sim=%r step=%d"
              % (path, d.shape[0], fixedstring(d['sim_name'][0]),
                 d['full_step_number'][0]))
        for r in d:
            print("  rank %3d (z=%d): total=%.3fs  pipeline=%.3fs  convolve=%.3fs  "
                  "maxrss=%.2fGB"
                  % (r['rank'], r['rank_z'], r['singlestep_total_time'],
                     r['slab_pipeline_time'], r['convolution_time'],
                     r['max_rss_bytes'] / 1e9))
