#Automated ELISA 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from matplotlib.lines import Line2D

#rev chi sq in within run SD also
# ============================================================================
# INPUT PARAMETERS — CONCENTRATION SPECIFIC
# ============================================================================

# Nominal concentrations (Vp/mL)
concentrations = [9.08e08, 4.54e08, 2.27e08, 1.13e08, 5.67e07, 2.84e07]

# Measured concentration mean and SD from within-run replicates
# C_mean: mean back-calculated Vp/mL titer  — fill in from data
# C_std:  SD of back-calculated titers       — captures within-run precision
C_mean = [8.34e08, 4.02e08, 2.13e08, 1.01e08, 4.92e07, 4.37e07]
C_std  = [6.03e07, 3.11e07, 2.03e07, 1.09e07, 1.22e07, 1.97e07]

# Between-series CV % for each concentration.
# Set to the MEASURED between-series CV from the validation (defined below as
# MEASURED_BW_CV) so the main three-panel figure is evaluated at each
# concentration's real day-to-day precision, not a flat placeholder. Previously
# this was a flat 7%, which made high-BW concentrations look better than they are.
BW_CV_PER_CONC = [7, 7, 7, 7, 7, 7]

# Replicates per concentration
N_REPLICATES_PER_CONC = [3, 3, 3, 3, 3, 3]

# Experimental design
P_SERIES = 4
DF_BW    = P_SERIES - 1

# Within-run (repeatability) degrees of freedom.
# C_std at each concentration is estimated from N replicates in a single run,
# so it is itself uncertain. We sample the within-run variance from an inverse
# chi-square (same principle as the between-series term) instead of treating
# C_std as exactly known. Without this the simulation is optimistically biased
# and disagrees with the validation report (e.g. conc 4 wrongly PASSES at n=3).
#   DF_W = N_rep - 1 for a single run of N_rep replicates.
# If C_std is pooled across runs, raise this to the pooled df.
DF_W_PER_CONC = [n - 1 for n in N_REPLICATES_PER_CONC]   # = 2 for n=3

# Master switch: applies to BOTH the main MCM and the heatmap, so the two
# figures can never disagree. Set False to treat C_std as exactly known.
# CAUTION: with DF_W = 2 the inverse-gamma shape is 1.0 and the sampled
# variance has NO finite mean -- the widening is then dominated by rare
# extreme draws rather than by the data. Only defensible if C_std is pooled
# across runs so DF_W >= 4. See notes in the response.
WITHIN_RUN_CHI_SQUARE = True

print("=" * 80)
print("PRECISION SIMULATION SETTINGS (CONCENTRATION-SPECIFIC)")
print("=" * 80)
print(f"Number of series (p): {P_SERIES}")
print(f"Degrees of freedom (between-series): {DF_BW}")
print("\nConcentration-Specific Settings:")
print("-" * 80)
for i, conc in enumerate(concentrations):
    within_cv = (C_std[i] / conc) * 100
    print(f"  {conc:.2e} Vp/mL:  BW CV = {BW_CV_PER_CONC[i]:5.1f}%,  "
          f"Within CV ≈ {within_cv:.1f}% (from C_std),  n = {N_REPLICATES_PER_CONC[i]}")
print("=" * 80)


# ============================================================================
# MEASURED VALIDATION DATA — fill in from your full validation study
# ============================================================================

# Between-series CV% measured at each concentration in the actual validation
MEASURED_BW_CV = [5.932, 6.981, 7.274, 3.121, 18.02, 23.20]

# Did the concentration level PASS in the actual validation TAE profile?
# True  = passed (expanded uncertainty within ±acceptance limit)
# False = failed
MEASURED_VALIDATION_PASS = [True, True, True, True, False, False]


# ============================================================================
# HEATMAP SETTINGS — easily adjustable
# ============================================================================

HM_BW_CV_MAX = 35     # upper end of BW CV sweep (%)
HM_N_STEPS   = 71     # number of BW CV steps  (resolution: HM_BW_CV_MAX / (HM_N_STEPS-1))
HM_N_TRIAL   = 100000  # MCM trials per heatmap cell (trade-off: speed vs smoothness)


# ============================================================================
# VALIDATION CRITERIA
# ============================================================================

MCM_MEAN_BIAS_LIMIT        = 30
MCM_ACCEPTANCE_RATE_TARGET = 90
MCM_INDIVIDUAL_BIAS_LIMIT  = 30

print("\nVALIDATION CRITERIA SETTINGS")
print("=" * 80)
print(f"MCM Mean Bias Limit:        ±{MCM_MEAN_BIAS_LIMIT}%")
print(f"MCM Acceptance Rate Target: ≥{MCM_ACCEPTANCE_RATE_TARGET}%")
print(f"MCM Individual Bias Limit:  ±{MCM_INDIVIDUAL_BIAS_LIMIT}%")
print("Chauvenet's Criterion: Applied to each MCM distribution")
print("=" * 80)


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def rinvchisq(n, df, variance_estimate):
    """
    Sample variances from the inverse chi-square distribution.
    Correctly models uncertainty in variance estimates from small samples.
    """
    alpha = df / 2.0
    beta  = alpha * variance_estimate
    return 1.0 / gamma.rvs(alpha, scale=1/beta, size=n)


def apply_chauvenet_criterion(data):
    chauvenet_table = {
        3: 1.38, 4: 1.54, 5: 1.65, 6: 1.73, 7: 1.80, 8: 1.86, 9: 1.91, 10: 1.96,
        15: 2.13, 20: 2.24, 25: 2.33, 30: 2.39, 50: 2.57, 100: 2.81, 300: 3.14,
        500: 3.29, 1000: 3.48, 2000: 3.71, 5000: 3.97, 10000: 4.16, 50000: 4.50,
        100000: 4.42
    }
    N = len(data)
    if N in chauvenet_table:
        chauvenet_factor = chauvenet_table[N]
    else:
        available = sorted(chauvenet_table.keys())
        if N > available[-1]:
            chauvenet_factor = chauvenet_table[available[-1]]
        elif N < available[0]:
            chauvenet_factor = chauvenet_table[available[0]]
        else:
            lower = max(s for s in available if s <= N)
            upper = min(s for s in available if s >= N)
            fl = chauvenet_table[lower]
            fu = chauvenet_table[upper]
            chauvenet_factor = fl + (fu - fl) * (N - lower) / (upper - lower)
    mean_val = np.mean(data)
    std_val  = np.std(data, ddof=1)
    mask     = np.abs(data - mean_val) <= chauvenet_factor * std_val
    cleaned  = data[mask]
    return cleaned, N - len(cleaned), chauvenet_factor


def find_reportable_range_by_interpolation(mcm_results, acceptance_target=90):
    """
    Find reportable range where the MCM acceptance rate (>= acceptance_target %
    of simulated points within the individual bias limit) is met.
    """
    from scipy.interpolate import interp1d
    results_sorted = sorted(mcm_results, key=lambda x: x['concentration'])
    concs       = np.array([r['concentration']   for r in results_sorted])
    acc_rates   = np.array([r['acceptance_rate'] for r in results_sorted])
    log_concs   = np.log10(concs)

    if len(concs) > 1:
        try:
            acc_interp = interp1d(log_concs, acc_rates, kind='linear', fill_value='extrapolate')
            log_fine   = np.linspace(log_concs.min(), log_concs.max(), 1000)
            acc_fine   = acc_interp(log_fine)
            conc_fine  = 10**log_fine
            valid_mask = acc_fine >= acceptance_target
            if np.any(valid_mask):
                idx = np.where(valid_mask)[0]
                return {
                    'lloq': conc_fine[idx[0]],
                    'uloq': conc_fine[idx[-1]],
                    'valid': True,
                }
        except Exception as e:
            print(f"  Warning: Interpolation failed: {e}")
    return {'valid': False}


# ----------------------------------------------------------------------------
# MCM core — direct Vp titer sampling with chi-square BW variance
# ----------------------------------------------------------------------------

def run_mcm_with_chi_square_precision(n_trial=100000):
    """
    Error propagation model:
        Csim = Cmeas + BW_error

    Cmeas ~ N(C_mean, C_std)         — within-run precision from measured SD
    BW_error ~ N(0, sd_BW)           — between-series error, shared across replicates
    sd_BW² ~ Inv-χ²(df_BW)          — chi-square variance uncertainty

    X_final = mean(Csim over n replicates)
    BW error is shared → does not reduce with n; within-run reduces by √n.
    """
    print(f"\nRunning MCM with {n_trial:,} trials per concentration...")
    print("Error propagation: Csim = Cmeas + BW_error")
    print("Within-run precision: C_std | BW variance: inverse chi-square sampling\n")

    mcm_distributions = []

    for i, true_conc in enumerate(concentrations):
        BW_CV        = BW_CV_PER_CONC[i]
        N_REPLICATES = N_REPLICATES_PER_CONC[i]
        within_cv    = (C_std[i] / true_conc) * 100

        print(f"Simulating {i+1}/{len(concentrations)}: {true_conc:.2e} Vp/mL")
        print(f"  C_mean = {C_mean[i]:.3e}, C_std = {C_std[i]:.3e} (within CV ≈ {within_cv:.1f}%)")
        print(f"  BW CV = {BW_CV:.1f}%, n = {N_REPLICATES}, df_BW = {DF_BW}")

        # Between-series variance — sampled from inverse chi-square
        var_BW_estimate     = (BW_CV / 100.0 * true_conc) ** 2
        variance_BW_samples = rinvchisq(n_trial, DF_BW, var_BW_estimate)
        sd_BW_samples       = np.sqrt(variance_BW_samples)

        print(f"  BW variance: mean={np.mean(variance_BW_samples):.2e}, "
              f"min={np.min(variance_BW_samples):.2e}, max={np.max(variance_BW_samples):.2e}")

        # BW error: one draw per trial, shared across all replicates in that series
        BW_errors = np.random.normal(0, sd_BW_samples)                           # (n_trial,)

        # Within-run variance is ALSO estimated from few replicates (df = DF_W).
        # Controlled by WITHIN_RUN_CHI_SQUARE so the heatmap uses the identical model.
        if WITHIN_RUN_CHI_SQUARE:
            DF_W               = DF_W_PER_CONC[i]
            variance_W_samples = rinvchisq(n_trial, DF_W, C_std[i] ** 2)
            sd_W_samples       = np.sqrt(variance_W_samples)                     # (n_trial,)
        else:
            sd_W_samples       = np.full(n_trial, C_std[i])

        # Within-run: independent draw per replicate per trial, with per-trial SD
        C_meas = C_mean[i] + np.random.normal(0, 1, (n_trial, N_REPLICATES)) \
                             * sd_W_samples[:, np.newaxis]                       # (n_trial, N_REPLICATES)

        # Csim = Cmeas + BW_error  (BW broadcast across replicates)
        X_replicates = C_meas + BW_errors[:, np.newaxis]                         # (n_trial, N_REPLICATES)
        X_mcm        = np.mean(X_replicates, axis=1)                             # (n_trial,)

        # Remove invalid values
        valid_mask      = np.isfinite(X_mcm) & (X_mcm > 0)
        X_clean_initial = X_mcm[valid_mask]
        print(f"  Valid simulations: {len(X_clean_initial):,}/{n_trial:,}")

        # Chauvenet criterion
        X_clean, outliers_removed, chauvenet_factor = apply_chauvenet_criterion(X_clean_initial)
        print(f"  Chauvenet factor: {chauvenet_factor:.2f} | "
              f"Outliers removed: {outliers_removed:,} ({outliers_removed/len(X_clean_initial)*100:.2f}%)")
        print(f"  Final points: {len(X_clean):,}\n")

        bias_distribution = ((X_clean - true_conc) / true_conc) * 100

        mcm_distributions.append({
            'concentration':         true_conc,
            'BW_CV':                 BW_CV,
            'N_REPLICATES':          N_REPLICATES,
            'X_mcm':                 X_clean,
            'X_mcm_original':        X_clean_initial,
            'mcm_bias_distribution': bias_distribution,
            'outliers_removed':      outliers_removed,
            'chauvenet_factor':      chauvenet_factor,
            'outlier_percentage':    outliers_removed / len(X_clean_initial) * 100,
            'variance_BW_samples':   variance_BW_samples,
        })

    return mcm_distributions


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def evaluate_mcm_validation_with_chauvenet(mcm_distributions):
    print(f"Evaluating MCM results:")
    print(f"  Mean bias limit:        ±{MCM_MEAN_BIAS_LIMIT}%")
    print(f"  Acceptance rate target: ≥{MCM_ACCEPTANCE_RATE_TARGET}%")
    print(f"  Expanded uncertainty:   Both bounds within ±{MCM_INDIVIDUAL_BIAS_LIMIT}%\n")

    mcm_results = []
    for mcm_data in mcm_distributions:
        conc      = mcm_data['concentration']
        bias_dist = mcm_data['mcm_bias_distribution']
        X_clean   = mcm_data['X_mcm']

        mean_bias       = np.mean(bias_dist)
        std_bias_rel    = np.std(bias_dist)
        acceptance_rate = (np.sum(np.abs(bias_dist) <= MCM_INDIVIDUAL_BIAS_LIMIT) / len(bias_dist)) * 100

        u_c      = np.std(bias_dist)
        U_90     = 1.645 * u_c
        eu_lower = mean_bias - U_90
        eu_upper = mean_bias + U_90

        pass_eu       = (eu_lower >= -MCM_INDIVIDUAL_BIAS_LIMIT) and (eu_upper <= MCM_INDIVIDUAL_BIAS_LIMIT)
        mb_valid      = abs(mean_bias) <= MCM_MEAN_BIAS_LIMIT
        ar_valid      = acceptance_rate >= MCM_ACCEPTANCE_RATE_TARGET
        overall_valid = mb_valid and ar_valid

        mcm_results.append({
            'concentration':             conc,
            'BW_CV':                     mcm_data['BW_CV'],
            'N_REPLICATES':              mcm_data['N_REPLICATES'],
            'mean_bias':                 mean_bias,
            'std_bias_relative':         std_bias_rel,
            'u_combined':                u_c,
            'expanded_uncertainty':      U_90,
            'eu_lower_90':               eu_lower,
            'eu_upper_90':               eu_upper,
            'mcm_abs_mean':              np.mean(X_clean),
            'mcm_abs_std':               np.std(X_clean),
            'mcm_abs_min':               np.min(X_clean),
            'mcm_abs_max':               np.max(X_clean),
            'acceptance_rate':           acceptance_rate,
            'pass_expanded_uncertainty': pass_eu,
            'mean_bias_valid':           mb_valid,
            'acceptance_rate_valid':     ar_valid,
            'mcm_overall_valid':         overall_valid,
            'bias_distribution':         bias_dist,
            'outliers_removed':          mcm_data['outliers_removed'],
            'chauvenet_factor':          mcm_data['chauvenet_factor'],
            'outlier_percentage':        mcm_data['outlier_percentage'],
            'original_data_size':        len(mcm_data['X_mcm_original']),
            'cleaned_data_size':         len(mcm_data['X_mcm']),
        })

        print(f"  {conc:.1e}: Mean bias = {mean_bias:6.1f}% ({'PASS' if mb_valid else 'FAIL'}), "
              f"Accept rate = {acceptance_rate:5.1f}% ({'PASS' if ar_valid else 'FAIL'}), "
              f"EU within ±{MCM_INDIVIDUAL_BIAS_LIMIT}% = {'YES' if pass_eu else 'NO'} "
              f"→ {'VALID' if overall_valid else 'INVALID'}")

    return mcm_results


# ----------------------------------------------------------------------------
# Plots — existing figure (unchanged)
# ----------------------------------------------------------------------------

def create_mcm_plots(mcm_results):
    mcm_rev  = mcm_results[::-1]
    fig      = plt.figure(figsize=(16, 12))
    concs    = [r['concentration']         for r in mcm_rev]
    labels   = [f'{c:.1e}'                 for c in concs]
    pos      = np.arange(len(concs))
    bias_dists = [r['bias_distribution']   for r in mcm_rev]
    eu_lower   = [r['eu_lower_90']         for r in mcm_rev]
    eu_upper   = [r['eu_upper_90']         for r in mcm_rev]
    means      = [r['mean_bias']           for r in mcm_rev]
    acc_rates  = [r['acceptance_rate']     for r in mcm_rev]
    acc_valid  = [r['acceptance_rate_valid']for r in mcm_rev]
    outlier_pct= [r['outlier_percentage']  for r in mcm_rev]
    n_reps     = [r['N_REPLICATES']        for r in mcm_rev]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    # --- Plot 1: Bias Distribution ---
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(pos, means,    'r-',  linewidth=3, marker='o', markersize=8, label='Mean Relative Bias', zorder=5)
    ax1.plot(pos, eu_upper, 'b--', linewidth=2, label='Expanded Uncertainty (k=2)', zorder=4)
    ax1.plot(pos, eu_lower, 'b--', linewidth=2, zorder=4)
    ax1.fill_between(pos, eu_lower, eu_upper, alpha=0.1, color='blue', zorder=1)

    max_vals = []
    for i, bias_vals in enumerate(bias_dists):
        n_scatter = min(200, len(bias_vals))
        idx  = np.random.choice(len(bias_vals), n_scatter, replace=False)
        vals = np.array(bias_vals)[idx]
        xjit = np.random.normal(i, 0.05, len(vals))
        ax1.scatter(xjit, vals, alpha=0.3, s=8, color=colors[i % len(colors)], zorder=2)
        max_vals.append(np.max(vals))

    for i, n_rep in enumerate(n_reps):
        ax1.text(i, max_vals[i] + 5, f'n={n_rep}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))

    ax1.axhline(y= MCM_INDIVIDUAL_BIAS_LIMIT, color='black', linestyle='--', linewidth=2,
                label=f'Acceptance Limits (±{MCM_INDIVIDUAL_BIAS_LIMIT}%)', zorder=3)
    ax1.axhline(y=-MCM_INDIVIDUAL_BIAS_LIMIT, color='black', linestyle='--', linewidth=2, zorder=3)
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=3)
    ax1.set_xticks(pos); ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_xlabel('Concentration (Vp/mL)'); ax1.set_ylabel('Relative Error (%)')
    ax1.set_title('MCM Bias Distribution\n(Direct Vp Titer Sampling — Csim = Cmeas + BW)')
    ax1.grid(True, alpha=0.3); ax1.legend(fontsize=10)

    # --- Plot 2: Acceptance Rate ---
    ax2 = plt.subplot(2, 2, 2)
    bar_colors = ['green' if r['acceptance_rate_valid'] else 'red' for r in mcm_rev]
    bars2 = ax2.bar(pos, acc_rates, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.axhline(y=90,                         color='green',  linestyle='--', linewidth=2, label='90% Excellent')
    ax2.axhline(y=MCM_ACCEPTANCE_RATE_TARGET, color='orange', linestyle='-',  linewidth=2,
                label=f'{MCM_ACCEPTANCE_RATE_TARGET}% Target')
    ax2.set_xticks(pos); ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_xlabel('Concentration (Vp/mL)'); ax2.set_ylabel(f'% Within ±{MCM_INDIVIDUAL_BIAS_LIMIT}% Bias')
    ax2.set_title('MCM Acceptance Rate')
    ax2.grid(True, alpha=0.3); ax2.legend(); ax2.set_ylim(0, 105)
    for bar, rate, valid in zip(bars2, acc_rates, acc_valid):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 1, f'{rate:.1f}%',
                 ha='center', va='bottom', fontweight='bold',
                 color='darkgreen' if valid else 'darkred')

    # --- Plot 3: Chauvenet Outlier Analysis ---
    ax3 = plt.subplot(2, 2, 3)
    bars3 = ax3.bar(pos, outlier_pct, color='coral', alpha=0.7, edgecolor='black', linewidth=1)
    ax3.axhline(y=1.0, color='green',  linestyle='--', linewidth=2, label='1% Good')
    ax3.axhline(y=2.5, color='orange', linestyle='--', linewidth=2, label='2.5% Acceptable')
    ax3.axhline(y=5.0, color='red',    linestyle='--', linewidth=2, label='5% High')
    ax3.set_xticks(pos); ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.set_xlabel('Concentration (Vp/mL)'); ax3.set_ylabel('Outliers Removed (%)')
    ax3.set_title('Chauvenet Criterion\nOutlier Removal Analysis')
    ax3.grid(True, alpha=0.3); ax3.legend()
    y_limit = max(max(outlier_pct) * 1.5, 0.15)
    ax3.set_ylim(0, y_limit)
    for bar, pct in zip(bars3, outlier_pct):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., min(h + 0.005, y_limit - 0.02),
                 f'{pct:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # --- Plot 4: Model Summary ---
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    param_lines = "\n    Concentration-Specific Parameters:\n"
    for i, conc in enumerate(concentrations):
        within_cv = (C_std[i] / conc) * 100
        param_lines += (f"    {conc:.1e}:  BW={BW_CV_PER_CONC[i]:.1f}%,  "
                        f"Within≈{within_cv:.1f}%,  n={N_REPLICATES_PER_CONC[i]}\n")

    summary_text = f"""
    PRECISION MODEL SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Using CHI-SQUARE Distribution
    for BW variance uncertainty

    Error Propagation:
      Csim = Cmeas + BW_error

      Cmeas ~ N(C_mean, C_std)
        within-run precision from SD

      BW_error ~ N(0, sd_BW)
        sd_BW² ~ Inv-χ²(df_BW)

      X_final = mean(Csim, n replicates)

    Global Parameters:
      Series (p) = {P_SERIES},  df_BW = {DF_BW}
{param_lines}
    BW error shared across replicates
    → does not reduce with n
    Within-run reduces by √n
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    ax4.text(0.02, 0.5, summary_text, fontsize=7, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    fig.tight_layout()
    fig.savefig('mcm_direct_vp_titer.png', dpi=300, bbox_inches='tight')
    fig.savefig('mcm_direct_vp_titer.pdf', bbox_inches='tight')
    plt.show()
    return fig


# ----------------------------------------------------------------------------
# Concordance heatmap — new figure
# ----------------------------------------------------------------------------

def create_concordance_heatmap():
    """
    Sweeps BW CV from 0 to HM_BW_CV_MAX for every concentration in one pass,
    computing MCM acceptance rate at each (concentration × BW CV) cell.

    What is shown:
      - Background colour: MCM acceptance rate (RdYlGn, 0–100 %)
      - Navy bar:          90 % acceptance threshold per row (= pass/fail boundary)
      - Coloured diamond ◆: measured BW CV from full validation study
            Cyan   = concordant, both PASS
            Red    = concordant, both FAIL
            Orange = discordant (MCM and validation disagree)

    n per concentration is taken from N_REPLICATES_PER_CONC; BW error is shared
    across replicates so increasing n only reduces within-run variance (C_std / √n),
    correctly narrowing the acceptance band at higher n.
    """
    bw_cv_sweep = np.linspace(0, HM_BW_CV_MAX, HM_N_STEPS)
    n_conc      = len(concentrations)
    acc_matrix  = np.zeros((n_conc, HM_N_STEPS))  # rows = conc (low→high), cols = BW CV

    step_size = HM_BW_CV_MAX / (HM_N_STEPS - 1)
    print(f"\nBuilding concordance heatmap...")
    print(f"  BW CV sweep : 0 → {HM_BW_CV_MAX}%  ({HM_N_STEPS} steps, Δ = {step_size:.2f}%/step)")
    print(f"  Trials/cell : {HM_N_TRIAL:,}   |   Total cells: {n_conc * HM_N_STEPS}")

    np.random.seed(42)

    for i, conc in enumerate(concentrations):
        N_REP = N_REPLICATES_PER_CONC[i]

        for j, bw_cv in enumerate(bw_cv_sweep):
            # Between-series variance: sampled via inverse chi-square
            if bw_cv == 0:
                BW_errors = np.zeros(HM_N_TRIAL)
            else:
                var_BW    = (bw_cv / 100.0 * conc) ** 2
                var_samp  = rinvchisq(HM_N_TRIAL, DF_BW, var_BW)
                BW_errors = np.random.normal(0, np.sqrt(var_samp))    # (HM_N_TRIAL,)

            # Within-run: independent per replicate per trial.
            # Uses the SAME model as the main MCM (WITHIN_RUN_CHI_SQUARE switch)
            # so the heatmap and the three-panel figure always agree.
            if WITHIN_RUN_CHI_SQUARE:
                var_W_samp = rinvchisq(HM_N_TRIAL, DF_W_PER_CONC[i], C_std[i] ** 2)
                sd_W_samp  = np.sqrt(var_W_samp)
            else:
                sd_W_samp  = np.full(HM_N_TRIAL, C_std[i])
            C_meas  = C_mean[i] + np.random.normal(0, 1, (HM_N_TRIAL, N_REP)) \
                                  * sd_W_samp[:, np.newaxis]                      # (trial, rep)
            X_reps  = C_meas + BW_errors[:, np.newaxis]                            # BW shared
            X_mcm   = np.mean(X_reps, axis=1)

            valid   = np.isfinite(X_mcm) & (X_mcm > 0)
            X_clean = X_mcm[valid]

            if len(X_clean) == 0:
                acc_matrix[i, j] = 0.0
            else:
                bias = ((X_clean - conc) / conc) * 100
                acc_matrix[i, j] = (np.sum(np.abs(bias) <= MCM_INDIVIDUAL_BIAS_LIMIT)
                                    / len(bias)) * 100

        print(f"  {conc:.2e} Vp/mL  (n={N_REP}) — done")

    # ---- Build figure -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 7))

    # Display with highest concentration at top → reverse rows
    acc_plot = acc_matrix[::-1, :]
    conc_labels_rev = [
        f'{c:.1e}   n={n}'
        for c, n in zip(concentrations[::-1], N_REPLICATES_PER_CONC[::-1])
    ]

    im = ax.imshow(
        acc_plot, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100, origin='upper',
        extent=[bw_cv_sweep[0], bw_cv_sweep[-1], n_conc - 0.5, -0.5]
    )

    # --- Navy vertical bar at 90% threshold per row -------------------------
    for i_plot, i_orig in enumerate(range(n_conc - 1, -1, -1)):
        row   = acc_matrix[i_orig]
        below = np.where(row < MCM_ACCEPTANCE_RATE_TARGET)[0]
        if len(below) > 0:
            thresh = bw_cv_sweep[below[0]]
            ax.plot([thresh, thresh], [i_plot - 0.45, i_plot + 0.45],
                    color='navy', linewidth=4.5, solid_capstyle='round', zorder=8)

    # --- Coloured diamond: measured BW CV + concordance assessment ----------
    # Horizontal gap (in BW CV % units) between each diamond and its text label.
    # Increase to push labels further right; decrease to pull them closer.
    LABEL_OFFSET = 0.9
    added_keys = set()
    for i_plot, i_orig in enumerate(range(n_conc - 1, -1, -1)):
        bw_meas   = MEASURED_BW_CV[i_orig]
        val_pass  = MEASURED_VALIDATION_PASS[i_orig]

        # MCM acceptance at nearest swept BW CV
        j_near    = np.argmin(np.abs(bw_cv_sweep - bw_meas))
        mcm_acc   = acc_matrix[i_orig, j_near]
        mcm_pass  = mcm_acc >= MCM_ACCEPTANCE_RATE_TARGET

        # Concordance colour logic
        if mcm_pass and val_pass:
            colour, lkey, lbl = '#00E5FF', 'cp', 'Concordant — both PASS'
        elif not mcm_pass and not val_pass:
            colour, lkey, lbl = '#FF3030', 'cf', 'Concordant — both FAIL'
        else:
            colour, lkey, lbl = '#FFA500', 'dc', 'Discordant (MCM ≠ Validation)'

        label_arg = lbl if lkey not in added_keys else ''
        added_keys.add(lkey)

        ax.scatter(bw_meas, i_plot, color=colour, s=230, zorder=10,
                   marker='D', edgecolors='white', linewidth=1.5, label=label_arg)
        # Annotation: measured BW% and MCM acceptance at that point
        ax.text(bw_meas + LABEL_OFFSET, i_plot,
                f'{bw_meas:.1f}%  ({mcm_acc:.0f}%)',
                va='center', ha='left', fontsize=7.5, fontweight='bold',
                color='white', zorder=11)

    # --- Colorbar -----------------------------------------------------------
    cbar = plt.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label(f'MCM Acceptance Rate  (% within ±{MCM_INDIVIDUAL_BIAS_LIMIT}% bias)',
                   fontsize=11)
    # Mark the 90% line on the colorbar (data coords = vmin…vmax = 0…100)
    cbar.ax.axhline(y=MCM_ACCEPTANCE_RATE_TARGET, color='navy',
                    linewidth=2, linestyle='--')
    cbar.ax.text(2.6, MCM_ACCEPTANCE_RATE_TARGET + 1.5,
                 f'{MCM_ACCEPTANCE_RATE_TARGET}%',
                 va='bottom', ha='left', fontsize=8.5, color='navy', fontweight='bold',
                 transform=cbar.ax.transData)

    # --- Legend -------------------------------------------------------------
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#00E5FF',
               markeredgecolor='white', markersize=7, label='Concordant — both PASS'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#FF3030',
               markeredgecolor='white', markersize=7, label='Concordant — both FAIL'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#FFA500',
               markeredgecolor='white', markersize=7, label='Discordant (MCM ≠ Validation)'),
        Line2D([0], [0], color='navy', linewidth=4.5,
               label=f'{MCM_ACCEPTANCE_RATE_TARGET}% acceptance threshold'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right',
              framealpha=0.92, edgecolor='gray', bbox_to_anchor=(0.995, 0.995))

    # --- Axes labels --------------------------------------------------------
    ax.set_yticks(range(n_conc))
    ax.set_yticklabels(conc_labels_rev, fontsize=10, family='monospace')
    ax.set_xlabel('Between-Series CV (%)', fontsize=12)
    ax.set_ylabel('Concentration (Vp/mL)', fontsize=12)

    ax.grid(False)

    fig.tight_layout()
    fig.savefig('mcm_concordance_heatmap.png', dpi=300, bbox_inches='tight')
    fig.savefig('mcm_concordance_heatmap.pdf', bbox_inches='tight')
    plt.show()

    return fig, acc_matrix


# ----------------------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------------------

def print_mcm_summary(mcm_results):
    print("\n" + "=" * 100)
    print("MCM VALIDATION SUMMARY — DIRECT Vp TITER SAMPLING (Csim = Cmeas + BW)")
    print("=" * 100)
    print(f"Mean Bias Criterion:        ±{MCM_MEAN_BIAS_LIMIT}%")
    print(f"Validity Criterion:         ≥{MCM_ACCEPTANCE_RATE_TARGET}% of simulated "
          f"points within ±{MCM_INDIVIDUAL_BIAS_LIMIT}% bias\n")

    mcm_rev = mcm_results[::-1]
    print(f"{'Concentration':<15} {'n':<4} {'BW CV':<8} {'Mean Bias (%)':<15} "
          f"{'Accept Rate':<13} {'Accept ≥' + str(MCM_ACCEPTANCE_RATE_TARGET) + '%':<14} {'Valid'}")
    print("-" * 100)

    valid_pass = []
    for r in mcm_rev:
        acc_status   = 'YES' if r['acceptance_rate_valid'] else 'NO'
        valid_status = 'YES' if r['mcm_overall_valid']     else 'NO'
        if r['acceptance_rate_valid']:
            valid_pass.append(r)
        print(f"{r['concentration']:.2e}  {r['N_REPLICATES']:<3}  {r['BW_CV']:>5.1f}%  "
              f"{r['mean_bias']:>12.2f}%  {r['acceptance_rate']:>10.1f}%  "
              f"{acc_status:>12}  {valid_status}")

    if valid_pass:
        interp_range = find_reportable_range_by_interpolation(mcm_rev, MCM_ACCEPTANCE_RATE_TARGET)
        valid_concs  = [r['concentration'] for r in valid_pass]
        print("\n" + "=" * 80)
        print("REPORTABLE RANGE")
        print("=" * 80)
        if interp_range['valid']:
            print(f"Interpolated Range: {interp_range['lloq']:.2e} – {interp_range['uloq']:.2e} Vp/mL")
        else:
            print(f"Discrete Range:     {min(valid_concs):.2e} – {max(valid_concs):.2e} Vp/mL")

    print("\n" + "=" * 100)


# ============================================================================
# MAIN
# ============================================================================

def run_chi_square_mcm_analysis():
    print("Starting MCM Analysis — Direct Vp Titer Sampling")
    print("Error propagation: Csim = Cmeas + BW_error")
    print("Within-run precision: C_std | BW variance: inverse chi-square sampling")
    print("=" * 80)
    np.random.seed(0)

    mcm_distributions = run_mcm_with_chi_square_precision(n_trial=100000)
    mcm_results       = evaluate_mcm_validation_with_chauvenet(mcm_distributions)

    print("\nGenerating main MCM plots...")
    fig_main = create_mcm_plots(mcm_results)

    print("\nGenerating concordance heatmap...")
    fig_heatmap, acc_matrix = create_concordance_heatmap()

    print_mcm_summary(mcm_results)
    print("\nMCM ANALYSIS COMPLETE")
    print("=" * 80)
    return mcm_results, fig_main, fig_heatmap


if __name__ == "__main__":
    mcm_results, fig_main, fig_heatmap = run_chi_square_mcm_analysis()
