import math
from scipy import stats


def get_float(prompt, default=None):
    """Get a float from user input, using default if input is blank."""
    s = input(prompt).strip()
    if s == "" and default is not None:
        return default
    try:
        return float(s)
    except ValueError:
        print("Invalid number entered. Please try again.")
        return get_float(prompt, default)


def get_int(prompt, default=None):
    """Get an integer from user input, using default if input is blank."""
    s = input(prompt).strip()
    if s == "" and default is not None:
        return default
    try:
        return int(s)
    except ValueError:
        print("Invalid integer entered. Please try again.")
        return get_int(prompt, default)


def main():
    print("\nWelcome to the Statistics Helper Program!")
    print("This program helps solve problems from Confidence Intervals (Chapter 5) and Hypothesis Testing (Chapter 6).")
    print("--------------------------------------------------------")
    print("Select the type of analysis:")
    print("1. Confidence Interval")
    print("2. Hypothesis Test")
    analysis_choice = input("Enter 1 or 2: ").strip()

    if analysis_choice == '1':
        handle_confidence_interval()
    elif analysis_choice == '2':
        handle_hypothesis_test()
    else:
        print("Invalid choice. Please restart the program and choose 1 or 2.")


#############################################
# Confidence Interval Functions
#############################################

def handle_confidence_interval():
    print("\nConfidence Interval Options:")
    print("1. Population Mean (Large Sample, z-interval)")
    print("2. Population Mean (Small Sample, t-interval)")
    print("3. Population Proportion")
    print("4. Difference Between Two Means (Independent Samples, Large Sample)")
    print("5. Paired Data (t-interval on differences)")
    ci_choice = input("Enter your choice (1-5): ").strip()

    if ci_choice == '1':
        handle_ci_mean_large()
    elif ci_choice == '2':
        handle_ci_mean_small()
    elif ci_choice == '3':
        handle_ci_proportion()
    elif ci_choice == '4':
        handle_ci_diff_means()
    elif ci_choice == '5':
        handle_ci_paired()
    else:
        print("Invalid choice for confidence interval.")


def handle_ci_mean_large():
    print("\nYou selected Confidence Interval for Population Mean (Large Sample, z-interval).")
    print("Are you solving for:")
    print("1. The confidence interval (given sample mean, σ, n, and confidence level)")
    print("2. The required sample size (given σ, desired margin of error M, and confidence level)")
    choice = input("Enter 1 or 2: ").strip()
    if choice == '1':
        print("\nFormula: CI = x̄ ± z₍α/2₎ * (σ / √n)")
        print("Where:")
        print("  x̄ = sample mean")
        print("  σ  = population standard deviation")
        print("  n  = sample size")
        print("  z₍α/2₎ = z-critical value for the desired confidence level (α = 1 - confidence level)")
        x_bar = get_float("\nEnter the sample mean (x̄): ")
        sigma = get_float("Enter the population standard deviation (σ): ")
        n = get_float("Enter the sample size (n): ")
        conf = get_float("Enter the confidence level (e.g., 0.95): ")
        alpha = 1 - conf
        z = stats.norm.ppf(1 - alpha / 2)
        margin = z * sigma / math.sqrt(n)
        lower = x_bar - margin
        upper = x_bar + margin
        print(f"\nResult: {conf * 100:.1f}% Confidence Interval is ({lower:.4f}, {upper:.4f})")
    elif choice == '2':
        print("\nFormula: n = (z₍α/2₎ * σ / M)²")
        print("Where:")
        print("  σ  = population standard deviation")
        print("  M  = desired margin of error")
        print("  z₍α/2₎ = z-critical value for the desired confidence level (α = 1 - confidence level)")
        sigma = get_float("\nEnter the population standard deviation (σ): ")
        M = get_float("Enter the desired margin of error (M): ")
        conf = get_float("Enter the confidence level (e.g., 0.95): ")
        alpha = 1 - conf
        z = stats.norm.ppf(1 - alpha / 2)
        n_val = (z * sigma / M) ** 2
        n_required = math.ceil(n_val)
        print(f"\nResult: The required sample size is {n_required}")
    else:
        print("Invalid choice.")


def handle_ci_mean_small():
    print("\nYou selected Confidence Interval for Population Mean (Small Sample, t-interval).")
    print("Formula: CI = x̄ ± t₍α/2, n-1₎ * (s / √n)")
    print("Where:")
    print("  x̄ = sample mean")
    print("  s  = sample standard deviation")
    print("  n  = sample size")
    print("  t₍α/2, n-1₎ = t-critical value with n-1 degrees of freedom")
    x_bar = get_float("\nEnter the sample mean (x̄): ")
    s = get_float("Enter the sample standard deviation (s): ")
    n = get_int("Enter the sample size (n): ")
    conf = get_float("Enter the confidence level (e.g., 0.95): ")
    alpha = 1 - conf
    t_val = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_val * s / math.sqrt(n)
    lower = x_bar - margin
    upper = x_bar + margin
    print(f"\nResult: {conf * 100:.1f}% Confidence Interval is ({lower:.4f}, {upper:.4f})")


def handle_ci_proportion():
    print("\nYou selected Confidence Interval for a Population Proportion.")
    print("Are you solving for:")
    print("1. The confidence interval (given number of successes, n, and confidence level)")
    print("2. The required sample size (given desired margin of error and an estimated proportion)")
    choice = input("Enter 1 or 2: ").strip()
    if choice == '1':
        print("\nFormula: CI = p̂ ± z₍α/2₎ * sqrt( p̂*(1 - p̂) / n )")
        print("Where:")
        print("  p̂ = sample proportion (number of successes / n)")
        print("  n  = sample size")
        print("  z₍α/2₎ = z-critical value for the desired confidence level")
        X = get_int("\nEnter the number of successes: ")
        n = get_int("Enter the sample size (n): ")
        conf = get_float("Enter the confidence level (e.g., 0.95): ")
        p_hat = X / n
        alpha = 1 - conf
        z = stats.norm.ppf(1 - alpha / 2)
        margin = z * math.sqrt(p_hat * (1 - p_hat) / n)
        lower = p_hat - margin
        upper = p_hat + margin
        print(f"\nResult: {conf * 100:.1f}% Confidence Interval for the proportion is ({lower:.4f}, {upper:.4f})")
    elif choice == '2':
        print("\nFormula: n = (z² * p*(1-p)) / M²")
        print("Where:")
        print("  p = estimated population proportion (if unknown, use 0.5)")
        print("  M = desired margin of error")
        print("  z = z-critical value for the desired confidence level")
        p_input = input("\nEnter the estimated population proportion (or type 'ng' if not given): ").strip()
        if p_input.lower() in ['ng', 'not given', '']:
            p = 0.5
        else:
            try:
                p = float(p_input)
            except ValueError:
                p = 0.5
        M = get_float("Enter the desired margin of error (M): ")
        conf = get_float("Enter the confidence level (e.g., 0.95): ")
        alpha = 1 - conf
        z = stats.norm.ppf(1 - alpha / 2)
        n_val = (z ** 2 * p * (1 - p)) / (M ** 2)
        n_required = math.ceil(n_val)
        print(f"\nResult: The required sample size is {n_required}")
    else:
        print("Invalid choice.")


def handle_ci_diff_means():
    print(
        "\nYou selected Confidence Interval for the Difference Between Two Means (Independent Samples, Large Sample).")
    print("Formula: CI = (x̄₁ - x̄₂) ± z₍α/2₎ * sqrt( (σ₁²/n₁) + (σ₂²/n₂) )")
    print("Where:")
    print("  x̄₁, σ₁, n₁ = sample mean, population standard deviation, and sample size for group 1")
    print("  x̄₂, σ₂, n₂ = sample mean, population standard deviation, and sample size for group 2")
    X1_bar = get_float("\nEnter the sample mean for group 1 (x̄₁): ")
    sigma1 = get_float("Enter the population standard deviation for group 1 (σ₁): ")
    n1 = get_float("Enter the sample size for group 1 (n₁): ")
    X2_bar = get_float("Enter the sample mean for group 2 (x̄₂): ")
    sigma2 = get_float("Enter the population standard deviation for group 2 (σ₂): ")
    n2 = get_float("Enter the sample size for group 2 (n₂): ")
    conf = get_float("Enter the confidence level (e.g., 0.95): ")
    alpha = 1 - conf
    z = stats.norm.ppf(1 - alpha / 2)
    diff = X1_bar - X2_bar
    margin = z * math.sqrt((sigma1 ** 2 / n1) + (sigma2 ** 2 / n2))
    lower = diff - margin
    upper = diff + margin
    print(
        f"\nResult: {conf * 100:.1f}% Confidence Interval for the difference between means is ({lower:.4f}, {upper:.4f})")


def handle_ci_paired():
    print("\nYou selected Confidence Interval for Paired Data (t-interval on differences).")
    print("Formula: CI = d̄ ± t₍α/2, n-1₎ * (s_d / √n)")
    print("Where:")
    print("  d̄ = mean of the differences (each pair's difference)")
    print("  s_d = standard deviation of the differences")
    print("  n = number of pairs")
    print("  t₍α/2, n-1₎ = t-critical value with n-1 degrees of freedom")
    n = get_int("\nEnter the number of paired observations (n): ")
    diffs = []
    print("Enter the difference for each pair (first value - second value):")
    for i in range(n):
        d = get_float(f"Difference for pair {i + 1}: ")
        diffs.append(d)
    conf = get_float("Enter the confidence level (e.g., 0.95): ")
    mean_diff = sum(diffs) / n
    s_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / (n - 1))
    alpha = 1 - conf
    t_val = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_val * s_diff / math.sqrt(n)
    lower = mean_diff - margin
    upper = mean_diff + margin
    print(f"\nResult: {conf * 100:.1f}% Confidence Interval for the mean difference is ({lower:.4f}, {upper:.4f})")


#############################################
# Hypothesis Testing Functions
#############################################

def handle_hypothesis_test():
    print("\nHypothesis Test Options:")
    print("1. Population Mean (Large Sample, z-test)")
    print("2. Population Mean (Small Sample, t-test)")
    print("3. Population Proportion")
    print("4. Difference Between Two Means (Independent Samples, z-test)")
    print("5. Paired Data (t-test on differences)")
    print("6. Chi-Square Test (Categorical Data)")
    test_choice = input("Enter your choice (1-6): ").strip()

    if test_choice == '1':
        hypothesis_test_mean_z()
    elif test_choice == '2':
        hypothesis_test_mean_t()
    elif test_choice == '3':
        hypothesis_test_proportion()
    elif test_choice == '4':
        hypothesis_test_diff_means()
    elif test_choice == '5':
        hypothesis_test_paired()
    elif test_choice == '6':
        handle_chi_square_test()
    else:
        print("Invalid choice for hypothesis test.")


########################
# Numbered Output Routines
########################

def print_numbered_steps_mean_z(x_bar, mu0, sigma, n, alt, alpha_val, z, p_value):
    # 1) Hypotheses
    print("1) Set up null and alternative hypotheses:")
    print(f"   H0: μ = {mu0}")
    if alt == '>':
        print(f"   H1: μ > {mu0}")
    elif alt == '<':
        print(f"   H1: μ < {mu0}")
    else:
        print(f"   H1: μ ≠ {mu0}")

    # 2) Test statistic
    print("2) Compute the test statistic:")
    print(f"   z = (x̄ - μ0) / (σ / √n) = {z:.4f}")

    # 3) Calculate p-value
    print("3) Calculate the p-value:")
    print(f"   p-value = {p_value:.4f}")

    # 4) State the conclusion
    print(f"4) State the conclusion at α = {alpha_val}:")
    if p_value < alpha_val:
        print("   Conclusion: Reject H0.")
    else:
        print("   Conclusion: Fail to reject H0.")

    # 5) (Optional note) - large sample assumption
    print("5) Note: Large-sample z-test is valid if n is large or σ is known (population standard deviation).")


def print_numbered_steps_mean_t(x_bar, mu0, s, n, alt, alpha_val, t_stat, p_value):
    # 1) Hypotheses
    print("1) Set up null and alternative hypotheses:")
    print(f"   H0: μ = {mu0}")
    if alt == '>':
        print(f"   H1: μ > {mu0}")
    elif alt == '<':
        print(f"   H1: μ < {mu0}")
    else:
        print(f"   H1: μ ≠ {mu0}")

    # 2) Test statistic
    print("2) Compute the test statistic:")
    print(f"   t = (x̄ - μ0) / (s / √n) = {t_stat:.4f}")

    # 3) Calculate p-value
    print("3) Calculate the p-value:")
    print(f"   p-value = {p_value:.4f}")

    # 4) State the conclusion
    print(f"4) State the conclusion at α = {alpha_val}:")
    if p_value < alpha_val:
        print("   Conclusion: Reject H0.")
    else:
        print("   Conclusion: Fail to reject H0.")

    # 5) (Optional note)
    print("5) Note: This t-test requires the population to be approximately normal or n large enough.")


def print_numbered_steps_prop_z(X, n, p0, alt, alpha_val, z, p_value):
    # 1) Hypotheses
    print("1) Set up null and alternative hypotheses:")
    print(f"   H0: p = {p0}")
    if alt == '>':
        print(f"   H1: p > {p0}")
    elif alt == '<':
        print(f"   H1: p < {p0}")
    else:
        print(f"   H1: p ≠ {p0}")

    # 2) Test statistic
    print("2) Compute the test statistic:")
    print(f"   z = (p̂ - p0) / √[p0(1 - p0)/n] = {z:.4f}")

    # 3) Calculate p-value
    print("3) Calculate the p-value:")
    print(f"   p-value = {p_value:.4f}")

    # 4) State the conclusion
    print(f"4) State the conclusion at α = {alpha_val}:")
    if p_value < alpha_val:
        print("   Conclusion: Reject H0.")
    else:
        print("   Conclusion: Fail to reject H0.")

    # 5) (Optional note)
    print("5) Note: Large-sample proportion test typically requires n*p0 >= 10 and n*(1 - p0) >= 10.")


def print_numbered_steps_diff_means_z(X1_bar, X2_bar, sigma1, sigma2, n1, n2, diff0, alt, alpha_val, z, p_value):
    # 1) Hypotheses
    print("1) Set up null and alternative hypotheses:")
    print(f"   H0: μ1 - μ2 = {diff0}")
    if alt == '>':
        print(f"   H1: μ1 - μ2 > {diff0}")
    elif alt == '<':
        print(f"   H1: μ1 - μ2 < {diff0}")
    else:
        print(f"   H1: μ1 - μ2 ≠ {diff0}")

    # 2) Test statistic
    print("2) Compute the test statistic:")
    print(f"   z = [({X1_bar:.2f} - {X2_bar:.2f}) - {diff0}] / √[(σ1²/n1) + (σ2²/n2)] = {z:.4f}")

    # 3) Calculate p-value
    print("3) Calculate the p-value:")
    print(f"   p-value = {p_value:.4f}")

    # 4) State the conclusion
    print(f"4) State the conclusion at α = {alpha_val}:")
    if p_value < alpha_val:
        print("   Conclusion: Reject H0.")
    else:
        print("   Conclusion: Fail to reject H0.")

    # 5) (Optional note)
    print("5) Note: Large-sample z-test for difference of means assumes both sample sizes are large or σ’s known.")


def print_numbered_steps_paired_t(diffs, mu0, alt, alpha_val, t_stat, p_value):
    # 1) Hypotheses
    print("1) Set up null and alternative hypotheses:")
    print(f"   H0: μD = {mu0}")
    if alt == '>':
        print(f"   H1: μD > {mu0}")
    elif alt == '<':
        print(f"   H1: μD < {mu0}")
    else:
        print(f"   H1: μD ≠ {mu0}")

    # 2) Test statistic
    print("2) Compute the test statistic:")
    print(f"   t = (d̄ - μ0) / (s_d / √n) = {t_stat:.4f}")

    # 3) Calculate p-value
    print("3) Calculate the p-value:")
    print(f"   p-value = {p_value:.4f}")

    # 4) State the conclusion
    print(f"4) State the conclusion at α = {alpha_val}:")
    if p_value < alpha_val:
        print("   Conclusion: Reject H0.")
    else:
        print("   Conclusion: Fail to reject H0.")

    # 5) (Optional note)
    print("5) Note: If n is large, the population of differences need not be normal, and a z-test could be used.")


########################
# Specific Test Functions
########################

def hypothesis_test_mean_z():
    print("\nFormula: z = (x̄ - μ₀) / (σ / √n)")
    print("Where:")
    print("  x̄ = sample mean")
    print("  μ₀ = hypothesized population mean")
    print("  σ  = population standard deviation")
    print("  n  = sample size")
    x_bar = get_float("\nEnter the sample mean (x̄): ")
    mu0 = get_float("Enter the hypothesized population mean (μ₀): ")
    sigma = get_float("Enter the population standard deviation (σ): ")
    n = get_int("Enter the sample size (n): ")
    alt = input("Enter the alternative hypothesis ('>' for μ > μ₀, '<' for μ < μ₀, '!=' for μ ≠ μ₀): ").strip()
    alpha_val = get_float("Enter the significance level (e.g., 0.05) [default=0.05]: ", default=0.05)
    z = (x_bar - mu0) / (sigma / math.sqrt(n))

    if alt == '>':
        p_value = 1 - stats.norm.cdf(z)
    elif alt == '<':
        p_value = stats.norm.cdf(z)
    elif alt == '!=':
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        print("Invalid alternative hypothesis option.")
        return

    print()  # blank line
    print_numbered_steps_mean_z(x_bar, mu0, sigma, n, alt, alpha_val, z, p_value)


def hypothesis_test_mean_t():
    print("\nFormula: t = (x̄ - μ₀) / (s / √n)")
    print("Where:")
    print("  x̄ = sample mean")
    print("  μ₀ = hypothesized population mean")
    print("  s  = sample standard deviation")
    print("  n  = sample size (degrees of freedom = n - 1)")
    x_bar = get_float("\nEnter the sample mean (x̄): ")
    mu0 = get_float("Enter the hypothesized population mean (μ₀): ")
    s = get_float("Enter the sample standard deviation (s): ")
    n = get_int("Enter the sample size (n): ")
    alt = input("Enter the alternative hypothesis ('>' for μ > μ₀, '<' for μ < μ₀, '!=' for μ ≠ μ₀): ").strip()
    alpha_val = get_float("Enter the significance level (e.g., 0.05) [default=0.05]: ", default=0.05)
    t_stat = (x_bar - mu0) / (s / math.sqrt(n))

    if alt == '>':
        p_value = 1 - stats.t.cdf(t_stat, df=n - 1)
    elif alt == '<':
        p_value = stats.t.cdf(t_stat, df=n - 1)
    elif alt == '!=':
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    else:
        print("Invalid alternative hypothesis option.")
        return

    print()  # blank line
    print_numbered_steps_mean_t(x_bar, mu0, s, n, alt, alpha_val, t_stat, p_value)


def hypothesis_test_proportion():
    print("\nFormula: z = (p̂ - p₀) / sqrt( p₀*(1 - p₀) / n )")
    print("Where:")
    print("  p̂ = sample proportion (number of successes / n)")
    print("  p₀ = hypothesized population proportion")
    print("  n  = sample size")
    X = get_int("\nEnter the number of successes: ")
    n = get_int("Enter the sample size (n): ")
    p0 = get_float("Enter the hypothesized proportion (p₀): ")
    alt = input("Enter the alternative hypothesis ('>' for p > p₀, '<' for p < p₀, '!=' for p ≠ p₀): ").strip()
    alpha_val = get_float("Enter the significance level (e.g., 0.05) [default=0.05]: ", default=0.05)
    p_hat = X / n
    z = (p_hat - p0) / math.sqrt(p0 * (1 - p0) / n)

    if alt == '>':
        p_value = 1 - stats.norm.cdf(z)
    elif alt == '<':
        p_value = stats.norm.cdf(z)
    elif alt == '!=':
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        print("Invalid alternative hypothesis option.")
        return

    print()  # blank line
    print_numbered_steps_prop_z(X, n, p0, alt, alpha_val, z, p_value)


def hypothesis_test_diff_means():
    print("\nFormula: z = [(x̄₁ - x̄₂) - Δ₀] / sqrt( (σ₁²/n₁) + (σ₂²/n₂) )")
    print("Where:")
    print("  x̄₁ = sample mean of group 1")
    print("  σ₁  = population standard deviation for group 1")
    print("  n₁  = sample size for group 1")
    print("  x̄₂ = sample mean of group 2")
    print("  σ₂  = population standard deviation for group 2")
    print("  n₂  = sample size for group 2")
    print("  Δ₀  = hypothesized difference (often 0)")
    X1_bar = get_float("\nEnter the sample mean for group 1 (x̄₁): ")
    sigma1 = get_float("Enter the population standard deviation for group 1 (σ₁): ")
    n1 = get_float("Enter the sample size for group 1 (n₁): ")
    X2_bar = get_float("Enter the sample mean for group 2 (x̄₂): ")
    sigma2 = get_float("Enter the population standard deviation for group 2 (σ₂): ")
    n2 = get_float("Enter the sample size for group 2 (n₂): ")
    diff0 = get_float("Enter the hypothesized difference (Δ₀, often 0): ")
    alt = input(
        "Enter the alternative hypothesis ('>' for (μ₁-μ₂) > Δ₀, '<' for (μ₁-μ₂) < Δ₀, '!=' for ≠ Δ₀): ").strip()
    alpha_val = get_float("Enter the significance level (e.g., 0.05) [default=0.05]: ", default=0.05)

    diff = X1_bar - X2_bar
    se = math.sqrt((sigma1 ** 2 / n1) + (sigma2 ** 2 / n2))
    z = (diff - diff0) / se

    if alt == '>':
        p_value = 1 - stats.norm.cdf(z)
    elif alt == '<':
        p_value = stats.norm.cdf(z)
    elif alt == '!=':
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        print("Invalid alternative hypothesis option.")
        return

    print()  # blank line
    print_numbered_steps_diff_means_z(X1_bar, X2_bar, sigma1, sigma2, n1, n2, diff0, alt, alpha_val, z, p_value)


def hypothesis_test_paired():
    print("\nFormula: t = (d̄ - μ₀) / (s_d / √n)")
    print("Where:")
    print("  d̄ = mean of the differences (each pair's difference)")
    print("  μ₀ = hypothesized mean difference (often 0)")
    print("  s_d = standard deviation of the differences")
    print("  n = number of paired observations (degrees of freedom = n - 1)")
    n = get_int("\nEnter the number of paired observations (n): ")
    diffs = []
    print("Enter the difference for each pair (first value - second value):")
    for i in range(n):
        d = get_float(f"Difference for pair {i + 1}: ")
        diffs.append(d)
    mu0 = get_float("Enter the hypothesized mean difference (μ₀, often 0): ")
    alt = input(
        "Enter the alternative hypothesis ('>' for mean difference > μ₀, '<' for mean difference < μ₀, '!=' for ≠ μ₀): ").strip()
    alpha_val = get_float("Enter the significance level (e.g., 0.05) [default=0.05]: ", default=0.05)

    mean_diff = sum(diffs) / n
    s_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / (n - 1))
    t_stat = (mean_diff - mu0) / (s_diff / math.sqrt(n))

    if alt == '>':
        p_value = 1 - stats.t.cdf(t_stat, df=n - 1)
    elif alt == '<':
        p_value = stats.t.cdf(t_stat, df=n - 1)
    elif alt == '!=':
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    else:
        print("Invalid alternative hypothesis option.")
        return

    print()  # blank line
    print_numbered_steps_paired_t(diffs, mu0, alt, alpha_val, t_stat, p_value)


########################
# Chi-Square Functions
########################

def handle_chi_square_test():
    print("\nChi-Square Test Options:")
    print("1. Goodness-of-Fit Test")
    print("2. Test for Independence/Homogeneity")
    chi_choice = input("Enter 1 or 2: ").strip()
    if chi_choice == '1':
        handle_chi_square_goodness_of_fit()
    elif chi_choice == '2':
        handle_chi_square_independence()
    else:
        print("Invalid choice for Chi-Square Test.")


def handle_chi_square_goodness_of_fit():
    print("\nChi-Square Goodness-of-Fit Test")
    print("Formula: χ² = Σ ((Oᵢ - Eᵢ)² / Eᵢ) for i = 1 to k")
    print("Where:")
    print("  Oᵢ = observed frequency in category i")
    print("  Eᵢ = expected frequency in category i")
    k = get_int("\nEnter the number of categories (k): ")
    observed = []
    for i in range(k):
        obs = get_float(f"Enter the observed frequency for category {i + 1}: ")
        observed.append(obs)
    total = sum(observed)
    mode = input("Do you want to provide hypothesized probabilities for each category? (y/n): ").strip().lower()
    expected = []
    if mode == 'y':
        print("Enter the hypothesized probability for each category (they should sum to 1):")
        for i in range(k):
            p0 = get_float(f"Probability for category {i + 1}: ")
            expected.append(total * p0)
    else:
        print("Enter the expected frequency for each category:")
        for i in range(k):
            exp_val = get_float(f"Expected frequency for category {i + 1}: ")
            expected.append(exp_val)
    chi_square = sum(((o - e) ** 2 / e) for o, e in zip(observed, expected) if e != 0)
    df = k - 1

    # Numbered style output:
    print("\n1) Set up the null and alternative hypotheses:")
    print("   H0: The distribution of the categorical variable matches the expected (or hypothesized) distribution.")
    print("   H1: The distribution does NOT match the expected distribution.")

    print("2) Compute the test statistic (χ²):")
    print(f"   χ² = {chi_square:.4f}")

    print("3) Degrees of freedom:")
    print(f"   df = k - 1 = {df}")

    alpha_val = get_float("4) Enter the significance level (e.g., 0.05) [default=0.05]: ", default=0.05)
    p_value = 1 - stats.chi2.cdf(chi_square, df)
    print(f"   p-value = {p_value:.4f}")

    print("5) State the conclusion:")
    if p_value < alpha_val:
        print("   Conclusion: Reject H0.")
    else:
        print("   Conclusion: Fail to reject H0.")


def handle_chi_square_independence():
    print("\nChi-Square Test for Independence/Homogeneity")
    print("Formula: χ² = Σ ((Oᵢⱼ - Eᵢⱼ)² / Eᵢⱼ) over all cells")
    print("Where:")
    print("  Oᵢⱼ = observed frequency in cell (i,j)")
    print("  Eᵢⱼ = (row_totalᵢ * col_totalⱼ) / grand_total")
    r = get_int("\nEnter the number of rows: ")
    c = get_int("Enter the number of columns: ")
    print("Enter the observed frequencies for each cell (row by row):")
    observed = []
    for i in range(r):
        row = []
        for j in range(c):
            obs = get_float(f"Observed frequency for cell ({i + 1}, {j + 1}): ")
            row.append(obs)
        observed.append(row)
    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[i][j] for i in range(r)) for j in range(c)]
    grand_total = sum(row_totals)
    chi_square = 0
    for i in range(r):
        for j in range(c):
            E_ij = row_totals[i] * col_totals[j] / grand_total
            if E_ij != 0:
                chi_square += (observed[i][j] - E_ij) ** 2 / E_ij
    df = (r - 1) * (c - 1)

    # Numbered style output:
    print("\n1) Set up the null and alternative hypotheses:")
    print("   H0: The two categorical variables are independent (or the distributions are homogeneous).")
    print("   H1: The two categorical variables are dependent (or the distributions differ).")

    print("2) Compute the test statistic (χ²):")
    print(f"   χ² = {chi_square:.4f}")

    print("3) Degrees of freedom:")
    print(f"   df = (r - 1)(c - 1) = {df}")

    alpha_val = get_float("4) Enter the significance level (e.g., 0.05) [default=0.05]: ", default=0.05)
    p_value = 1 - stats.chi2.cdf(chi_square, df)
    print(f"   p-value = {p_value:.4f}")

    print("5) State the conclusion:")
    if p_value < alpha_val:
        print("   Conclusion: Reject H0.")
    else:
        print("   Conclusion: Fail to reject H0.")


#############################################

def conclusion(alpha, p_value):
    """(No longer used directly for the final output, but kept for reference.)"""
    if p_value < alpha:
        print("Conclusion: Reject the null hypothesis.\n")
    else:
        print("Conclusion: Fail to reject the null hypothesis.\n")


if __name__ == "__main__":
    main()
