import re

def parse_distribution_section(lines):
    pattern = re.compile(r"\[([0-9.e+-]+),\s*([0-9.e+inf-]+)\):\s*(\d+)\s+values")
    parsed = []
    for line in lines:
        match = pattern.search(line)
        if match:
            lower = float(match.group(1))
            upper = float('inf') if 'inf' in match.group(2) else float(match.group(2))
            count = int(match.group(3))
            parsed.append((lower, upper, count))
    return parsed

def format_bin(lower, upper, count):
    upper_str = 'inf' if upper == float('inf') else f'{upper:.5e}'
    lower_str = f'{lower:.5e}'
    return f"[{lower_str}, {upper_str}): {count} values"

def convert_uncertainty_to_certainty_bins(unc_bins):
    cert_bins = []
    for lower, upper, count in unc_bins:
        if upper == float('inf'):
            cert_lower = 0.0
            cert_upper = 1.0
        else:
            cert_lower = max(0.0, min(1.0, 1.0 - upper))
            cert_upper = max(0.0, min(1.0, 1.0 - lower))

        cert_lower, cert_upper = min(cert_lower, cert_upper), max(cert_lower, cert_upper)

        if cert_lower == cert_upper:
            cert_upper += 1e-10

        cert_bins.append((cert_lower, cert_upper, count))
    return cert_bins

def process_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    output_lines = []
    inside_uncertainty_block = False
    unc_lines = []

    for line in lines:
        output_lines.append(line.rstrip())

        if line.strip().startswith("Uncertainty norm distribution:"):
            inside_uncertainty_block = True
            unc_lines = []
            continue

        if inside_uncertainty_block:
            if re.match(r"\[.*\):.*values", line):
                unc_lines.append(line)
            else:
                inside_uncertainty_block = False
                unc_bins = parse_distribution_section(unc_lines)
                cert_bins = convert_uncertainty_to_certainty_bins(unc_bins)
                output_lines.append("")
                output_lines.append("Certainty norm distribution:")
                output_lines.extend([format_bin(*b) for b in cert_bins])
                output_lines.append("")
                output_lines.append(line.rstrip())

    if inside_uncertainty_block:
        unc_bins = parse_distribution_section(unc_lines)
        cert_bins = convert_uncertainty_to_certainty_bins(unc_bins)
        output_lines.append("Certainty norm distribution:")
        output_lines.extend([format_bin(*b) for b in cert_bins])

    return "\n".join(output_lines)

processed_text = process_file("distribution_log.txt")
with open("distribution_log2.txt", "w") as f:
    f.write(processed_text)