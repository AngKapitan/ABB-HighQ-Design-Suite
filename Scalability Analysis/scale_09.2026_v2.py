import numpy as np
import math
from scipy import integrate
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# 1. CONSTANTS & SYSTEM PARAMETERS
# ==============================================================================
f_sw = 28.5e3
T_s = 1.0 / f_sw
mu_0 = 4 * math.pi * 1e-7
mu_r = 3300.0 
sigma = 5.96e7
L_m = 200e-6

# Core Loss Constants (Pre-calculated Steinmetz)
alpha = 1.4062949
beta = 2.7267175
k = 2.0420624
def kern(th): return (abs(math.cos(th))**alpha) * (2**(beta - alpha))
Kint, _ = integrate.quad(kern, 0, 2*math.pi)
ki = k / (2 * (math.pi**(alpha - 1)) * Kint)

# Target Current Density
J_TARGET = 10 / (0.3 * 6) * 1e6 

# ==============================================================================
# 2. PHYSICS MODULES
# ==============================================================================
def _calculate_R_prime_basic(w, l, h, mu_0):
    if w <= 0 or l <= 0 or h <= 0: return float('inf')
    try:
        log_term = (math.pi * h) / (4 * l)
        if log_term <= 0: return float('inf')
        val = mu_0 * (w / (2 * l) + (2 / math.pi) * (1 + math.log(log_term)))
        return 1 / val if val != 0 else float('inf')
    except: return float('inf')

def _calculate_3d_gap_reluctance(leg_width, core_thickness, gap_length, h_dimension, mu_0):
    if leg_width <= 0 or core_thickness <= 0: return float('inf')
    w1, t1 = leg_width, core_thickness
    w2, t2 = core_thickness, leg_width
    s1 = _calculate_R_prime_basic(w1, gap_length/2, h_dimension, mu_0) / (gap_length / (mu_0 * t1))
    s2 = _calculate_R_prime_basic(w2, gap_length/2, h_dimension, mu_0) / (gap_length / (mu_0 * t2))
    return s1 * s2 * (gap_length / (mu_0 * leg_width * core_thickness))

def calculate_core_loss(V_pri_pk, Np, A_calc, V_core):
    delta_B = (V_pri_pk * (T_s/2)) / (Np * A_calc)
    dBdt = delta_B / (T_s/2 - 400e-9)
    loss_density = ki * (dBdt**alpha) * (delta_B**(beta-alpha))
    return loss_density * V_core

def calculate_rdc_flatwise(l_cent, w_cent, w_wire, d_left_right, d_inter, Np, sigma, h_wire):
    w_1 = w_cent + 2 * (w_wire / 2 + d_left_right)
    l_1 = l_cent + 2 * (w_wire / 2 + d_left_right)
    lt_1 = 2 * (w_1 + l_1)
    common_difference = 8 * (w_wire + d_inter)
    total_sum = Np * (2 * lt_1 + (2*Np - 1) * common_difference) 
    return (1 / sigma) * total_sum / (h_wire * w_wire)

def calculate_rac_dowell(f_s, w_wire, r_dc, sigma, mu_0):
    if f_s == 0: return r_dc, float('inf'), 0, 1.0
    delta = math.sqrt(1 / (sigma * math.pi * f_s * mu_0))
    eta = w_wire / delta
    try:
        F_ac = (eta / 2) * ((math.sinh(eta) + math.sin(eta))/(math.cosh(eta) - math.cos(eta)) + (math.sinh(eta) - math.sin(eta))/(math.cosh(eta) + math.cos(eta)))
    except: F_ac = eta
    return F_ac * r_dc, delta, eta, F_ac

# ==============================================================================
# 3. BASELINE GENERATOR
# ==============================================================================
def get_baseline_design(V_in, I_in, Np, w_wire, h_wire, w_cent=10e-3, l_g=0.1e-3, d_inter=0.1e-3):
    d_left_right = 1e-3
    d_upper_down = 1e-3
    w_out = w_cent / 2
    l_window = h_wire + 2 * d_upper_down
    w_window = w_wire * 2 * Np + (2 * Np - 1) * d_inter + 2 * d_left_right
    
    target_reluctance = Np**2 / L_m
    low, high = 0.001, 0.5
    for _ in range(50):
        mid = (low + high) / 2
        A_calc = mid * w_cent
        l_1 = (l_window + w_out - l_g) / 2
        h_dim = l_1 - w_cent / 4
        if A_calc <= 0 or h_dim <= 0: high = mid; continue
        MPL = 2*l_window + 2*w_window + w_cent + math.pi/4*w_out - 2*l_g
        R_core = MPL / (mu_0 * mu_r * A_calc)
        Rg_tot = R_core + _calculate_3d_gap_reluctance(w_cent, mid, l_g, h_dim, mu_0) + 0.5*_calculate_3d_gap_reluctance(w_out, mid, l_g, h_dim, mu_0)
        if Rg_tot > target_reluctance: low = mid
        else: high = mid
            
    l_cent_final = (low + high) / 2
    A_calc_final = l_cent_final * w_cent
    B_act = (V_in * T_s) / (4 * Np * A_calc_final)
    
    return {
        "V_old": V_in, "I_old": I_in, "Np_old": Np, "w_cent_old": w_cent,
        "l_cent_old": l_cent_final, "A_calc_old": A_calc_final, "l_g_old": l_g,
        "w_wire_old": w_wire, "h_wire_old": h_wire, "d_inter_old": d_inter,
        "d_left_right_old": d_left_right, "d_upper_down_old": d_upper_down,
        "w_window_old": w_window, "l_window_old": l_window, "w_out_old": w_out,
        "B_act_baseline": B_act
    }

# ==============================================================================
# 4. DETERMINISTIC SCALER (Tunes L_g to maintain L_m)
# ==============================================================================
def scale_geometry(baseline, V_new, I_new, Np_fixed):
    lambda_V = math.sqrt(V_new / baseline['V_old'])
    lambda_I = math.sqrt(I_new / baseline['I_old'])
    
    # Scale Core (Voltage)
    w_cent = baseline['w_cent_old'] * lambda_V
    l_cent = baseline['l_cent_old'] * lambda_V
    w_out = baseline['w_out_old'] * lambda_V
    A_calc = w_cent * l_cent
    
    # Scale Window (Current - assuming rigorous geometric scaling)
    w_window = baseline['w_window_old'] * lambda_I
    l_window = baseline['l_window_old'] * lambda_I
    w_wire = baseline['w_wire_old'] * lambda_I
    h_wire = baseline['h_wire_old'] * lambda_I
    
    # Scale Clearances (Current - to preserve Ku geometric proportionality)
    d_inter = baseline['d_inter_old'] * lambda_I
    d_left_right = baseline['d_left_right_old'] * lambda_I
    d_upper_down = baseline['d_upper_down_old'] * lambda_I
    
    # Binary Search for Air Gap (l_g) to maintain L_m
    target_reluctance = Np_fixed**2 / L_m
    low_g, high_g = 0.0001, 0.1  # Search between 0.1mm and 100mm
    
    for _ in range(50):
        mid_g = (low_g + high_g) / 2
        l_1 = (l_window + w_out - mid_g) / 2
        h_dim = l_1 - w_cent / 4
        
        if l_1 <= 0 or h_dim <= 0:
            high_g = mid_g; continue
            
        MPL = 2*l_window + 2*w_window + w_cent + math.pi/4*w_out - 2*mid_g
        R_core = MPL / (mu_0 * mu_r * A_calc)
        
        Rg_center = _calculate_3d_gap_reluctance(w_cent, l_cent, mid_g, h_dim, mu_0)
        Rg_outer = _calculate_3d_gap_reluctance(w_out, l_cent, mid_g, h_dim, mu_0)
        Rg_tot = R_core + Rg_center + 0.5 * Rg_outer
        
        if Rg_tot > target_reluctance: high_g = mid_g
        else: low_g = mid_g
            
    l_g_final = (low_g + high_g) / 2
    
    # Final Geometric Verifications & Output Calculations
    B_act = (V_new * T_s) / (4 * Np_fixed * A_calc)
    MPL_final = 2*l_window + 2*w_window + w_cent + math.pi/4*w_out - 2*l_g_final
    
    overall_width = 2 * w_window + w_cent + 2 * w_out
    overall_height = l_window + 2 * w_out
    overall_length = l_cent + 2 * w_window
    V_box = overall_length * overall_width * overall_height
    
    non_ferrite_area = (w_window * l_window) * 2 + (l_g_final * (2 * w_out + w_cent))
    V_core = l_cent * (overall_height * overall_width - non_ferrite_area)

    return {
        "lambda_V": lambda_V, "lambda_I": lambda_I, "w_cent": w_cent,
        "l_cent": l_cent, "l_g": l_g_final, "A_calc": A_calc, "V_core": V_core,
        "w_wire": w_wire, "h_wire": h_wire, "d_inter": d_inter,
        "d_left_right": d_left_right, "d_upper_down": d_upper_down,
        "overall_width": overall_width, "overall_height": overall_height, 
        "overall_length": overall_length, "MPL": MPL_final, 
        "B_act": B_act, "V_box": V_box
    }

# ==============================================================================
# 5. EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # Define Baseline Targets
    V_OLD = 100.0
    I_OLD = 10.0
    N_FIXED = 8
    W_WIRE_OLD = 0.3e-3
    H_WIRE_OLD = I_OLD / (J_TARGET * W_WIRE_OLD)

    # 1. Establish the baseline design
    baseline = get_baseline_design(V_OLD, I_OLD, N_FIXED, W_WIRE_OLD, H_WIRE_OLD)
    
    # 2. Define New Targets
    V_NEW = 100.0*math.sqrt(10)  # Increased Voltage
    I_NEW = 10.0*math.sqrt(10)   # Increased Current
    
    # 3. Run Deterministic Scaler
    scaled = scale_geometry(baseline, V_NEW, I_NEW, N_FIXED)
    
    # 4. Calculate New Losses
    p_core_new = calculate_core_loss(V_NEW, N_FIXED, scaled['A_calc'], scaled['V_core'])
    rdc_new = calculate_rdc_flatwise(scaled['l_cent'], scaled['w_cent'], scaled['w_wire'], 
                                     scaled['d_left_right'], scaled['d_inter'], N_FIXED, 
                                     sigma, scaled['h_wire'])
    rac_new, delta, eta, F_ac = calculate_rac_dowell(f_sw, scaled['w_wire'], rdc_new, sigma, mu_0)
    p_cu_new = rac_new * (I_NEW**2)
    
    # 5. Output Results
    print(f"--- BASELINE DESIGN ({int(V_OLD)}V, {int(I_OLD)}A) ---")
    print(f"B_act: {baseline['B_act_baseline']:.4f} T")
    print(f"Core Area (A_c): {baseline['A_calc_old']*1e4:.2f} cm^2")
    print(f"Air Gap (l_g): {baseline['l_g_old']*1e3:.2f} mm")
    
    print(f"\n--- SCALED DESIGN ({int(V_NEW)}V, {int(I_NEW)}A) ---")
    print(f"Scaling Factors -> lambda_V: {scaled['lambda_V']:.3f}, lambda_I: {scaled['lambda_I']:.3f}")
    print(f"B_act: {scaled['B_act']:.4f} T")
    print(f"Core Area (A_c): {scaled['A_calc']*1e4:.2f} cm^2")
    print(f"New Air Gap (l_g) for 200uH: {scaled['l_g']*1e3:.2f} mm")
    print(f"New Wire Width: {scaled['w_wire']*1e3:.2f} mm")
    print(f"New Wire Height: {scaled['h_wire']*1e3:.2f} mm")
    
    print("\n--- CLEARANCES & INSULATION ---")
    print(f"Inter-winding (d_inter): {scaled['d_inter']*1e3:.2f} mm")
    print(f"Horizontal Clearance (d_left_right): {scaled['d_left_right']*1e3:.2f} mm")
    print(f"Vertical Clearance (d_upper_down): {scaled['d_upper_down']*1e3:.2f} mm")
    
    print("\n--- PHYSICAL DIMENSIONS ---")
    print(f"Magnetic Path Length (MPL): {scaled['MPL']*1e3:.2f} mm")
    print(f"Bounding Box Width: {scaled['overall_width']*1e3:.2f} mm")
    print(f"Bounding Box Height: {scaled['overall_height']*1e3:.2f} mm")
    print(f"Bounding Box Depth: {scaled['overall_length']*1e3:.2f} mm")
    print(f"Total Box Volume: {scaled['V_box']*1e6:.2f} cm^3")

    print("\n--- LOSS & AC RESISTANCE CHECK ---")
    print(f"Copper Loss: {p_cu_new:.1f} W")
    print(f"Core Loss: {p_core_new:.1f} W")
    print(f"Skin Depth: {delta*1e3:.3f} mm")
    
    # Check w_wire against skin depth for flatwise windings
    if scaled['w_wire'] > delta:
        print(f">> WARNING: w_wire ({scaled['w_wire']*1e3:.2f} mm) exceeds skin depth ({delta*1e3:.2f} mm). Dowell F_ac will spike! <<")