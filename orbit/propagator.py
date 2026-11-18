import math
import numpy as np
from datetime import datetime, timezone, timedelta

# Astronomical Constants
AU_KM = 149597870.7  # km per AU
LUNAR_DISTANCE_AU = 0.002569555  # 1 LD in AU (~384,400 km)
EARTH_RADIUS_AU = 6378.137 / AU_KM
GAUSS_K = 0.01720209895  # Gaussian gravitational constant (au^(3/2) / day)

# Standard Mean Orbital Elements for Earth (J2000 heliocentric ecliptic)
EARTH_ELEMENTS = {
    "a": 1.00000011,      # au
    "e": 0.01671022,
    "i": 0.00005,         # deg
    "om": -11.26064,      # deg
    "w": 102.94719,       # deg
    "ma": 100.46435,      # deg at J2000
    "epoch_mjd": 51544.5  # Jan 1 2000 12:00 UTC
}

# Mars Mean Orbital Elements for outer reference
MARS_ELEMENTS = {
    "a": 1.52366231,
    "e": 0.09341233,
    "i": 1.85061,
    "om": 49.57854,
    "w": 286.46230,
    "ma": 19.3870,
    "epoch_mjd": 51544.5
}

def solve_kepler(M_rad, e, tol=1e-8, max_iter=30):
    """Solve Kepler's equation M = E - e*sin(E) for Eccentric Anomaly E using Newton-Raphson."""
    M_rad = M_rad % (2.0 * math.pi)
    E = M_rad if e < 0.8 else math.pi
    for _ in range(max_iter):
        f = E - e * math.sin(E) - M_rad
        f_prime = 1.0 - e * math.cos(E)
        if abs(f_prime) < 1e-12:
            break
        delta = f / f_prime
        E -= delta
        if abs(delta) < tol:
            break
    return E

def orbital_elements_to_vectors(a, e, i_deg, om_deg, w_deg):
    """Compute Gaussian orientation unit vectors P and Q in heliocentric ecliptic frame."""
    i = math.radians(i_deg)
    om = math.radians(om_deg)
    w = math.radians(w_deg)
    
    sin_om = math.sin(om)
    cos_om = math.cos(om)
    sin_w = math.sin(w)
    cos_w = math.cos(w)
    sin_i = math.sin(i)
    cos_i = math.cos(i)
    
    # Vector P points toward perihelion
    px = cos_om * cos_w - sin_om * sin_w * cos_i
    py = sin_om * cos_w + cos_om * sin_w * cos_i
    pz = sin_w * sin_i
    
    # Vector Q points 90 deg ahead of P in orbital plane
    qx = -cos_om * sin_w - sin_om * cos_w * cos_i
    qy = -sin_om * sin_w + cos_om * cos_w * cos_i
    qz = cos_w * sin_i
    
    return (px, py, pz), (qx, qy, qz)

def position_from_true_anomaly(nu_rad, a, e, P, Q):
    """Compute heliocentric position (X, Y, Z) in AU given true anomaly nu."""
    r = (a * (1.0 - e**2)) / (1.0 + e * math.cos(nu_rad))
    x_prime = r * math.cos(nu_rad)
    y_prime = r * math.sin(nu_rad)
    
    x = x_prime * P[0] + y_prime * Q[0]
    y = x_prime * P[1] + y_prime * Q[1]
    z = x_prime * P[2] + y_prime * Q[2]
    return x, y, z, r

def generate_orbit_path_points(a, e, i_deg, om_deg, w_deg, num_points=120):
    """Generate 3D coordinates along full ellipse for plotting."""
    P, Q = orbital_elements_to_vectors(a, e, i_deg, om_deg, w_deg)
    points = []
    for step in range(num_points + 1):
        nu = (2.0 * math.pi * step) / num_points
        x, y, z, r = position_from_true_anomaly(nu, a, e, P, Q)
        points.append({
            "x": round(x, 5),
            "y": round(y, 5),
            "z": round(z, 5),
            "r": round(r, 5)
        })
    return points

def datetime_to_mjd(dt):
    """Convert a UTC datetime to Modified Julian Date (MJD)."""
    # Julian Date from epoch
    epoch = datetime(1858, 11, 17, 0, 0, 0, tzinfo=timezone.utc)
    delta = dt - epoch
    return delta.total_seconds() / 86400.0

def position_at_time(elements, target_mjd):
    """Propagate Keplerian orbit to target MJD and calculate 3D position."""
    a = float(elements["a"])
    e = float(elements["e"])
    i_deg = float(elements.get("i", 0.0))
    om_deg = float(elements.get("om", 0.0))
    w_deg = float(elements.get("w", 0.0))
    ma_deg = float(elements.get("ma", 0.0))
    epoch_mjd = float(elements.get("epoch_mjd", 51544.5))
    
    # Mean motion n (radians per day)
    # n = k / a^(3/2)
    n = GAUSS_K / (a ** 1.5)
    
    dt_days = target_mjd - epoch_mjd
    m_target = math.radians(ma_deg) + n * dt_days
    
    # Solve Kepler equation
    E = solve_kepler(m_target, e)
    
    # True anomaly nu
    nu = 2.0 * math.atan2(math.sqrt(max(0.0, 1.0 + e)) * math.sin(E / 2.0),
                          math.sqrt(max(0.0, 1.0 - e)) * math.cos(E / 2.0))
                          
    P, Q = orbital_elements_to_vectors(a, e, i_deg, om_deg, w_deg)
    x, y, z, r = position_from_true_anomaly(nu, a, e, P, Q)
    
    return {
        "x": round(x, 5),
        "y": round(y, 5),
        "z": round(z, 5),
        "r": round(r, 5),
        "true_anomaly_deg": round(math.degrees(nu) % 360, 2)
    }

def propagate_asteroid_and_earth(asteroid_elements: dict, days_window=180, step_days=2):
    """
    Propagate both Earth and Asteroid over a selectable time window.
    Returns:
      - 3D paths for Asteroid and Earth orbits
      - Time-series positions (x, y, z, distance_au, distance_ld)
      - Closest approach event in window (date, distance, relative velocity)
      - Physical notices & methodology limitations
    """
    a = max(0.2, float(asteroid_elements.get("a", 1.4)))
    e = max(0.0001, min(0.999, float(asteroid_elements.get("e", 0.3))))
    i = max(0.0, float(asteroid_elements.get("i", 10.0)))
    om = float(asteroid_elements.get("om", 0.0))
    w = float(asteroid_elements.get("w", 0.0))
    ma = float(asteroid_elements.get("ma", 0.0))
    epoch_mjd = float(asteroid_elements.get("epoch_mjd", 59000.0))
    
    clean_ast_elements = {
        "a": a, "e": e, "i": i, "om": om, "w": w, "ma": ma, "epoch_mjd": epoch_mjd
    }
    
    # 1. Orbit full ellipses
    asteroid_orbit_path = generate_orbit_path_points(a, e, i, om, w, num_points=120)
    earth_orbit_path = generate_orbit_path_points(
        EARTH_ELEMENTS["a"], EARTH_ELEMENTS["e"], EARTH_ELEMENTS["i"],
        EARTH_ELEMENTS["om"], EARTH_ELEMENTS["w"], num_points=120
    )
    mars_orbit_path = generate_orbit_path_points(
        MARS_ELEMENTS["a"], MARS_ELEMENTS["e"], MARS_ELEMENTS["i"],
        MARS_ELEMENTS["om"], MARS_ELEMENTS["w"], num_points=120
    )
    
    # 2. Time steps propagation starting from current UTC time
    now_utc = datetime.now(timezone.utc)
    current_mjd = datetime_to_mjd(now_utc)
    
    time_series = []
    min_dist_au = float("inf")
    closest_approach_event = None
    
    total_steps = max(10, int(days_window / step_days))
    for step in range(total_steps + 1):
        target_dt = now_utc + timedelta(days=step * step_days)
        target_mjd = current_mjd + (step * step_days)
        
        pos_ast = position_at_time(clean_ast_elements, target_mjd)
        pos_earth = position_at_time(EARTH_ELEMENTS, target_mjd)
        
        # Distance between Earth and Asteroid
        dx = pos_ast["x"] - pos_earth["x"]
        dy = pos_ast["y"] - pos_earth["y"]
        dz = pos_ast["z"] - pos_earth["z"]
        dist_au = math.sqrt(dx**2 + dy**2 + dz**2)
        dist_ld = dist_au / LUNAR_DISTANCE_AU
        dist_km = dist_au * AU_KM
        
        entry = {
            "date": target_dt.strftime("%Y-%m-%d"),
            "days_from_now": step * step_days,
            "asteroid_pos": pos_ast,
            "earth_pos": pos_earth,
            "distance_au": round(dist_au, 5),
            "distance_ld": round(dist_ld, 2),
            "distance_km": round(dist_km, 0)
        }
        time_series.append(entry)
        
        if dist_au < min_dist_au:
            min_dist_au = dist_au
            closest_approach_event = {
                "date": target_dt.strftime("%Y-%m-%d"),
                "days_from_now": step * step_days,
                "min_distance_au": round(dist_au, 5),
                "min_distance_ld": round(dist_ld, 2),
                "min_distance_km": round(dist_km, 0),
                "asteroid_pos": pos_ast,
                "earth_pos": pos_earth,
                "is_close_approach": dist_au <= 0.05
            }
            
    # Perihelion and Aphelion distances
    perihelion_au = a * (1.0 - e)
    aphelion_au = a * (1.0 + e)
    orbital_period_yr = a ** 1.5
    orbital_period_days = orbital_period_yr * 365.25
    
    return {
        "generated_at": now_utc.isoformat(),
        "parameters": {
            "semi_major_axis_au": round(a, 4),
            "eccentricity": round(e, 4),
            "inclination_deg": round(i, 2),
            "longitude_node_deg": round(om, 2),
            "arg_perihelion_deg": round(w, 2),
            "perihelion_distance_au": round(perihelion_au, 4),
            "aphelion_distance_au": round(aphelion_au, 4),
            "orbital_period_yr": round(orbital_period_yr, 2),
            "orbital_period_days": round(orbital_period_days, 1)
        },
        "asteroid_orbit_path": asteroid_orbit_path,
        "earth_orbit_path": earth_orbit_path,
        "mars_orbit_path": mars_orbit_path,
        "current_positions": {
            "asteroid": time_series[0]["asteroid_pos"],
            "earth": time_series[0]["earth_pos"],
            "current_distance_au": time_series[0]["distance_au"],
            "current_distance_ld": time_series[0]["distance_ld"]
        },
        "closest_approach_in_window": closest_approach_event,
        "time_series": time_series,
        "disclaimer": (
            "NOTICE: This trajectory is computed using standard 2-body Keplerian orbital propagation in a heliocentric "
            "ecliptic J2000 frame. It serves as an intuitive geometric visualization of orbital paths and nominal approach geometry. "
            "It does NOT replace high-precision numerical N-body integrators (e.g., NASA JPL Horizons / DE441) which incorporate "
            "planetary gravitational perturbations, general relativity, solar radiation pressure, and Yarkovsky thermal drift."
        )
    }

if __name__ == "__main__":
    # Test sample with 99942 Apophis elements
    apophis = {
        "a": 0.9224, "e": 0.1912, "i": 3.33, "om": 204.04, "w": 126.40, "ma": 134.7, "epoch_mjd": 59000.0
    }
    result = propagate_asteroid_and_earth(apophis, days_window=90, step_days=5)
    print("Test Keplerian propagation for Apophis:")
    print("  Min distance in window:", result["closest_approach_in_window"])
    print("  Asteroid path points count:", len(result["asteroid_orbit_path"]))
    print("  Time series steps:", len(result["time_series"]))
