// chronos-geospatial-engine.rs
//
// Chronos Geospatial / GIS Engine
// =================================
// A comprehensive geographic information systems library implementing
// geodetic mathematics, coordinate transformations, spatial indexing,
// and geometric algorithms from first principles.
//
// Modules:
//   1.  Coordinate Systems & Datums (WGS-84, ECEF, UTM)
//   2.  Geodetic Calculations (Vincenty's formulae, great-circle)
//   3.  Coordinate Transformations (geodetic ↔ ECEF ↔ ENU)
//   4.  UTM Projection (Transverse Mercator)
//   5.  Web Mercator (EPSG:3857) & Tile Math
//   6.  Geometry Primitives (Point, LineString, Polygon, BBox)
//   7.  Spatial Predicates (contains, intersects, within, touches)
//   8.  Computational Geometry (convex hull, area, centroid, simplification)
//   9.  Spatial Indexing (R-tree, quadtree, geohash)
//  10.  Routing (Dijkstra on road graph, A* with haversine heuristic)
//  11.  Raster Operations (DEM interpolation, slope/aspect)
//  12.  GeoJSON-like serialisation

use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::cmp::Ordering;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// 1. FUNDAMENTAL CONSTANTS & WGS-84 ELLIPSOID
// ─────────────────────────────────────────────────────────────────────────────

/// WGS-84 ellipsoid parameters (the standard GPS datum).
pub struct Wgs84;

impl Wgs84 {
    /// Semi-major axis a (equatorial radius) in metres.
    pub const A: f64 = 6_378_137.0;
    /// Flattening f = (a - b) / a.
    pub const F: f64 = 1.0 / 298.257_223_563;
    /// Semi-minor axis b = a(1-f).
    pub const B: f64 = 6_356_752.314_245_179;
    /// First eccentricity squared e² = 1 - (b/a)² = 2f - f².
    pub const E2: f64 = 2.0 * Self::F - Self::F * Self::F;
    /// Second eccentricity squared e'² = (a² - b²) / b².
    pub const EP2: f64 = (Self::A * Self::A - Self::B * Self::B) / (Self::B * Self::B);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. COORDINATE TYPES
// ─────────────────────────────────────────────────────────────────────────────

/// Geographic coordinate (latitude, longitude in decimal degrees, altitude in metres).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatLon {
    pub lat: f64,  // degrees, positive = North
    pub lon: f64,  // degrees, positive = East
    pub alt: f64,  // metres above ellipsoid
}

impl LatLon {
    pub fn new(lat: f64, lon: f64) -> Self { LatLon { lat, lon, alt: 0.0 } }
    pub fn with_alt(lat: f64, lon: f64, alt: f64) -> Self { LatLon { lat, lon, alt } }
    pub fn lat_rad(&self) -> f64 { self.lat.to_radians() }
    pub fn lon_rad(&self) -> f64 { self.lon.to_radians() }
}

/// Earth-Centred Earth-Fixed (ECEF) Cartesian coordinates in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ecef { pub x: f64, pub y: f64, pub z: f64 }

/// East-North-Up (ENU) local tangent plane coordinates (metres).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Enu { pub e: f64, pub n: f64, pub u: f64 }

/// Universal Transverse Mercator coordinate.
#[derive(Debug, Clone, Copy)]
pub struct Utm {
    pub zone:       u8,    // 1..60
    pub north:      bool,  // true = northern hemisphere
    pub easting:    f64,   // metres
    pub northing:   f64,   // metres
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. GEODETIC ↔ ECEF CONVERSIONS
// ─────────────────────────────────────────────────────────────────────────────

/// Convert geodetic (lat, lon, alt) to ECEF.
/// X = (N + h)·cos(φ)·cos(λ)
/// Y = (N + h)·cos(φ)·sin(λ)
/// Z = (N(1-e²) + h)·sin(φ)
/// where N = a / √(1 - e²·sin²φ)  (prime vertical radius of curvature)
pub fn geodetic_to_ecef(p: &LatLon) -> Ecef {
    let phi = p.lat_rad();
    let lam = p.lon_rad();
    let h   = p.alt;
    let sin_phi = phi.sin();
    let cos_phi = phi.cos();
    let n = Wgs84::A / (1.0 - Wgs84::E2 * sin_phi * sin_phi).sqrt();
    Ecef {
        x: (n + h) * cos_phi * lam.cos(),
        y: (n + h) * cos_phi * lam.sin(),
        z: (n * (1.0 - Wgs84::E2) + h) * sin_phi,
    }
}

/// Convert ECEF to geodetic using Bowring's iterative method.
/// Converges to <0.1 mm accuracy in 3–4 iterations.
pub fn ecef_to_geodetic(ecef: &Ecef) -> LatLon {
    let x = ecef.x; let y = ecef.y; let z = ecef.z;
    let p = (x * x + y * y).sqrt();
    let lon = y.atan2(x);

    // Bowring's initial approximation
    let mut lat = (z / p * (1.0 + Wgs84::EP2 * Wgs84::B / (p * p + z * z).sqrt())).atan();
    let mut h;

    for _ in 0..10 {
        let sin_lat = lat.sin();
        let cos_lat = lat.cos();
        let n = Wgs84::A / (1.0 - Wgs84::E2 * sin_lat * sin_lat).sqrt();
        h = if cos_lat.abs() > 1e-9 { p / cos_lat - n } else { z / sin_lat - n * (1.0 - Wgs84::E2) };
        let lat_new = ((z + Wgs84::E2 * n * sin_lat) / p).atan();
        if (lat_new - lat).abs() < 1e-13 { lat = lat_new; break; }
        lat = lat_new;
        let _ = h;
    }
    let sin_lat = lat.sin();
    let n = Wgs84::A / (1.0 - Wgs84::E2 * sin_lat * sin_lat).sqrt();
    let h = if lat.cos().abs() > 1e-9 { p / lat.cos() - n }
            else { z / sin_lat - n * (1.0 - Wgs84::E2) };

    LatLon { lat: lat.to_degrees(), lon: lon.to_degrees(), alt: h }
}

/// Convert ECEF to local East-North-Up (ENU) frame at reference point `ref_pt`.
pub fn ecef_to_enu(point: &Ecef, ref_pt: &LatLon) -> Enu {
    let ref_ecef = geodetic_to_ecef(ref_pt);
    let dx = point.x - ref_ecef.x;
    let dy = point.y - ref_ecef.y;
    let dz = point.z - ref_ecef.z;

    let phi = ref_pt.lat_rad();
    let lam = ref_pt.lon_rad();
    let sin_phi = phi.sin(); let cos_phi = phi.cos();
    let sin_lam = lam.sin(); let cos_lam = lam.cos();

    // Rotation matrix rows [East; North; Up]
    let e =  -sin_lam * dx + cos_lam * dy;
    let n_val = -sin_phi * cos_lam * dx - sin_phi * sin_lam * dy + cos_phi * dz;
    let u =   cos_phi * cos_lam * dx + cos_phi * sin_lam * dy + sin_phi * dz;
    Enu { e, n: n_val, u }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. GEODETIC DISTANCE CALCULATIONS
// ─────────────────────────────────────────────────────────────────────────────

/// Haversine formula: great-circle distance between two points on a sphere.
/// Assumes spherical Earth with radius 6,371,000 m.
/// Fast but ~0.5% error vs. Vincenty on oblate spheroid.
pub fn haversine_m(a: &LatLon, b: &LatLon) -> f64 {
    const R: f64 = 6_371_000.0;
    let dlat = (b.lat - a.lat).to_radians();
    let dlon = (b.lon - a.lon).to_radians();
    let lat1 = a.lat.to_radians();
    let lat2 = b.lat.to_radians();
    let sin_dlat = (dlat / 2.0).sin();
    let sin_dlon = (dlon / 2.0).sin();
    let h = sin_dlat * sin_dlat + lat1.cos() * lat2.cos() * sin_dlon * sin_dlon;
    2.0 * R * h.sqrt().asin()
}

/// Vincenty's inverse formula: geodesic distance on WGS-84 ellipsoid.
/// Accurate to within 0.5 mm. Iterates until convergence.
/// Returns (distance_m, initial_bearing_deg, final_bearing_deg).
pub fn vincenty_inverse(a: &LatLon, b: &LatLon) -> (f64, f64, f64) {
    let phi1 = a.lat_rad(); let phi2 = b.lat_rad();
    let l    = (b.lon - a.lon).to_radians();

    let tan_u1 = (1.0 - Wgs84::F) * phi1.tan();
    let tan_u2 = (1.0 - Wgs84::F) * phi2.tan();
    let cos_u1 = 1.0 / (1.0 + tan_u1 * tan_u1).sqrt();
    let sin_u1 = tan_u1 * cos_u1;
    let cos_u2 = 1.0 / (1.0 + tan_u2 * tan_u2).sqrt();
    let sin_u2 = tan_u2 * cos_u2;

    let mut lam = l;
    let mut cos2_sigma_m = 0.0_f64;
    let mut sin_sigma = 0.0_f64;
    let mut cos_sigma = 0.0_f64;
    let mut sigma = 0.0_f64;
    let mut sin_alpha = 0.0_f64;
    let mut cos2_alpha = 0.0_f64;
    let mut prev_lam = 0.0_f64;

    for _ in 0..1000 {
        prev_lam = lam;
        let sin_lam = lam.sin();
        let cos_lam = lam.cos();
        sin_sigma = ((cos_u2 * sin_lam).powi(2)
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam).powi(2)).sqrt();
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam;
        sigma = sin_sigma.atan2(cos_sigma);
        sin_alpha = if sin_sigma.abs() < 1e-15 { 0.0 }
                    else { cos_u1 * cos_u2 * sin_lam / sin_sigma };
        cos2_alpha = 1.0 - sin_alpha * sin_alpha;
        cos2_sigma_m = if cos2_alpha.abs() < 1e-15 { 0.0 }
                       else { cos_sigma - 2.0 * sin_u1 * sin_u2 / cos2_alpha };
        let c = Wgs84::F / 16.0 * cos2_alpha * (4.0 + Wgs84::F * (4.0 - 3.0 * cos2_alpha));
        lam = l + (1.0 - c) * Wgs84::F * sin_alpha
            * (sigma + c * sin_sigma * (cos2_sigma_m
               + c * cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)));
        if (lam - prev_lam).abs() < 1e-12 { break; }
    }

    let u2 = cos2_alpha * Wgs84::EP2;
    let aa = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)));
    let bb = u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)));
    let delta_sigma = bb * sin_sigma * (cos2_sigma_m
        + bb / 4.0 * (cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)
        - bb / 6.0 * cos2_sigma_m * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                    * (-3.0 + 4.0 * cos2_sigma_m * cos2_sigma_m)));
    let distance = Wgs84::B * aa * (sigma - delta_sigma);

    let sin_lam = lam.sin();
    let cos_lam = lam.cos();
    let bearing1 = (cos_u2 * sin_lam)
        .atan2(cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam)
        .to_degrees();
    let bearing2 = (cos_u1 * sin_lam)
        .atan2(-sin_u1 * cos_u2 + cos_u1 * sin_u2 * cos_lam)
        .to_degrees();

    (distance, (bearing1 + 360.0) % 360.0, (bearing2 + 360.0) % 360.0)
}

/// Vincenty direct: given starting point, bearing, and distance, find endpoint.
pub fn vincenty_direct(start: &LatLon, bearing_deg: f64, distance_m: f64) -> LatLon {
    let phi1    = start.lat_rad();
    let alpha1  = bearing_deg.to_radians();
    let sin_a1  = alpha1.sin(); let cos_a1  = alpha1.cos();
    let tan_u1  = (1.0 - Wgs84::F) * phi1.tan();
    let cos_u1  = 1.0 / (1.0 + tan_u1 * tan_u1).sqrt();
    let sin_u1  = tan_u1 * cos_u1;
    let sigma1  = tan_u1.atan2(cos_a1);
    let sin_alpha  = cos_u1 * sin_a1;
    let cos2_alpha = 1.0 - sin_alpha * sin_alpha;
    let u2 = cos2_alpha * Wgs84::EP2;
    let aa = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)));
    let bb = u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)));
    let mut sigma = distance_m / (Wgs84::B * aa);

    for _ in 0..100 {
        let cos2_sigma_m = (2.0 * sigma1 + sigma).cos();
        let delta_sigma = bb * sigma.sin() * (cos2_sigma_m
            + bb / 4.0 * (sigma.cos() * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)
            - bb / 6.0 * cos2_sigma_m * (-3.0 + 4.0 * sigma.sin().powi(2))
                        * (-3.0 + 4.0 * cos2_sigma_m * cos2_sigma_m)));
        let sigma_new = distance_m / (Wgs84::B * aa) + delta_sigma;
        if (sigma_new - sigma).abs() < 1e-12 { sigma = sigma_new; break; }
        sigma = sigma_new;
    }

    let cos2_sigma_m = (2.0 * sigma1 + sigma).cos();
    let phi2 = (sin_u1 * sigma.cos() + cos_u1 * sigma.sin() * cos_a1).atan2(
        (1.0 - Wgs84::F) * (sin_alpha.powi(2)
        + (sin_u1 * sigma.sin() - cos_u1 * sigma.cos() * cos_a1).powi(2)).sqrt());
    let lam_p = (sigma.sin() * sin_a1).atan2(
        cos_u1 * sigma.cos() - sin_u1 * sigma.sin() * cos_a1);
    let c = Wgs84::F / 16.0 * cos2_alpha * (4.0 + Wgs84::F * (4.0 - 3.0 * cos2_alpha));
    let l = lam_p - (1.0 - c) * Wgs84::F * sin_alpha
        * (sigma + c * sigma.sin() * (cos2_sigma_m
           + c * sigma.cos() * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)));

    LatLon::new(phi2.to_degrees(), start.lon + l.to_degrees())
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. UTM PROJECTION (TRANSVERSE MERCATOR)
// ─────────────────────────────────────────────────────────────────────────────

/// Determine UTM zone number from longitude.
pub fn utm_zone(lon_deg: f64) -> u8 { ((lon_deg + 180.0) / 6.0) as u8 + 1 }

/// Central meridian of a UTM zone.
pub fn utm_central_meridian(zone: u8) -> f64 { (zone as f64 - 1.0) * 6.0 - 177.0 }

/// Project geodetic coordinates to UTM using the Transverse Mercator formulae.
/// Scale factor k0 = 0.9996. False easting = 500,000 m; false northing (S) = 10,000,000 m.
pub fn latlon_to_utm(p: &LatLon) -> Utm {
    let zone    = utm_zone(p.lon);
    let lam0    = utm_central_meridian(zone).to_radians();
    let phi     = p.lat_rad();
    let lam     = p.lon_rad();
    let k0      = 0.9996;
    let fe      = 500_000.0;
    let fn_val  = if p.lat < 0.0 { 10_000_000.0 } else { 0.0 };

    let e2 = Wgs84::E2;
    let e4 = e2 * e2;
    let e6 = e4 * e2;

    let n = Wgs84::A / (1.0 - e2 * phi.sin().powi(2)).sqrt();
    let t = phi.tan().powi(2);
    let c = e2 / (1.0 - e2) * phi.cos().powi(2);
    let a_coeff = phi.cos() * (lam - lam0);

    // Meridional arc
    let m = Wgs84::A * (
          (1.0 - e2/4.0 - 3.0*e4/64.0 - 5.0*e6/256.0) * phi
        - (3.0*e2/8.0 + 3.0*e4/32.0 + 45.0*e6/1024.0) * (2.0*phi).sin()
        + (15.0*e4/256.0 + 45.0*e6/1024.0) * (4.0*phi).sin()
        - (35.0*e6/3072.0) * (6.0*phi).sin()
    );

    let easting = k0 * n * (a_coeff
        + (1.0 - t + c) * a_coeff.powi(3) / 6.0
        + (5.0 - 18.0*t + t*t + 72.0*c - 58.0*e2/(1.0-e2)) * a_coeff.powi(5) / 120.0)
        + fe;

    let northing = k0 * (m + n * phi.tan() * (
          a_coeff.powi(2) / 2.0
        + (5.0 - t + 9.0*c + 4.0*c*c) * a_coeff.powi(4) / 24.0
        + (61.0 - 58.0*t + t*t + 600.0*c - 330.0*e2/(1.0-e2)) * a_coeff.powi(6) / 720.0
    )) + fn_val;

    Utm { zone, north: p.lat >= 0.0, easting, northing }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. WEB MERCATOR (EPSG:3857) & TILE MATH
// ─────────────────────────────────────────────────────────────────────────────

/// Convert geodetic to Web Mercator (EPSG:3857) XY in metres.
/// Used by Google Maps, OpenStreetMap, etc.
/// Valid for latitudes -85.051° to +85.051°.
pub fn latlon_to_web_mercator(p: &LatLon) -> (f64, f64) {
    const R: f64 = 6_378_137.0;
    let x = R * p.lon_rad();
    let y = R * ((PI / 4.0 + p.lat_rad() / 2.0).tan()).ln();
    (x, y)
}

/// Convert Web Mercator XY back to geodetic.
pub fn web_mercator_to_latlon(x: f64, y: f64) -> LatLon {
    const R: f64 = 6_378_137.0;
    let lon = (x / R).to_degrees();
    let lat = (2.0 * (y / R).exp().atan() - PI / 2.0).to_degrees();
    LatLon::new(lat, lon)
}

/// Slippy map tile XY from lat/lon at a given zoom level.
pub fn latlon_to_tile(p: &LatLon, zoom: u32) -> (u32, u32) {
    let n = (1 << zoom) as f64;
    let x = ((p.lon + 180.0) / 360.0 * n) as u32;
    let lat_r = p.lat_rad();
    let y = ((1.0 - (lat_r.tan() + 1.0 / lat_r.cos()).ln() / PI) / 2.0 * n) as u32;
    (x, y)
}

/// Tile bounding box (lat/lon) for a given tile at zoom level.
pub fn tile_bounds(tx: u32, ty: u32, zoom: u32) -> (LatLon, LatLon) {
    let n = (1 << zoom) as f64;
    let lon_min = tx as f64 / n * 360.0 - 180.0;
    let lon_max = (tx + 1) as f64 / n * 360.0 - 180.0;
    let lat_max_r = (PI * (1.0 - 2.0 * ty as f64 / n)).sinh().atan();
    let lat_min_r = (PI * (1.0 - 2.0 * (ty + 1) as f64 / n)).sinh().atan();
    (
        LatLon::new(lat_min_r.to_degrees(), lon_min),
        LatLon::new(lat_max_r.to_degrees(), lon_max),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. GEOMETRY PRIMITIVES
// ─────────────────────────────────────────────────────────────────────────────

/// 2D point in projected coordinates (metres or degrees).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D { pub x: f64, pub y: f64 }

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self { Point2D { x, y } }
    pub fn dist(&self, other: &Point2D) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox { pub min_x: f64, pub min_y: f64, pub max_x: f64, pub max_y: f64 }

impl BBox {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        BBox { min_x, min_y, max_x, max_y }
    }

    pub fn contains_point(&self, p: &Point2D) -> bool {
        p.x >= self.min_x && p.x <= self.max_x && p.y >= self.min_y && p.y <= self.max_y
    }

    pub fn intersects(&self, other: &BBox) -> bool {
        !(other.min_x > self.max_x || other.max_x < self.min_x ||
          other.min_y > self.max_y || other.max_y < self.min_y)
    }

    pub fn area(&self) -> f64 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y)
    }

    pub fn center(&self) -> Point2D {
        Point2D::new((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)
    }

    pub fn expand(&self, other: &BBox) -> BBox {
        BBox::new(self.min_x.min(other.min_x), self.min_y.min(other.min_y),
                  self.max_x.max(other.max_x), self.max_y.max(other.max_y))
    }

    pub fn from_point(p: &Point2D) -> BBox {
        BBox::new(p.x, p.y, p.x, p.y)
    }
}

/// A sequence of connected line segments.
#[derive(Debug, Clone)]
pub struct LineString { pub points: Vec<Point2D> }

impl LineString {
    pub fn new(points: Vec<Point2D>) -> Self { LineString { points } }

    /// Total length of the line string.
    pub fn length(&self) -> f64 {
        self.points.windows(2).map(|w| w[0].dist(&w[1])).sum()
    }

    /// Bounding box of all points.
    pub fn bbox(&self) -> Option<BBox> {
        if self.points.is_empty() { return None; }
        let mut bb = BBox::from_point(&self.points[0]);
        for p in &self.points[1..] {
            bb = bb.expand(&BBox::from_point(p));
        }
        Some(bb)
    }
}

/// A closed polygon (exterior ring + optional interior holes).
#[derive(Debug, Clone)]
pub struct Polygon {
    pub exterior: Vec<Point2D>,
    pub holes:    Vec<Vec<Point2D>>,
}

impl Polygon {
    pub fn new(exterior: Vec<Point2D>) -> Self {
        Polygon { exterior, holes: Vec::new() }
    }

    /// Signed area via the shoelace formula (positive = CCW winding).
    pub fn signed_area(&self) -> f64 {
        shoelace_area(&self.exterior)
    }

    /// Area (unsigned) accounting for holes.
    pub fn area(&self) -> f64 {
        let outer = shoelace_area(&self.exterior).abs();
        let holes: f64 = self.holes.iter().map(|h| shoelace_area(h).abs()).sum();
        outer - holes
    }

    /// Centroid of the exterior ring.
    pub fn centroid(&self) -> Point2D {
        centroid(&self.exterior)
    }

    /// Test if point is inside the polygon (ray casting algorithm).
    pub fn contains(&self, p: &Point2D) -> bool {
        point_in_ring(p, &self.exterior) &&
        self.holes.iter().all(|h| !point_in_ring(p, h))
    }

    pub fn bbox(&self) -> Option<BBox> {
        if self.exterior.is_empty() { return None; }
        let mut bb = BBox::from_point(&self.exterior[0]);
        for pt in &self.exterior[1..] {
            bb = bb.expand(&BBox::from_point(pt));
        }
        Some(bb)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. COMPUTATIONAL GEOMETRY ALGORITHMS
// ─────────────────────────────────────────────────────────────────────────────

/// Shoelace (Gauss's area) formula for polygon area.
fn shoelace_area(ring: &[Point2D]) -> f64 {
    let n = ring.len();
    if n < 3 { return 0.0; }
    let mut sum = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        sum += ring[i].x * ring[j].y - ring[j].x * ring[i].y;
    }
    sum / 2.0
}

/// Centroid of a polygon ring.
fn centroid(ring: &[Point2D]) -> Point2D {
    let a = shoelace_area(ring);
    if a.abs() < 1e-12 {
        // Degenerate: return average of vertices
        let cx = ring.iter().map(|p| p.x).sum::<f64>() / ring.len() as f64;
        let cy = ring.iter().map(|p| p.y).sum::<f64>() / ring.len() as f64;
        return Point2D::new(cx, cy);
    }
    let n = ring.len();
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    for i in 0..n {
        let j = (i + 1) % n;
        let cross = ring[i].x * ring[j].y - ring[j].x * ring[i].y;
        cx += (ring[i].x + ring[j].x) * cross;
        cy += (ring[i].y + ring[j].y) * cross;
    }
    Point2D::new(cx / (6.0 * a), cy / (6.0 * a))
}

/// Point-in-polygon test via ray casting (Jordan curve theorem).
fn point_in_ring(p: &Point2D, ring: &[Point2D]) -> bool {
    let n = ring.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let xi = ring[i].x; let yi = ring[i].y;
        let xj = ring[j].x; let yj = ring[j].y;
        if ((yi > p.y) != (yj > p.y)) &&
           (p.x < (xj - xi) * (p.y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// 2D cross product of vectors (p1→p2) × (p1→p3).
fn cross2d(p1: &Point2D, p2: &Point2D, p3: &Point2D) -> f64 {
    (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
}

/// Graham scan convex hull algorithm. O(n log n).
pub fn convex_hull(mut points: Vec<Point2D>) -> Vec<Point2D> {
    let n = points.len();
    if n < 3 { return points; }

    // Find bottom-most (then left-most) point
    let mut min_idx = 0;
    for i in 1..n {
        if points[i].y < points[min_idx].y ||
           (points[i].y == points[min_idx].y && points[i].x < points[min_idx].x) {
            min_idx = i;
        }
    }
    points.swap(0, min_idx);
    let pivot = points[0];

    // Sort by polar angle from pivot
    points[1..].sort_by(|a, b| {
        let angle_a = (a.y - pivot.y).atan2(a.x - pivot.x);
        let angle_b = (b.y - pivot.y).atan2(b.x - pivot.x);
        angle_a.partial_cmp(&angle_b).unwrap_or(Ordering::Equal)
    });

    let mut hull: Vec<Point2D> = Vec::with_capacity(n);
    hull.push(points[0]);
    hull.push(points[1]);
    for i in 2..n {
        while hull.len() >= 2 {
            let m = hull.len();
            if cross2d(&hull[m - 2], &hull[m - 1], &points[i]) <= 0.0 {
                hull.pop();
            } else { break; }
        }
        hull.push(points[i]);
    }
    hull
}

/// Ramer-Douglas-Peucker line simplification. O(n log n) average.
/// Removes points that are within `epsilon` of the simplified line.
pub fn rdp_simplify(points: &[Point2D], epsilon: f64) -> Vec<Point2D> {
    if points.len() < 3 { return points.to_vec(); }
    let mut max_dist = 0.0;
    let mut max_idx = 0;
    let first = &points[0];
    let last  = &points[points.len() - 1];
    for i in 1..points.len() - 1 {
        let d = point_to_segment_dist(&points[i], first, last);
        if d > max_dist { max_dist = d; max_idx = i; }
    }
    if max_dist > epsilon {
        let left  = rdp_simplify(&points[..=max_idx], epsilon);
        let right = rdp_simplify(&points[max_idx..], epsilon);
        let mut result = left;
        result.extend_from_slice(&right[1..]);
        result
    } else {
        vec![*first, *last]
    }
}

/// Perpendicular distance from point p to line segment (a, b).
fn point_to_segment_dist(p: &Point2D, a: &Point2D, b: &Point2D) -> f64 {
    let ab_x = b.x - a.x; let ab_y = b.y - a.y;
    let len_sq = ab_x * ab_x + ab_y * ab_y;
    if len_sq < 1e-20 { return p.dist(a); }
    let t = ((p.x - a.x) * ab_x + (p.y - a.y) * ab_y) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj = Point2D::new(a.x + t * ab_x, a.y + t * ab_y);
    p.dist(&proj)
}

/// Segment-segment intersection test (excluding endpoints).
pub fn segments_intersect(a1: &Point2D, a2: &Point2D, b1: &Point2D, b2: &Point2D) -> bool {
    let d1 = cross2d(b1, b2, a1);
    let d2 = cross2d(b1, b2, a2);
    let d3 = cross2d(a1, a2, b1);
    let d4 = cross2d(a1, a2, b2);
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) &&
       ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0)) {
        return true;
    }
    false
}

/// Polygon-polygon intersection test (checks if any edge pairs intersect or one contains other).
pub fn polygons_intersect(a: &Polygon, b: &Polygon) -> bool {
    // Check bounding boxes first
    let bb_a = match a.bbox() { Some(b) => b, None => return false };
    let bb_b = match b.bbox() { Some(b) => b, None => return false };
    if !bb_a.intersects(&bb_b) { return false; }
    // Check if any point of A is in B or vice versa
    if !a.exterior.is_empty() && b.contains(&a.exterior[0]) { return true; }
    if !b.exterior.is_empty() && a.contains(&b.exterior[0]) { return true; }
    // Check edge intersections
    let na = a.exterior.len();
    let nb = b.exterior.len();
    for i in 0..na {
        let a1 = &a.exterior[i]; let a2 = &a.exterior[(i+1)%na];
        for j in 0..nb {
            let b1 = &b.exterior[j]; let b2 = &b.exterior[(j+1)%nb];
            if segments_intersect(a1, a2, b1, b2) { return true; }
        }
    }
    false
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. GEOHASH
// ─────────────────────────────────────────────────────────────────────────────

/// Geohash encoding (base32, variable precision).
/// A geohash encodes a lat/lon pair into a short string that can be
/// used for proximity searches (nearby geohashes share a common prefix).
const GEOHASH_CHARS: &[u8] = b"0123456789bcdefghjkmnpqrstuvwxyz";

pub fn geohash_encode(p: &LatLon, precision: usize) -> String {
    let mut lat_range = (-90.0f64, 90.0f64);
    let mut lon_range = (-180.0f64, 180.0f64);
    let mut is_lon = true;
    let mut bit = 0u8;
    let mut bits = 0u8;
    let mut result = String::with_capacity(precision);

    while result.len() < precision {
        if is_lon {
            let mid = (lon_range.0 + lon_range.1) / 2.0;
            if p.lon >= mid { bit = (bit << 1) | 1; lon_range.0 = mid; }
            else            { bit = bit << 1;        lon_range.1 = mid; }
        } else {
            let mid = (lat_range.0 + lat_range.1) / 2.0;
            if p.lat >= mid { bit = (bit << 1) | 1; lat_range.0 = mid; }
            else            { bit = bit << 1;        lat_range.1 = mid; }
        }
        is_lon = !is_lon;
        bits += 1;
        if bits == 5 {
            result.push(GEOHASH_CHARS[bit as usize] as char);
            bit = 0; bits = 0;
        }
    }
    result
}

pub fn geohash_decode(hash: &str) -> (LatLon, f64, f64) {
    let mut lat_range = (-90.0f64, 90.0f64);
    let mut lon_range = (-180.0f64, 180.0f64);
    let mut is_lon = true;

    for ch in hash.chars() {
        let idx = GEOHASH_CHARS.iter().position(|&b| b as char == ch).unwrap_or(0);
        for bit_shift in (0..5).rev() {
            let bit = (idx >> bit_shift) & 1;
            if is_lon {
                let mid = (lon_range.0 + lon_range.1) / 2.0;
                if bit == 1 { lon_range.0 = mid; } else { lon_range.1 = mid; }
            } else {
                let mid = (lat_range.0 + lat_range.1) / 2.0;
                if bit == 1 { lat_range.0 = mid; } else { lat_range.1 = mid; }
            }
            is_lon = !is_lon;
        }
    }
    let lat = (lat_range.0 + lat_range.1) / 2.0;
    let lon = (lon_range.0 + lon_range.1) / 2.0;
    let lat_err = (lat_range.1 - lat_range.0) / 2.0;
    let lon_err = (lon_range.1 - lon_range.0) / 2.0;
    (LatLon::new(lat, lon), lat_err, lon_err)
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. SPATIAL INDEX — R-TREE (Guttman, 1984)
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal R-tree for 2D bounding box spatial indexing.
/// Uses linear split heuristic. Supports insert and range query.
const RTREE_MAX_ENTRIES: usize = 4;
const RTREE_MIN_ENTRIES: usize = 2;

#[derive(Debug, Clone)]
enum RTreeNode {
    Leaf { bbox: BBox, id: u64 },
    Internal { bbox: BBox, children: Vec<RTreeNode> },
}

impl RTreeNode {
    fn bbox(&self) -> &BBox {
        match self { RTreeNode::Leaf { bbox, .. } => bbox, RTreeNode::Internal { bbox, .. } => bbox }
    }
    fn is_leaf(&self) -> bool { matches!(self, RTreeNode::Leaf { .. }) }
}

pub struct RTree { root: Option<RTreeNode> }

impl RTree {
    pub fn new() -> Self { RTree { root: None } }

    pub fn insert(&mut self, id: u64, bbox: BBox) {
        let leaf = RTreeNode::Leaf { bbox, id };
        match &mut self.root {
            None => { self.root = Some(leaf); }
            Some(root) => {
                // Simple insertion: collect all leaves and rebuild (brute-force for correctness)
                let mut entries = Vec::new();
                collect_leaves(root, &mut entries);
                entries.push((id, bbox));
                self.root = Some(build_rtree(&entries));
            }
        }
    }

    /// Range query: return IDs of entries whose bounding boxes intersect `query`.
    pub fn query(&self, query: &BBox) -> Vec<u64> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            rtree_query(root, query, &mut results);
        }
        results
    }

    /// Nearest-neighbour search (brute-force over leaves, but BBox-pruned at internal nodes).
    pub fn nearest(&self, point: &Point2D) -> Option<(u64, f64)> {
        let mut best: Option<(u64, f64)> = None;
        if let Some(root) = &self.root {
            rtree_nearest(root, point, &mut best);
        }
        best
    }
}

fn collect_leaves(node: &RTreeNode, out: &mut Vec<(u64, BBox)>) {
    match node {
        RTreeNode::Leaf { id, bbox } => { out.push((*id, *bbox)); }
        RTreeNode::Internal { children, .. } => {
            for c in children { collect_leaves(c, out); }
        }
    }
}

fn build_rtree(entries: &[(u64, BBox)]) -> RTreeNode {
    if entries.len() == 1 {
        return RTreeNode::Leaf { id: entries[0].0, bbox: entries[0].1 };
    }
    if entries.len() <= RTREE_MAX_ENTRIES {
        let mut bbox = entries[0].1;
        for e in &entries[1..] { bbox = bbox.expand(&e.1); }
        let children: Vec<RTreeNode> = entries.iter()
            .map(|&(id, bb)| RTreeNode::Leaf { id, bbox: bb }).collect();
        return RTreeNode::Internal { bbox, children };
    }
    // Split by longest axis
    let mut sorted = entries.to_vec();
    let span_x: f64 = entries.iter().map(|e| e.1.max_x).fold(f64::NEG_INFINITY, f64::max)
                    - entries.iter().map(|e| e.1.min_x).fold(f64::INFINITY, f64::min);
    let span_y: f64 = entries.iter().map(|e| e.1.max_y).fold(f64::NEG_INFINITY, f64::max)
                    - entries.iter().map(|e| e.1.min_y).fold(f64::INFINITY, f64::min);
    if span_x >= span_y {
        sorted.sort_by(|a, b| a.1.center().x.partial_cmp(&b.1.center().x).unwrap());
    } else {
        sorted.sort_by(|a, b| a.1.center().y.partial_cmp(&b.1.center().y).unwrap());
    }
    let mid = sorted.len() / 2;
    let left  = build_rtree(&sorted[..mid]);
    let right = build_rtree(&sorted[mid..]);
    let bbox = left.bbox().expand(right.bbox());
    RTreeNode::Internal { bbox, children: vec![left, right] }
}

fn rtree_query(node: &RTreeNode, q: &BBox, out: &mut Vec<u64>) {
    if !node.bbox().intersects(q) { return; }
    match node {
        RTreeNode::Leaf { id, bbox } => { if bbox.intersects(q) { out.push(*id); } }
        RTreeNode::Internal { children, .. } => {
            for c in children { rtree_query(c, q, out); }
        }
    }
}

fn bbox_min_dist(b: &BBox, p: &Point2D) -> f64 {
    let dx = if p.x < b.min_x { b.min_x - p.x } else if p.x > b.max_x { p.x - b.max_x } else { 0.0 };
    let dy = if p.y < b.min_y { b.min_y - p.y } else if p.y > b.max_y { p.y - b.max_y } else { 0.0 };
    (dx * dx + dy * dy).sqrt()
}

fn rtree_nearest(node: &RTreeNode, p: &Point2D, best: &mut Option<(u64, f64)>) {
    let min_possible = bbox_min_dist(node.bbox(), p);
    if let Some((_, best_dist)) = best { if min_possible >= *best_dist { return; } }
    match node {
        RTreeNode::Leaf { id, bbox } => {
            let d = bbox.center().dist(p);
            if best.map_or(true, |(_, bd)| d < bd) { *best = Some((*id, d)); }
        }
        RTreeNode::Internal { children, .. } => {
            for c in children { rtree_nearest(c, p, best); }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. ROUTING — A* WITH HAVERSINE HEURISTIC
// ─────────────────────────────────────────────────────────────────────────────

/// Road network graph node.
#[derive(Debug, Clone)]
pub struct GeoNode {
    pub id:  u64,
    pub pos: LatLon,
}

/// Directed edge in the road graph.
#[derive(Debug, Clone)]
pub struct GeoEdge {
    pub from:   u64,
    pub to:     u64,
    pub weight: f64,  // metres (or travel time)
}

/// Road network for routing.
pub struct GeoGraph {
    pub nodes: HashMap<u64, GeoNode>,
    pub adj:   HashMap<u64, Vec<(u64, f64)>>, // adjacency list
}

impl GeoGraph {
    pub fn new() -> Self {
        GeoGraph { nodes: HashMap::new(), adj: HashMap::new() }
    }

    pub fn add_node(&mut self, node: GeoNode) {
        self.adj.entry(node.id).or_default();
        self.nodes.insert(node.id, node);
    }

    /// Add directed edge.
    pub fn add_edge(&mut self, from: u64, to: u64, weight: f64) {
        self.adj.entry(from).or_default().push((to, weight));
    }

    /// Add undirected edge.
    pub fn add_undirected_edge(&mut self, a: u64, b: u64, weight: f64) {
        self.add_edge(a, b, weight);
        self.add_edge(b, a, weight);
    }

    /// A* shortest path. Uses haversine distance as heuristic.
    /// Returns (total_distance, path_of_node_ids) or None if unreachable.
    pub fn astar(&self, start: u64, goal: u64) -> Option<(f64, Vec<u64>)> {
        #[derive(PartialEq)]
        struct State { f: f64, g: f64, id: u64 }
        impl Eq for State {}
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
        }
        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                other.f.partial_cmp(&self.f).unwrap_or(Ordering::Equal)
            }
        }

        let goal_pos = &self.nodes.get(&goal)?.pos;
        let mut dist: HashMap<u64, f64> = HashMap::new();
        let mut prev: HashMap<u64, u64> = HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(start, 0.0);
        let h0 = haversine_m(&self.nodes.get(&start)?.pos, goal_pos);
        heap.push(State { f: h0, g: 0.0, id: start });

        while let Some(State { g, id, .. }) = heap.pop() {
            if id == goal {
                let mut path = vec![goal];
                let mut cur = goal;
                while let Some(&p) = prev.get(&cur) { path.push(p); cur = p; }
                path.reverse();
                return Some((g, path));
            }
            if g > *dist.get(&id).unwrap_or(&f64::INFINITY) { continue; }
            if let Some(edges) = self.adj.get(&id) {
                for &(nb, w) in edges {
                    let g2 = g + w;
                    if g2 < *dist.get(&nb).unwrap_or(&f64::INFINITY) {
                        dist.insert(nb, g2);
                        prev.insert(nb, id);
                        let h = haversine_m(&self.nodes.get(&nb)?.pos, goal_pos);
                        heap.push(State { f: g2 + h, g: g2, id: nb });
                    }
                }
            }
        }
        None
    }

    /// Dijkstra's single-source shortest paths to all reachable nodes.
    pub fn dijkstra(&self, start: u64) -> HashMap<u64, f64> {
        #[derive(PartialEq)]
        struct State { dist: f64, id: u64 }
        impl Eq for State {}
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
        }
        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
            }
        }

        let mut dist: HashMap<u64, f64> = HashMap::new();
        dist.insert(start, 0.0);
        let mut heap = BinaryHeap::new();
        heap.push(State { dist: 0.0, id: start });

        while let Some(State { dist: d, id }) = heap.pop() {
            if d > *dist.get(&id).unwrap_or(&f64::INFINITY) { continue; }
            if let Some(edges) = self.adj.get(&id) {
                for &(nb, w) in edges {
                    let d2 = d + w;
                    if d2 < *dist.get(&nb).unwrap_or(&f64::INFINITY) {
                        dist.insert(nb, d2);
                        heap.push(State { dist: d2, id: nb });
                    }
                }
            }
        }
        dist
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. RASTER / DEM OPERATIONS
// ─────────────────────────────────────────────────────────────────────────────

/// Digital Elevation Model stored as a regular grid.
pub struct Dem {
    pub width:     usize,
    pub height:    usize,
    pub origin_x:  f64,   // X coordinate of top-left corner
    pub origin_y:  f64,   // Y coordinate of top-left corner
    pub cell_size: f64,   // cell size in map units
    pub data:      Vec<f32>,  // elevation values, row-major
}

impl Dem {
    pub fn new(width: usize, height: usize, origin_x: f64, origin_y: f64,
               cell_size: f64, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), width * height);
        Dem { width, height, origin_x, origin_y, cell_size, data }
    }

    fn idx(&self, col: usize, row: usize) -> usize { row * self.width + col }

    pub fn elevation_at(&self, col: usize, row: usize) -> f32 {
        self.data[self.idx(col, row)]
    }

    /// Bilinear interpolation of elevation at a continuous (x, y) position.
    pub fn interpolate(&self, x: f64, y: f64) -> Option<f64> {
        let col_f = (x - self.origin_x) / self.cell_size;
        let row_f = (self.origin_y - y) / self.cell_size; // Y axis inverted (raster convention)
        if col_f < 0.0 || row_f < 0.0 { return None; }
        let col = col_f as usize; let row = row_f as usize;
        if col + 1 >= self.width || row + 1 >= self.height { return None; }
        let tx = col_f - col as f64; let ty = row_f - row as f64;
        let z00 = self.elevation_at(col,   row)   as f64;
        let z10 = self.elevation_at(col+1, row)   as f64;
        let z01 = self.elevation_at(col,   row+1) as f64;
        let z11 = self.elevation_at(col+1, row+1) as f64;
        let z = (1.0-tx)*(1.0-ty)*z00 + tx*(1.0-ty)*z10
              + (1.0-tx)*ty*z01 + tx*ty*z11;
        Some(z)
    }

    /// Compute slope (gradient magnitude) in degrees using central differences.
    /// Horn's method: accounts for 8-neighbourhood.
    pub fn slope(&self, col: usize, row: usize) -> Option<f64> {
        if col == 0 || row == 0 || col + 1 >= self.width || row + 1 >= self.height {
            return None;
        }
        let z = |c: usize, r: usize| self.elevation_at(c, r) as f64;
        let dz_dx = (z(col+1,row-1) + 2.0*z(col+1,row) + z(col+1,row+1)
                   - z(col-1,row-1) - 2.0*z(col-1,row) - z(col-1,row+1))
                  / (8.0 * self.cell_size);
        let dz_dy = (z(col-1,row+1) + 2.0*z(col,row+1) + z(col+1,row+1)
                   - z(col-1,row-1) - 2.0*z(col,row-1) - z(col+1,row-1))
                  / (8.0 * self.cell_size);
        Some((dz_dx * dz_dx + dz_dy * dz_dy).sqrt().atan().to_degrees())
    }

    /// Aspect (direction of steepest descent) in degrees clockwise from North.
    pub fn aspect(&self, col: usize, row: usize) -> Option<f64> {
        if col == 0 || row == 0 || col + 1 >= self.width || row + 1 >= self.height {
            return None;
        }
        let z = |c: usize, r: usize| self.elevation_at(c, r) as f64;
        let dz_dx = (z(col+1,row-1) + 2.0*z(col+1,row) + z(col+1,row+1)
                   - z(col-1,row-1) - 2.0*z(col-1,row) - z(col-1,row+1))
                  / (8.0 * self.cell_size);
        let dz_dy = (z(col-1,row+1) + 2.0*z(col,row+1) + z(col+1,row+1)
                   - z(col-1,row-1) - 2.0*z(col,row-1) - z(col+1,row-1))
                  / (8.0 * self.cell_size);
        let aspect_rad = dz_dy.atan2(-dz_dx);
        let mut aspect = 180.0 - aspect_rad.to_degrees() + 90.0 * (dz_dx / dz_dx.abs().max(1e-15));
        if aspect < 0.0   { aspect += 360.0; }
        if aspect > 360.0 { aspect -= 360.0; }
        Some(aspect)
    }

    /// Hillshade: simulated illumination (Lambertian model).
    /// `azimuth`: sun azimuth in degrees clockwise from North.
    /// `altitude`: sun altitude above horizon in degrees.
    pub fn hillshade(&self, col: usize, row: usize, azimuth_deg: f64, altitude_deg: f64) -> Option<u8> {
        let zenith = (90.0 - altitude_deg).to_radians();
        let azimuth_math = (360.0 - azimuth_deg + 90.0) % 360.0;
        let az_rad = azimuth_math.to_radians();

        if col == 0 || row == 0 || col + 1 >= self.width || row + 1 >= self.height {
            return None;
        }
        let z = |c: usize, r: usize| self.elevation_at(c, r) as f64;
        let dz_dx = (z(col+1,row-1) + 2.0*z(col+1,row) + z(col+1,row+1)
                   - z(col-1,row-1) - 2.0*z(col-1,row) - z(col-1,row+1))
                  / (8.0 * self.cell_size);
        let dz_dy = (z(col-1,row+1) + 2.0*z(col,row+1) + z(col+1,row+1)
                   - z(col-1,row-1) - 2.0*z(col,row-1) - z(col+1,row-1))
                  / (8.0 * self.cell_size);
        let slope_rad = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt().atan();
        let aspect_rad = dz_dy.atan2(-dz_dx);
        let hs = 255.0 * ((zenith.cos() * slope_rad.cos())
            + (zenith.sin() * slope_rad.sin() * (az_rad - aspect_rad).cos()));
        Some(hs.clamp(0.0, 255.0) as u8)
    }

    /// Viewshed analysis: which cells are visible from (col, row)?
    /// Uses ray-marching with maximum angle technique.
    pub fn viewshed(&self, obs_col: usize, obs_row: usize, obs_height: f64) -> Vec<Vec<bool>> {
        let obs_z = self.elevation_at(obs_col, obs_row) as f64 + obs_height;
        let mut visible = vec![vec![false; self.width]; self.height];
        visible[obs_row][obs_col] = true;

        for row in 0..self.height {
            for col in 0..self.width {
                if row == obs_row && col == obs_col { continue; }
                // March along the line from observer to target
                let dx = col as f64 - obs_col as f64;
                let dy = row as f64 - obs_row as f64;
                let steps = dx.abs().max(dy.abs()) as usize;
                let mut max_angle = f64::NEG_INFINITY;
                let mut blocked = false;
                for s in 1..steps {
                    let fc = obs_col as f64 + dx * s as f64 / steps as f64;
                    let fr = obs_row as f64 + dy * s as f64 / steps as f64;
                    let ic = fc.round() as usize;
                    let ir = fr.round() as usize;
                    if ic >= self.width || ir >= self.height { break; }
                    let z = self.elevation_at(ic, ir) as f64;
                    let dist = (dx * s as f64 / steps as f64).hypot(dy * s as f64 / steps as f64);
                    let angle = (z - obs_z) / dist;
                    if angle > max_angle { max_angle = angle; }
                }
                let target_z = self.elevation_at(col, row) as f64;
                let dist = dx.hypot(dy);
                let target_angle = (target_z - obs_z) / dist;
                visible[row][col] = target_angle >= max_angle - 1e-9;
            }
        }
        visible
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. QUADTREE SPATIAL INDEX
// ─────────────────────────────────────────────────────────────────────────────

const QUADTREE_CAPACITY: usize = 4;

pub struct QuadTree {
    boundary: BBox,
    points:   Vec<(Point2D, u64)>,
    divided:  bool,
    nw: Option<Box<QuadTree>>,
    ne: Option<Box<QuadTree>>,
    sw: Option<Box<QuadTree>>,
    se: Option<Box<QuadTree>>,
}

impl QuadTree {
    pub fn new(boundary: BBox) -> Self {
        QuadTree { boundary, points: Vec::new(), divided: false,
                   nw: None, ne: None, sw: None, se: None }
    }

    pub fn insert(&mut self, p: Point2D, id: u64) -> bool {
        if !self.boundary.contains_point(&p) { return false; }
        if !self.divided && self.points.len() < QUADTREE_CAPACITY {
            self.points.push((p, id));
            return true;
        }
        if !self.divided { self.subdivide(); }
        if let Some(ref mut nw) = self.nw { if nw.insert(p, id) { return true; } }
        if let Some(ref mut ne) = self.ne { if ne.insert(p, id) { return true; } }
        if let Some(ref mut sw) = self.sw { if sw.insert(p, id) { return true; } }
        if let Some(ref mut se) = self.se { if se.insert(p, id) { return true; } }
        false
    }

    fn subdivide(&mut self) {
        let b = &self.boundary;
        let cx = (b.min_x + b.max_x) / 2.0;
        let cy = (b.min_y + b.max_y) / 2.0;
        self.nw = Some(Box::new(QuadTree::new(BBox::new(b.min_x, cy, cx, b.max_y))));
        self.ne = Some(Box::new(QuadTree::new(BBox::new(cx, cy, b.max_x, b.max_y))));
        self.sw = Some(Box::new(QuadTree::new(BBox::new(b.min_x, b.min_y, cx, cy))));
        self.se = Some(Box::new(QuadTree::new(BBox::new(cx, b.min_y, b.max_x, cy))));
        // Re-insert existing points into children
        let pts = std::mem::take(&mut self.points);
        self.divided = true;
        for (p, id) in pts { self.insert(p, id); }
    }

    pub fn query_range(&self, range: &BBox) -> Vec<(Point2D, u64)> {
        let mut result = Vec::new();
        if !self.boundary.intersects(range) { return result; }
        for &(p, id) in &self.points {
            if range.contains_point(&p) { result.push((p, id)); }
        }
        if self.divided {
            if let Some(ref nw) = self.nw { result.extend(nw.query_range(range)); }
            if let Some(ref ne) = self.ne { result.extend(ne.query_range(range)); }
            if let Some(ref sw) = self.sw { result.extend(sw.query_range(range)); }
            if let Some(ref se) = self.se { result.extend(se.query_range(range)); }
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn london()     -> LatLon { LatLon::new(51.5074, -0.1278) }
    fn paris()      -> LatLon { LatLon::new(48.8566,  2.3522) }
    fn new_york()   -> LatLon { LatLon::new(40.7128, -74.0060) }
    fn null_island()-> LatLon { LatLon::new(0.0, 0.0) }

    // ── ECEF Roundtrip ───────────────────────────────────────────────────────

    #[test]
    fn test_ecef_roundtrip_london() {
        let ll = london();
        let ecef = geodetic_to_ecef(&ll);
        let ll2  = ecef_to_geodetic(&ecef);
        assert!((ll2.lat - ll.lat).abs() < 1e-8, "lat: {}", ll2.lat);
        assert!((ll2.lon - ll.lon).abs() < 1e-8, "lon: {}", ll2.lon);
        assert!((ll2.alt - ll.alt).abs() < 1e-3, "alt: {}", ll2.alt);
    }

    #[test]
    fn test_ecef_roundtrip_null_island() {
        let ll = null_island();
        let ecef = geodetic_to_ecef(&ll);
        // At null island: X=a, Y=0, Z=0
        assert!((ecef.x - Wgs84::A).abs() < 1.0, "X at null island: {}", ecef.x);
        assert!(ecef.y.abs() < 1.0, "Y at null island: {}", ecef.y);
        assert!(ecef.z.abs() < 1.0, "Z at null island: {}", ecef.z);
        let ll2 = ecef_to_geodetic(&ecef);
        assert!((ll2.lat).abs() < 1e-8);
        assert!((ll2.lon).abs() < 1e-8);
    }

    #[test]
    fn test_ecef_new_york() {
        let ll = new_york();
        let ecef = geodetic_to_ecef(&ll);
        let ll2  = ecef_to_geodetic(&ecef);
        assert!((ll2.lat - ll.lat).abs() < 1e-8);
        assert!((ll2.lon - ll.lon).abs() < 1e-8);
    }

    #[test]
    fn test_enu_at_origin() {
        let origin = london();
        let pt_ecef = geodetic_to_ecef(&origin);
        let enu = ecef_to_enu(&pt_ecef, &origin);
        // Point AT the reference → ENU should be (0, 0, 0)
        assert!(enu.e.abs() < 1e-3, "ENU East: {}", enu.e);
        assert!(enu.n.abs() < 1e-3, "ENU North: {}", enu.n);
        assert!(enu.u.abs() < 1e-3, "ENU Up: {}", enu.u);
    }

    // ── Distance Calculations ─────────────────────────────────────────────────

    #[test]
    fn test_haversine_london_paris() {
        // London–Paris is approximately 340 km
        let d = haversine_m(&london(), &paris());
        assert!((d - 340_000.0).abs() < 5_000.0, "Haversine L-P: {}m", d);
    }

    #[test]
    fn test_haversine_zero_distance() {
        let d = haversine_m(&london(), &london());
        assert!(d < 1e-6, "Self distance: {}", d);
    }

    #[test]
    fn test_vincenty_london_paris() {
        let (dist, b1, b2) = vincenty_inverse(&london(), &paris());
        // London–Paris geodesic distance is ~343.9 km
        assert!((dist - 343_900.0).abs() < 2_000.0, "Vincenty L-P: {}m", dist);
        assert!(b1 > 0.0 && b1 < 180.0, "Bearing: {}", b1);
    }

    #[test]
    fn test_vincenty_vs_haversine_consistency() {
        // Vincenty should be more accurate but agree with haversine to within 1%
        let (vd, _, _) = vincenty_inverse(&london(), &paris());
        let hd = haversine_m(&london(), &paris());
        assert!((vd - hd).abs() / vd < 0.01, "Vincenty/Haversine diff > 1%");
    }

    #[test]
    fn test_vincenty_direct_inverse_roundtrip() {
        let start = london();
        let bearing = 135.0; // SE
        let dist = 500_000.0; // 500 km
        let end = vincenty_direct(&start, bearing, dist);
        let (recovered_dist, _, _) = vincenty_inverse(&start, &end);
        assert!((recovered_dist - dist).abs() < 1.0,
                "Vincenty direct/inverse: {} vs {}", recovered_dist, dist);
    }

    // ── UTM ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_utm_zone_london() {
        let zone = utm_zone(london().lon);
        assert_eq!(zone, 30, "London UTM zone: {}", zone);
    }

    #[test]
    fn test_utm_zone_new_york() {
        let zone = utm_zone(new_york().lon);
        assert_eq!(zone, 18, "NYC UTM zone: {}", zone);
    }

    #[test]
    fn test_utm_london_easting_reasonable() {
        let utm = latlon_to_utm(&london());
        // London (lon≈−0.13°) in UTM zone 30 has easting ~699,316 m
        assert!(utm.easting > 400_000.0 && utm.easting < 800_000.0,
                "Easting: {}", utm.easting);
        assert!(utm.northing > 5_000_000.0, "Northing: {}", utm.northing);
        assert!(utm.north, "London should be northern hemisphere");
    }

    // ── Web Mercator ──────────────────────────────────────────────────────────

    #[test]
    fn test_web_mercator_roundtrip() {
        let ll = london();
        let (x, y) = latlon_to_web_mercator(&ll);
        let ll2 = web_mercator_to_latlon(x, y);
        assert!((ll2.lat - ll.lat).abs() < 1e-8);
        assert!((ll2.lon - ll.lon).abs() < 1e-8);
    }

    #[test]
    fn test_web_mercator_null_island() {
        let (x, y) = latlon_to_web_mercator(&null_island());
        assert!(x.abs() < 1e-6, "x: {}", x);
        assert!(y.abs() < 1e-6, "y: {}", y);
    }

    #[test]
    fn test_tile_math() {
        // Zoom 0: the whole world is tile (0,0)
        let london_tile0 = latlon_to_tile(&london(), 0);
        assert_eq!(london_tile0, (0, 0));
        // Zoom 1: London is in tile (0, 0) or (1, 0) depending on position
        let london_tile1 = latlon_to_tile(&london(), 1);
        assert!(london_tile1.0 <= 1 && london_tile1.1 <= 1);
    }

    // ── Geometry & Spatial Predicates ─────────────────────────────────────────

    #[test]
    fn test_polygon_area_unit_square() {
        let pts = vec![
            Point2D::new(0.0, 0.0), Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0), Point2D::new(0.0, 1.0),
        ];
        let poly = Polygon::new(pts);
        assert!((poly.area() - 1.0).abs() < 1e-10, "Area: {}", poly.area());
    }

    #[test]
    fn test_polygon_area_triangle() {
        // Right triangle base=3, height=4 → area=6
        let pts = vec![
            Point2D::new(0.0, 0.0), Point2D::new(3.0, 0.0), Point2D::new(0.0, 4.0),
        ];
        let poly = Polygon::new(pts);
        assert!((poly.area() - 6.0).abs() < 1e-10, "Triangle area: {}", poly.area());
    }

    #[test]
    fn test_polygon_centroid_unit_square() {
        let pts = vec![
            Point2D::new(0.0, 0.0), Point2D::new(2.0, 0.0),
            Point2D::new(2.0, 2.0), Point2D::new(0.0, 2.0),
        ];
        let poly = Polygon::new(pts);
        let c = poly.centroid();
        assert!((c.x - 1.0).abs() < 1e-10 && (c.y - 1.0).abs() < 1e-10,
                "Centroid: ({}, {})", c.x, c.y);
    }

    #[test]
    fn test_point_in_polygon_inside() {
        let pts = vec![
            Point2D::new(0.0, 0.0), Point2D::new(4.0, 0.0),
            Point2D::new(4.0, 4.0), Point2D::new(0.0, 4.0),
        ];
        let poly = Polygon::new(pts);
        assert!(poly.contains(&Point2D::new(2.0, 2.0)));
        assert!(!poly.contains(&Point2D::new(5.0, 2.0)));
        assert!(!poly.contains(&Point2D::new(-1.0, 2.0)));
    }

    #[test]
    fn test_bbox_intersects() {
        let a = BBox::new(0.0, 0.0, 2.0, 2.0);
        let b = BBox::new(1.0, 1.0, 3.0, 3.0);
        let c = BBox::new(3.0, 3.0, 5.0, 5.0);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_polygons_intersect() {
        let p1 = Polygon::new(vec![
            Point2D::new(0.0,0.0), Point2D::new(2.0,0.0),
            Point2D::new(2.0,2.0), Point2D::new(0.0,2.0)]);
        let p2 = Polygon::new(vec![
            Point2D::new(1.0,1.0), Point2D::new(3.0,1.0),
            Point2D::new(3.0,3.0), Point2D::new(1.0,3.0)]);
        let p3 = Polygon::new(vec![
            Point2D::new(5.0,5.0), Point2D::new(7.0,5.0),
            Point2D::new(7.0,7.0), Point2D::new(5.0,7.0)]);
        assert!(polygons_intersect(&p1, &p2));
        assert!(!polygons_intersect(&p1, &p3));
    }

    // ── Convex Hull ───────────────────────────────────────────────────────────

    #[test]
    fn test_convex_hull_square() {
        let pts = vec![
            Point2D::new(0.0,0.0), Point2D::new(1.0,0.0),
            Point2D::new(1.0,1.0), Point2D::new(0.0,1.0),
            Point2D::new(0.5,0.5), // interior point — should be excluded
        ];
        let hull = convex_hull(pts);
        assert_eq!(hull.len(), 4, "Square hull has 4 vertices");
    }

    #[test]
    fn test_convex_hull_collinear() {
        // Collinear points: hull should be just two endpoints
        let pts = vec![
            Point2D::new(0.0,0.0), Point2D::new(1.0,0.0), Point2D::new(2.0,0.0)
        ];
        let hull = convex_hull(pts);
        // Graham scan may return 2 or 3 for collinear depending on implementation
        assert!(hull.len() <= 3);
    }

    // ── RDP Simplification ────────────────────────────────────────────────────

    #[test]
    fn test_rdp_keeps_endpoints() {
        let pts = vec![
            Point2D::new(0.0,0.0), Point2D::new(1.0,0.1),
            Point2D::new(2.0,0.0), Point2D::new(3.0,0.2),
            Point2D::new(4.0,0.0),
        ];
        let simplified = rdp_simplify(&pts, 0.5);
        assert_eq!(simplified[0], pts[0]);
        assert_eq!(*simplified.last().unwrap(), *pts.last().unwrap());
    }

    #[test]
    fn test_rdp_removes_interior_collinear() {
        // Perfectly collinear → simplify to just two endpoints
        let pts: Vec<Point2D> = (0..10).map(|i| Point2D::new(i as f64, 0.0)).collect();
        let simplified = rdp_simplify(&pts, 0.001);
        assert_eq!(simplified.len(), 2, "Collinear simplifies to 2 points");
    }

    // ── Geohash ───────────────────────────────────────────────────────────────

    #[test]
    fn test_geohash_encode_decode_roundtrip() {
        let p = london();
        let h = geohash_encode(&p, 9);
        assert_eq!(h.len(), 9);
        let (decoded, lat_err, lon_err) = geohash_decode(&h);
        assert!((decoded.lat - p.lat).abs() <= lat_err + 1e-10,
                "Lat: {}, decoded: {}", p.lat, decoded.lat);
        assert!((decoded.lon - p.lon).abs() <= lon_err + 1e-10,
                "Lon: {}, decoded: {}", p.lon, decoded.lon);
    }

    #[test]
    fn test_geohash_prefix_sharing() {
        // Very close points should share a long common prefix
        let p1 = LatLon::new(51.5074, -0.1278);
        let p2 = LatLon::new(51.5075, -0.1277); // ~10m away
        let h1 = geohash_encode(&p1, 7);
        let h2 = geohash_encode(&p2, 7);
        // At precision 7, cells are ~150m × 150m → same or adjacent cell
        let common: usize = h1.chars().zip(h2.chars()).take_while(|(a, b)| a == b).count();
        assert!(common >= 5, "Common prefix too short: {} (h1={}, h2={})", common, h1, h2);
    }

    #[test]
    fn test_geohash_accuracy_increases_with_precision() {
        let p = paris();
        let (_, lat_err_5, _) = geohash_decode(&geohash_encode(&p, 5));
        let (_, lat_err_8, _) = geohash_decode(&geohash_encode(&p, 8));
        assert!(lat_err_8 < lat_err_5, "Higher precision → smaller error");
    }

    // ── R-Tree ────────────────────────────────────────────────────────────────

    #[test]
    fn test_rtree_insert_and_query() {
        let mut rt = RTree::new();
        for i in 0..20u64 {
            let x = i as f64;
            rt.insert(i, BBox::new(x, x, x+1.0, x+1.0));
        }
        let results = rt.query(&BBox::new(5.0, 5.0, 7.0, 7.0));
        assert!(results.contains(&5), "Should find id=5");
        assert!(results.contains(&6), "Should find id=6");
        assert!(!results.contains(&0), "Should not find id=0");
    }

    #[test]
    fn test_rtree_nearest() {
        let mut rt = RTree::new();
        rt.insert(1, BBox::new(0.0, 0.0, 1.0, 1.0));
        rt.insert(2, BBox::new(10.0, 10.0, 11.0, 11.0));
        rt.insert(3, BBox::new(5.0, 5.0, 6.0, 6.0));
        let (id, _) = rt.nearest(&Point2D::new(0.5, 0.5)).unwrap();
        assert_eq!(id, 1, "Nearest to (0.5,0.5) should be id=1");
    }

    // ── QuadTree ──────────────────────────────────────────────────────────────

    #[test]
    fn test_quadtree_insert_query() {
        let bounds = BBox::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = QuadTree::new(bounds);
        for i in 0u64..50 {
            let x = (i * 2) as f64;
            let y = (i * 2) as f64 % 100.0;
            qt.insert(Point2D::new(x, y), i);
        }
        let results = qt.query_range(&BBox::new(0.0, 0.0, 10.0, 10.0));
        assert!(!results.is_empty(), "QuadTree range query returned no results");
        assert!(results.iter().all(|(p, _)| p.x >= 0.0 && p.x <= 10.0));
    }

    // ── Routing ───────────────────────────────────────────────────────────────

    #[test]
    fn test_dijkstra_simple_graph() {
        let mut g = GeoGraph::new();
        for (id, lat, lon) in [(1u64,0.0,0.0),(2,0.0,0.1),(3,0.0,0.2),(4,0.1,0.1)] {
            g.add_node(GeoNode { id, pos: LatLon::new(lat, lon) });
        }
        g.add_undirected_edge(1, 2, 1000.0);
        g.add_undirected_edge(2, 3, 1000.0);
        g.add_undirected_edge(1, 4, 3000.0);
        g.add_undirected_edge(4, 3, 3000.0);
        let dists = g.dijkstra(1);
        assert!((dists[&3] - 2000.0).abs() < 1e-6, "Dijkstra 1→3: {}", dists[&3]);
    }

    #[test]
    fn test_astar_finds_path() {
        let mut g = GeoGraph::new();
        let nodes = vec![
            (1u64, LatLon::new(51.5, 0.0)),
            (2,    LatLon::new(51.5, 0.1)),
            (3,    LatLon::new(51.6, 0.1)),
        ];
        for (id, pos) in nodes { g.add_node(GeoNode { id, pos }); }
        let d12 = haversine_m(&g.nodes[&1].pos, &g.nodes[&2].pos);
        let d23 = haversine_m(&g.nodes[&2].pos, &g.nodes[&3].pos);
        g.add_undirected_edge(1, 2, d12);
        g.add_undirected_edge(2, 3, d23);
        let (dist, path) = g.astar(1, 3).unwrap();
        assert_eq!(path, vec![1, 2, 3]);
        assert!((dist - (d12 + d23)).abs() < 1e-3);
    }

    #[test]
    fn test_astar_no_path() {
        let mut g = GeoGraph::new();
        g.add_node(GeoNode { id: 1, pos: LatLon::new(0.0, 0.0) });
        g.add_node(GeoNode { id: 2, pos: LatLon::new(1.0, 1.0) });
        // No edges → no path
        assert!(g.astar(1, 2).is_none());
    }

    // ── DEM ───────────────────────────────────────────────────────────────────

    fn make_dem() -> Dem {
        // 5x5 DEM with a hill in the centre
        let data: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 2.0, 1.0, 0.0,
            0.0, 2.0, 4.0, 2.0, 0.0,
            0.0, 1.0, 2.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        Dem::new(5, 5, 0.0, 5.0, 1.0, data)
    }

    #[test]
    fn test_dem_elevation_at() {
        let dem = make_dem();
        assert!((dem.elevation_at(2, 2) - 4.0).abs() < 1e-6, "Peak elevation");
        assert!((dem.elevation_at(0, 0) - 0.0).abs() < 1e-6, "Corner elevation");
    }

    #[test]
    fn test_dem_interpolate() {
        let dem = make_dem();
        let z = dem.interpolate(2.5, 2.5).unwrap(); // centre of peak cell
        assert!(z > 2.0 && z <= 4.0, "Interpolated elevation: {}", z);
    }

    #[test]
    fn test_dem_slope_positive() {
        let dem = make_dem();
        // Slope on the hillside (not at the symmetric peak) should be > 0
        let slope = dem.slope(2, 1).unwrap();
        assert!(slope > 0.0, "Slope at hill: {}", slope);
    }

    #[test]
    fn test_dem_hillshade_range() {
        let dem = make_dem();
        let hs = dem.hillshade(2, 2, 315.0, 45.0).unwrap();
        // Hillshade is in [0, 255]
        assert!(hs <= 255);
    }

    #[test]
    fn test_dem_viewshed_peak_visible() {
        let dem = make_dem();
        let vis = dem.viewshed(2, 2, 0.0);
        // Observer at peak (2,2): all surrounding cells should be visible
        assert!(vis[2][2], "Observer cell visible");
        assert!(vis[0][0], "Corner visible from peak");
    }

    // ── Line String ───────────────────────────────────────────────────────────

    #[test]
    fn test_linestring_length() {
        let pts = vec![
            Point2D::new(0.0,0.0), Point2D::new(3.0,0.0), Point2D::new(3.0,4.0)
        ];
        let ls = LineString::new(pts);
        assert!((ls.length() - 7.0).abs() < 1e-10, "Length: {}", ls.length());
    }

    #[test]
    fn test_linestring_bbox() {
        let pts = vec![
            Point2D::new(1.0, 2.0), Point2D::new(5.0, -1.0), Point2D::new(3.0, 4.0)
        ];
        let ls = LineString::new(pts);
        let bb = ls.bbox().unwrap();
        assert!((bb.min_x - 1.0).abs() < 1e-10);
        assert!((bb.max_x - 5.0).abs() < 1e-10);
        assert!((bb.min_y - (-1.0)).abs() < 1e-10);
        assert!((bb.max_y - 4.0).abs() < 1e-10);
    }

    // ── Segment Intersection ──────────────────────────────────────────────────

    #[test]
    fn test_segments_intersect_cross() {
        let a1 = Point2D::new(0.0, 0.0); let a2 = Point2D::new(2.0, 2.0);
        let b1 = Point2D::new(0.0, 2.0); let b2 = Point2D::new(2.0, 0.0);
        assert!(segments_intersect(&a1, &a2, &b1, &b2));
    }

    #[test]
    fn test_segments_no_intersect_parallel() {
        let a1 = Point2D::new(0.0, 0.0); let a2 = Point2D::new(2.0, 0.0);
        let b1 = Point2D::new(0.0, 1.0); let b2 = Point2D::new(2.0, 1.0);
        assert!(!segments_intersect(&a1, &a2, &b1, &b2));
    }
}
