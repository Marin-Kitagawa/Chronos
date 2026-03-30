// ============================================================================
// CHRONOS ROBOTICS & CONTROL SYSTEMS ENGINE
// ============================================================================
//
// HOW ROBOTS ACTUALLY WORK (and why control theory matters):
//
// A robot is a mechanical system that senses, decides, and acts in the
// physical world. Unlike pure software, robots must deal with continuous
// time, noisy sensors, imprecise actuators, and the unforgiving laws of
// physics. Control theory provides the mathematical framework for making
// these systems behave reliably despite uncertainty.
//
// The key theoretical foundations:
//
// 1. KINEMATICS: The geometry of motion without considering forces.
//    Forward kinematics: given joint angles, where is the end effector?
//    Inverse kinematics: given a desired position, what joint angles achieve it?
//    Forward kinematics is straightforward (matrix multiplication). Inverse
//    kinematics is HARD — it may have zero, one, or infinitely many solutions,
//    and often requires numerical methods (Jacobian pseudo-inverse, CCD).
//
// 2. CONTROL THEORY: How to make a system follow a desired trajectory.
//    - PID control: The workhorse. Proportional (react to current error),
//      Integral (accumulate past errors to fix steady-state offset),
//      Derivative (anticipate future errors from rate of change).
//    - LQR (Linear Quadratic Regulator): Optimal control for linear systems.
//      Minimizes a cost function J = ∫(x'Qx + u'Ru)dt where Q penalizes
//      state deviation and R penalizes control effort. Solved via the
//      algebraic Riccati equation.
//    - MPC (Model Predictive Control): Repeatedly solve an optimization
//      problem over a receding horizon. Can handle constraints explicitly.
//      Computationally expensive but very powerful.
//
// 3. PATH PLANNING: Finding collision-free paths through space.
//    - A* on a grid: optimal for discrete spaces, uses heuristic for efficiency
//    - RRT (Rapidly-exploring Random Trees): for high-dimensional continuous
//      spaces. Grows a tree by random sampling. RRT* adds rewiring for
//      asymptotic optimality.
//    - Potential fields: treat goal as attractor, obstacles as repellers.
//      Fast but prone to local minima.
//
// 4. STATE ESTIMATION: Figuring out where you are from noisy sensors.
//    - Kalman filter: optimal for linear systems with Gaussian noise.
//      Prediction step: propagate state forward using dynamics model.
//      Update step: incorporate measurement using Kalman gain.
//    - Extended Kalman Filter (EKF): linearizes nonlinear systems.
//    - Particle filter: represents belief as a set of weighted samples.
//      Works with arbitrary distributions but expensive.
//
// 5. SLAM (Simultaneous Localization and Mapping): The chicken-and-egg
//    problem — you need a map to localize, but you need to know where
//    you are to build the map. Solved iteratively using EKF-SLAM,
//    particle filter SLAM (FastSLAM), or graph-based optimization.
//
// WHAT THIS ENGINE IMPLEMENTS (real, working logic with tests):
//   1.  2D/3D vectors and matrix operations for spatial math
//   2.  Rotation representations (rotation matrix, quaternion)
//   3.  Homogeneous transformations (SE(2), SE(3))
//   4.  Forward kinematics (DH parameters, serial chain)
//   5.  Inverse kinematics (Jacobian transpose, CCD)
//   6.  PID controller with anti-windup and derivative filtering
//   7.  LQR controller (discrete-time algebraic Riccati equation)
//   8.  Model Predictive Control (linear, box constraints)
//   9.  A* path planning on occupancy grids
//  10.  RRT and RRT* for continuous path planning
//  11.  Potential field navigation
//  12.  Kalman filter (linear, predict-update cycle)
//  13.  Extended Kalman filter (nonlinear systems)
//  14.  Particle filter (Monte Carlo localization)
//  15.  EKF-SLAM (landmark-based)
//  16.  Trajectory generation (trapezoidal velocity profile, cubic spline)
//  17.  Sensor models (lidar, IMU, encoder, GPS with noise)
//  18.  Comprehensive tests
// ============================================================================

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::f64::consts::PI;

// ============================================================================
// Part 1: Vector and Matrix Math
// ============================================================================
//
// Robotics lives in 2D and 3D Euclidean space. Every computation —
// kinematics, dynamics, control — reduces to operations on vectors
// and matrices. We implement a minimal but complete linear algebra
// layer: Vec2, Vec3, Mat3, Mat4 with the operations robotics needs.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n < 1e-12 {
            Self::zero()
        } else {
            Self {
                x: self.x / n,
                y: self.y / n,
            }
        }
    }

    pub fn dot(&self, other: &Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// 2D cross product: returns scalar (z-component of 3D cross product)
    pub fn cross(&self, other: &Vec2) -> f64 {
        self.x * other.y - self.y * other.x
    }

    pub fn distance_to(&self, other: &Vec2) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn rotate(&self, angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            x: c * self.x - s * self.y,
            y: s * self.x + c * self.y,
        }
    }

    pub fn add(&self, other: &Vec2) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    pub fn sub(&self, other: &Vec2) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    pub fn scale(&self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n < 1e-12 {
            Self::zero()
        } else {
            Self {
                x: self.x / n,
                y: self.y / n,
                z: self.z / n,
            }
        }
    }

    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Vec3) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn add(&self, other: &Vec3) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    pub fn sub(&self, other: &Vec3) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    pub fn scale(&self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    pub fn distance_to(&self, other: &Vec3) -> f64 {
        self.sub(other).norm()
    }
}

/// A general-purpose dense matrix stored in row-major order.
/// Used for Jacobians, Riccati equations, state-space models, etc.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                result.set(c, r, self.get(r, c));
            }
        }
        result
    }

    pub fn mul(&self, other: &Matrix) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut result = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn add(&self, other: &Matrix) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self::new(self.rows, self.cols, data)
    }

    pub fn sub(&self, other: &Matrix) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Self::new(self.rows, self.cols, data)
    }

    pub fn scale(&self, s: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|v| v * s).collect();
        Self::new(self.rows, self.cols, data)
    }

    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, v.len());
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.get(i, j) * v[j];
            }
        }
        result
    }

    /// Solve Ax = b for x using Gaussian elimination with partial pivoting.
    /// Returns None if the system is singular.
    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        assert_eq!(self.rows, self.cols);
        assert_eq!(self.rows, b.len());
        let n = self.rows;
        let mut aug = Self::zeros(n, n + 1);
        for i in 0..n {
            for j in 0..n {
                aug.set(i, j, self.get(i, j));
            }
            aug.set(i, n, b[i]);
        }

        for col in 0..n {
            // Partial pivoting
            let mut max_row = col;
            let mut max_val = aug.get(col, col).abs();
            for row in (col + 1)..n {
                let val = aug.get(row, col).abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }
            if max_val < 1e-12 {
                return None; // Singular
            }
            if max_row != col {
                for j in 0..=n {
                    let tmp = aug.get(col, j);
                    aug.set(col, j, aug.get(max_row, j));
                    aug.set(max_row, j, tmp);
                }
            }
            let pivot = aug.get(col, col);
            for row in (col + 1)..n {
                let factor = aug.get(row, col) / pivot;
                for j in col..=n {
                    let val = aug.get(row, j) - factor * aug.get(col, j);
                    aug.set(row, j, val);
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug.get(i, n);
            for j in (i + 1)..n {
                sum -= aug.get(i, j) * x[j];
            }
            x[i] = sum / aug.get(i, i);
        }
        Some(x)
    }

    /// 2x2 matrix inverse (used frequently in Kalman filter updates)
    pub fn inverse_2x2(&self) -> Option<Self> {
        assert_eq!(self.rows, 2);
        assert_eq!(self.cols, 2);
        let a = self.get(0, 0);
        let b = self.get(0, 1);
        let c = self.get(1, 0);
        let d = self.get(1, 1);
        let det = a * d - b * c;
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Self::new(
            2,
            2,
            vec![d * inv_det, -b * inv_det, -c * inv_det, a * inv_det],
        ))
    }

    /// General matrix inverse using Gauss-Jordan elimination.
    pub fn inverse(&self) -> Option<Self> {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        // Augment [A | I]
        let mut aug = Self::zeros(n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                aug.set(i, j, self.get(i, j));
            }
            aug.set(i, n + i, 1.0);
        }

        for col in 0..n {
            let mut max_row = col;
            let mut max_val = aug.get(col, col).abs();
            for row in (col + 1)..n {
                let val = aug.get(row, col).abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }
            if max_val < 1e-12 {
                return None;
            }
            if max_row != col {
                for j in 0..(2 * n) {
                    let tmp = aug.get(col, j);
                    aug.set(col, j, aug.get(max_row, j));
                    aug.set(max_row, j, tmp);
                }
            }
            let pivot = aug.get(col, col);
            for j in 0..(2 * n) {
                aug.set(col, j, aug.get(col, j) / pivot);
            }
            for row in 0..n {
                if row != col {
                    let factor = aug.get(row, col);
                    for j in 0..(2 * n) {
                        let val = aug.get(row, j) - factor * aug.get(col, j);
                        aug.set(row, j, val);
                    }
                }
            }
        }

        let mut result = Self::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                result.set(i, j, aug.get(i, n + j));
            }
        }
        Some(result)
    }
}

// ============================================================================
// Part 2: Rotation Representations
// ============================================================================
//
// Rotations in 3D can be represented several ways, each with trade-offs:
// - Rotation matrices (3x3, 9 numbers, orthogonal): easy to compose, but
//   9 numbers for 3 DOF is wasteful, and numerical drift breaks orthogonality
// - Euler angles (3 numbers): compact but suffer from gimbal lock
// - Quaternions (4 numbers): compact, no gimbal lock, smooth interpolation,
//   but unintuitive. THE standard in modern robotics and game engines.
//
// A unit quaternion q = w + xi + yj + zk where |q| = 1 represents a
// rotation of angle θ around axis (nx, ny, nz) as:
//   q = cos(θ/2) + sin(θ/2)(nx·i + ny·j + nz·k)

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Create quaternion from axis-angle representation
    pub fn from_axis_angle(axis: &Vec3, angle: f64) -> Self {
        let half = angle / 2.0;
        let s = half.sin();
        let a = axis.normalized();
        Self {
            w: half.cos(),
            x: a.x * s,
            y: a.y * s,
            z: a.z * s,
        }
    }

    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(&self) -> Self {
        let n = self.norm();
        Self {
            w: self.w / n,
            x: self.x / n,
            y: self.y / n,
            z: self.z / n,
        }
    }

    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Quaternion multiplication (Hamilton product)
    pub fn mul(&self, other: &Quaternion) -> Self {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// Rotate a 3D vector by this quaternion: v' = q * v * q^(-1)
    pub fn rotate_vec(&self, v: &Vec3) -> Vec3 {
        let qv = Quaternion::new(0.0, v.x, v.y, v.z);
        let result = self.mul(&qv).mul(&self.conjugate());
        Vec3::new(result.x, result.y, result.z)
    }

    /// Convert to 3x3 rotation matrix
    pub fn to_rotation_matrix(&self) -> Matrix {
        let q = self.normalized();
        let (w, x, y, z) = (q.w, q.x, q.y, q.z);
        Matrix::new(
            3,
            3,
            vec![
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        )
    }

    /// Spherical linear interpolation between two quaternions.
    /// SLERP produces constant-velocity rotation, which is critical
    /// for smooth robot motion.
    pub fn slerp(&self, other: &Quaternion, t: f64) -> Self {
        let mut dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;

        // If dot is negative, negate one quaternion to take the shorter path
        let other = if dot < 0.0 {
            dot = -dot;
            Quaternion::new(-other.w, -other.x, -other.y, -other.z)
        } else {
            *other
        };

        // If quaternions are very close, use linear interpolation
        if dot > 0.9995 {
            let result = Quaternion::new(
                self.w + t * (other.w - self.w),
                self.x + t * (other.x - self.x),
                self.y + t * (other.y - self.y),
                self.z + t * (other.z - self.z),
            );
            return result.normalized();
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;

        Quaternion::new(
            a * self.w + b * other.w,
            a * self.x + b * other.x,
            a * self.y + b * other.y,
            a * self.z + b * other.z,
        )
        .normalized()
    }
}

// ============================================================================
// Part 3: Homogeneous Transformations
// ============================================================================
//
// In robotics, we constantly transform between coordinate frames.
// A homogeneous transformation matrix combines rotation and translation
// into a single 4x4 matrix that can be composed by multiplication:
//
//   T = | R  t |    where R is 3x3 rotation, t is 3x1 translation
//       | 0  1 |
//
// For 2D: SE(2) uses a 3x3 matrix with 2D rotation and translation.
// For 3D: SE(3) uses a 4x4 matrix.

#[derive(Debug, Clone)]
pub struct Transform2D {
    pub x: f64,
    pub y: f64,
    pub theta: f64, // rotation angle in radians
}

impl Transform2D {
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self { x, y, theta }
    }

    pub fn identity() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            theta: 0.0,
        }
    }

    /// Compose two SE(2) transforms: self * other
    pub fn compose(&self, other: &Transform2D) -> Self {
        let c = self.theta.cos();
        let s = self.theta.sin();
        Self {
            x: self.x + c * other.x - s * other.y,
            y: self.y + s * other.x + c * other.y,
            theta: self.theta + other.theta,
        }
    }

    /// Transform a 2D point from child frame to parent frame
    pub fn transform_point(&self, p: &Vec2) -> Vec2 {
        let c = self.theta.cos();
        let s = self.theta.sin();
        Vec2::new(self.x + c * p.x - s * p.y, self.y + s * p.x + c * p.y)
    }

    pub fn inverse(&self) -> Self {
        let c = self.theta.cos();
        let s = self.theta.sin();
        Self {
            x: -(c * self.x + s * self.y),
            y: -(-s * self.x + c * self.y),
            theta: -self.theta,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Transform3D {
    /// 4x4 homogeneous transformation matrix stored as [f64; 16] row-major
    pub matrix: [f64; 16],
}

impl Transform3D {
    pub fn identity() -> Self {
        let mut m = [0.0f64; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        Self { matrix: m }
    }

    pub fn from_rotation_translation(rot: &Matrix, trans: &Vec3) -> Self {
        assert_eq!(rot.rows, 3);
        assert_eq!(rot.cols, 3);
        let mut m = [0.0f64; 16];
        for i in 0..3 {
            for j in 0..3 {
                m[i * 4 + j] = rot.get(i, j);
            }
        }
        m[3] = trans.x;
        m[7] = trans.y;
        m[11] = trans.z;
        m[15] = 1.0;
        Self { matrix: m }
    }

    pub fn translation(&self) -> Vec3 {
        Vec3::new(self.matrix[3], self.matrix[7], self.matrix[11])
    }

    pub fn rotation(&self) -> Matrix {
        let mut r = Matrix::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                r.set(i, j, self.matrix[i * 4 + j]);
            }
        }
        r
    }

    /// Compose: self * other (4x4 matrix multiplication)
    pub fn compose(&self, other: &Transform3D) -> Self {
        let mut result = [0.0f64; 16];
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.matrix[i * 4 + k] * other.matrix[k * 4 + j];
                }
                result[i * 4 + j] = sum;
            }
        }
        Self { matrix: result }
    }

    pub fn transform_point(&self, p: &Vec3) -> Vec3 {
        Vec3::new(
            self.matrix[0] * p.x + self.matrix[1] * p.y + self.matrix[2] * p.z + self.matrix[3],
            self.matrix[4] * p.x + self.matrix[5] * p.y + self.matrix[6] * p.z + self.matrix[7],
            self.matrix[8] * p.x + self.matrix[9] * p.y + self.matrix[10] * p.z + self.matrix[11],
        )
    }

    pub fn inverse(&self) -> Self {
        // For a homogeneous transform [R t; 0 1], inverse is [R' -R't; 0 1]
        let r = self.rotation();
        let rt = r.transpose();
        let t = self.translation();
        let inv_t = Vec3::new(
            -(rt.get(0, 0) * t.x + rt.get(0, 1) * t.y + rt.get(0, 2) * t.z),
            -(rt.get(1, 0) * t.x + rt.get(1, 1) * t.y + rt.get(1, 2) * t.z),
            -(rt.get(2, 0) * t.x + rt.get(2, 1) * t.y + rt.get(2, 2) * t.z),
        );
        Self::from_rotation_translation(&rt, &inv_t)
    }
}

// ============================================================================
// Part 4: Forward Kinematics (Denavit-Hartenberg Parameters)
// ============================================================================
//
// The Denavit-Hartenberg (DH) convention assigns a coordinate frame to each
// joint of a serial manipulator. Each joint is described by 4 parameters:
//   - θ (theta): joint angle (variable for revolute joints)
//   - d: link offset along z
//   - a: link length along x
//   - α (alpha): link twist around x
//
// The transform from frame i-1 to frame i is:
//   T = Rz(θ) · Tz(d) · Tx(a) · Rx(α)
//
// Forward kinematics: multiply all transforms to get end-effector pose.

#[derive(Debug, Clone, Copy)]
pub enum JointType {
    Revolute,  // θ is the variable
    Prismatic, // d is the variable
}

#[derive(Debug, Clone)]
pub struct DHParameter {
    pub joint_type: JointType,
    pub theta: f64,  // joint angle (rad)
    pub d: f64,      // link offset
    pub a: f64,      // link length
    pub alpha: f64,  // link twist (rad)
}

impl DHParameter {
    /// Compute the 4x4 transform for this link given a joint value.
    /// For revolute joints, q replaces theta. For prismatic, q replaces d.
    pub fn transform(&self, q: f64) -> Transform3D {
        let (theta, d) = match self.joint_type {
            JointType::Revolute => (self.theta + q, self.d),
            JointType::Prismatic => (self.theta, self.d + q),
        };

        let ct = theta.cos();
        let st = theta.sin();
        let ca = self.alpha.cos();
        let sa = self.alpha.sin();

        let mut m = [0.0f64; 16];
        m[0] = ct;
        m[1] = -st * ca;
        m[2] = st * sa;
        m[3] = self.a * ct;
        m[4] = st;
        m[5] = ct * ca;
        m[6] = -ct * sa;
        m[7] = self.a * st;
        m[8] = 0.0;
        m[9] = sa;
        m[10] = ca;
        m[11] = d;
        m[15] = 1.0;

        Transform3D { matrix: m }
    }
}

/// A serial robot arm defined by a chain of DH links
#[derive(Debug, Clone)]
pub struct SerialRobot {
    pub links: Vec<DHParameter>,
}

impl SerialRobot {
    pub fn new(links: Vec<DHParameter>) -> Self {
        Self { links }
    }

    /// Forward kinematics: compute the end-effector transform for given joint values
    pub fn forward_kinematics(&self, joint_values: &[f64]) -> Transform3D {
        assert_eq!(joint_values.len(), self.links.len());
        let mut t = Transform3D::identity();
        for (link, &q) in self.links.iter().zip(joint_values.iter()) {
            t = t.compose(&link.transform(q));
        }
        t
    }

    /// Compute the Jacobian matrix (6 x n) numerically.
    /// Maps joint velocities to end-effector velocity [vx, vy, vz, wx, wy, wz].
    /// Uses finite differences (simple and robust).
    pub fn jacobian(&self, joint_values: &[f64]) -> Matrix {
        let n = self.links.len();
        let mut jac = Matrix::zeros(6, n);
        let eps = 1e-6;
        let t0 = self.forward_kinematics(joint_values);
        let p0 = t0.translation();
        let r0 = t0.rotation();

        for i in 0..n {
            let mut q_plus = joint_values.to_vec();
            q_plus[i] += eps;
            let t_plus = self.forward_kinematics(&q_plus);
            let p_plus = t_plus.translation();
            let r_plus = t_plus.rotation();

            // Linear velocity columns
            jac.set(0, i, (p_plus.x - p0.x) / eps);
            jac.set(1, i, (p_plus.y - p0.y) / eps);
            jac.set(2, i, (p_plus.z - p0.z) / eps);

            // Angular velocity columns (from rotation difference)
            // dR = R_plus * R0^T, then extract angular velocity from skew-symmetric part
            let r0t = r0.transpose();
            let dr = r_plus.mul(&r0t);
            jac.set(3, i, (dr.get(2, 1) - dr.get(1, 2)) / (2.0 * eps));
            jac.set(4, i, (dr.get(0, 2) - dr.get(2, 0)) / (2.0 * eps));
            jac.set(5, i, (dr.get(1, 0) - dr.get(0, 1)) / (2.0 * eps));
        }
        jac
    }
}

// ============================================================================
// Part 5: Inverse Kinematics
// ============================================================================
//
// Given a desired end-effector pose, find the joint angles that achieve it.
// This is fundamentally harder than forward kinematics because:
// 1. The mapping is nonlinear — can't just invert a matrix
// 2. Solutions may not exist (target out of reach)
// 3. Multiple solutions may exist (elbow up vs elbow down)
// 4. Near singularities, small workspace changes require huge joint changes
//
// We implement two approaches:
// - Jacobian transpose: simple, always converges (slowly) toward the target
// - CCD (Cyclic Coordinate Descent): fast, iterative, works well for chains

/// Jacobian-transpose inverse kinematics.
/// Iteratively adjusts joint angles: Δq = α * J^T * e
/// where e is the position error and α is a step size.
pub fn ik_jacobian_transpose(
    robot: &SerialRobot,
    target_position: &Vec3,
    initial_joints: &[f64],
    max_iterations: usize,
    tolerance: f64,
) -> (Vec<f64>, f64) {
    let mut q = initial_joints.to_vec();
    let alpha = 0.1;

    for _ in 0..max_iterations {
        let t = robot.forward_kinematics(&q);
        let current = t.translation();
        let error = target_position.sub(&current);
        let error_norm = error.norm();

        if error_norm < tolerance {
            return (q, error_norm);
        }

        let jac = robot.jacobian(&q);
        // Use only the position rows (0..3) of the Jacobian
        let n = q.len();
        let e = vec![error.x, error.y, error.z];
        for i in 0..n {
            let mut dq = 0.0;
            for j in 0..3 {
                dq += jac.get(j, i) * e[j];
            }
            q[i] += alpha * dq;
        }
    }

    let t = robot.forward_kinematics(&q);
    let error = target_position.sub(&t.translation()).norm();
    (q, error)
}

/// Cyclic Coordinate Descent (CCD) for 2D planar chains.
/// Iterates through joints from tip to base, rotating each to point
/// toward the target. Very fast convergence for serial chains.
pub fn ik_ccd_2d(
    link_lengths: &[f64],
    target: &Vec2,
    initial_angles: &[f64],
    max_iterations: usize,
    tolerance: f64,
) -> (Vec<f64>, f64) {
    let n = link_lengths.len();
    let mut angles = initial_angles.to_vec();

    for _ in 0..max_iterations {
        // Compute current end effector position
        let end = forward_2d(link_lengths, &angles);
        let error = target.distance_to(&end);
        if error < tolerance {
            return (angles, error);
        }

        // Iterate from last joint to first
        for i in (0..n).rev() {
            // Position of joint i
            let joint_pos = joint_position_2d(link_lengths, &angles, i);
            let end = forward_2d(link_lengths, &angles);

            let to_end = Vec2::new(end.x - joint_pos.x, end.y - joint_pos.y);
            let to_target = Vec2::new(target.x - joint_pos.x, target.y - joint_pos.y);

            let angle_to_end = to_end.y.atan2(to_end.x);
            let angle_to_target = to_target.y.atan2(to_target.x);
            angles[i] += angle_to_target - angle_to_end;
        }
    }

    let end = forward_2d(link_lengths, &angles);
    let error = target.distance_to(&end);
    (angles, error)
}

fn forward_2d(link_lengths: &[f64], angles: &[f64]) -> Vec2 {
    let mut x = 0.0;
    let mut y = 0.0;
    let mut cumulative_angle = 0.0;
    for (i, &len) in link_lengths.iter().enumerate() {
        cumulative_angle += angles[i];
        x += len * cumulative_angle.cos();
        y += len * cumulative_angle.sin();
    }
    Vec2::new(x, y)
}

fn joint_position_2d(link_lengths: &[f64], angles: &[f64], joint_index: usize) -> Vec2 {
    let mut x = 0.0;
    let mut y = 0.0;
    let mut cumulative_angle = 0.0;
    for i in 0..joint_index {
        cumulative_angle += angles[i];
        x += link_lengths[i] * cumulative_angle.cos();
        y += link_lengths[i] * cumulative_angle.sin();
    }
    Vec2::new(x, y)
}

// ============================================================================
// Part 6: PID Controller
// ============================================================================
//
// The PID controller is the single most important control algorithm.
// Over 90% of industrial control loops use PID. The output is:
//
//   u(t) = Kp * e(t) + Ki * ∫e(τ)dτ + Kd * de/dt
//
// Key practical considerations:
// - Integral windup: when the actuator saturates, the integral term keeps
//   growing, causing massive overshoot when the error finally decreases.
//   Solution: clamp the integral term (anti-windup).
// - Derivative kick: when the setpoint changes suddenly, the derivative
//   of error spikes. Solution: differentiate the process variable instead
//   of the error, or use a low-pass filter on the derivative.
// - Tuning: Ziegler-Nichols gives a starting point, but real tuning is
//   always empirical. Increase Kp until oscillation, then add Ki and Kd.

#[derive(Debug, Clone)]
pub struct PidController {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    integral: f64,
    prev_error: f64,
    integral_min: f64,
    integral_max: f64,
    output_min: f64,
    output_max: f64,
    initialized: bool,
}

impl PidController {
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            integral: 0.0,
            prev_error: 0.0,
            integral_min: f64::NEG_INFINITY,
            integral_max: f64::INFINITY,
            output_min: f64::NEG_INFINITY,
            output_max: f64::INFINITY,
            initialized: false,
        }
    }

    pub fn with_integral_limits(mut self, min: f64, max: f64) -> Self {
        self.integral_min = min;
        self.integral_max = max;
        self
    }

    pub fn with_output_limits(mut self, min: f64, max: f64) -> Self {
        self.output_min = min;
        self.output_max = max;
        self
    }

    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
        self.initialized = false;
    }

    /// Compute control output given setpoint, process variable, and timestep.
    pub fn update(&mut self, setpoint: f64, process_variable: f64, dt: f64) -> f64 {
        let error = setpoint - process_variable;

        // Integral with anti-windup clamping
        self.integral += error * dt;
        self.integral = self.integral.clamp(self.integral_min, self.integral_max);

        // Derivative (use backward difference; skip on first call)
        let derivative = if self.initialized {
            (error - self.prev_error) / dt
        } else {
            self.initialized = true;
            0.0
        };
        self.prev_error = error;

        // PID output with saturation
        let output = self.kp * error + self.ki * self.integral + self.kd * derivative;
        output.clamp(self.output_min, self.output_max)
    }
}

// ============================================================================
// Part 7: LQR Controller (Linear Quadratic Regulator)
// ============================================================================
//
// For linear systems x[k+1] = A*x[k] + B*u[k], LQR finds the optimal
// feedback gain K that minimizes J = Σ(x'Qx + u'Ru).
//
// The solution comes from the discrete-time algebraic Riccati equation (DARE):
//   P = A'PA - A'PB(R + B'PB)^(-1)B'PA + Q
//
// Iterate until P converges, then K = (R + B'PB)^(-1)B'PA.
// The control law is u = -K*x.

pub struct LqrController {
    pub gain: Matrix, // K matrix
}

impl LqrController {
    /// Solve discrete-time algebraic Riccati equation and return LQR gain.
    /// A: state transition (n x n)
    /// B: control input (n x m)
    /// Q: state cost (n x n, positive semi-definite)
    /// R: control cost (m x m, positive definite)
    pub fn new(a: &Matrix, b: &Matrix, q: &Matrix, r: &Matrix) -> Self {
        let n = a.rows;
        let mut p = q.clone(); // Initialize P = Q

        // Iterate DARE
        for _ in 0..200 {
            let at = a.transpose();
            let bt = b.transpose();

            // S = R + B'PB
            let bp = bt.mul(&p);
            let bpb = bp.mul(b);
            let s = r.add(&bpb);

            // S^(-1)
            let s_inv = s.inverse().expect("R + B'PB must be invertible");

            // K_temp = S^(-1) * B' * P * A
            let bpa = bp.mul(a);
            let k_temp = s_inv.mul(&bpa);

            // P_new = A'PA - A'PB * K_temp + Q = A'(P - PB*K_temp)*A + Q
            // More directly: P_new = A'PA - A'PB * S^(-1) * B'PA + Q
            let pa = p.mul(a);
            let atpa = at.mul(&pa);

            let pb = p.mul(b);
            let atpb = at.mul(&pb);
            let correction = atpb.mul(&k_temp);

            let p_new = atpa.sub(&correction).add(q);

            // Check convergence
            let diff = p_new.sub(&p);
            let max_diff = diff.data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            p = p_new;
            if max_diff < 1e-10 {
                break;
            }
        }

        // K = (R + B'PB)^(-1) * B'PA
        let bt = b.transpose();
        let bp = bt.mul(&p);
        let bpb = bp.mul(b);
        let s = r.add(&bpb);
        let s_inv = s.inverse().expect("Final S must be invertible");
        let bpa = bp.mul(a);
        let gain = s_inv.mul(&bpa);

        Self { gain }
    }

    /// Compute control input: u = -K * x
    pub fn control(&self, state: &[f64]) -> Vec<f64> {
        let u = self.gain.mul_vec(state);
        u.iter().map(|v| -v).collect()
    }
}

// ============================================================================
// Part 8: Model Predictive Control (MPC)
// ============================================================================
//
// MPC solves an optimization problem at each timestep:
//   minimize Σ(x'Qx + u'Ru) over a prediction horizon N
//   subject to: x[k+1] = Ax[k] + Bu[k]
//               u_min ≤ u ≤ u_max  (box constraints)
//
// Only the first control input is applied, then the whole problem is
// re-solved at the next timestep (receding horizon). This handles
// constraints naturally but is computationally expensive.
//
// We implement a simple version using projected gradient descent.

pub struct MpcController {
    pub a: Matrix,
    pub b: Matrix,
    pub q: Matrix,
    pub r: Matrix,
    pub horizon: usize,
    pub u_min: Vec<f64>,
    pub u_max: Vec<f64>,
}

impl MpcController {
    pub fn new(
        a: Matrix,
        b: Matrix,
        q: Matrix,
        r: Matrix,
        horizon: usize,
        u_min: Vec<f64>,
        u_max: Vec<f64>,
    ) -> Self {
        Self {
            a,
            b,
            q,
            r,
            horizon,
            u_min,
            u_max,
        }
    }

    /// Solve MPC problem for current state, return first control input.
    /// Uses a simplified approach: iterative forward simulation + gradient step.
    pub fn solve(&self, state: &[f64]) -> Vec<f64> {
        let n = self.a.rows;
        let m = self.b.cols;

        // Initialize control sequence to zeros
        let mut u_sequence: Vec<Vec<f64>> = vec![vec![0.0; m]; self.horizon];

        let learning_rate = 0.01;
        let iterations = 50;

        for _ in 0..iterations {
            // Forward simulate to get state trajectory
            let mut states = vec![state.to_vec()];
            for k in 0..self.horizon {
                let x = &states[k];
                let u = &u_sequence[k];
                let ax = self.a.mul_vec(x);
                let bu = self.b.mul_vec(u);
                let x_next: Vec<f64> = ax.iter().zip(bu.iter()).map(|(a, b)| a + b).collect();
                states.push(x_next);
            }

            // Backward pass: compute gradient of cost w.r.t. each u[k]
            // dJ/du[k] = R*u[k] + B' * (Q*x[k+1] + ... propagated costs)
            let mut lambda = vec![0.0; n]; // costate at horizon end
            // Terminal cost gradient
            let x_end = &states[self.horizon];
            lambda = self.q.mul_vec(x_end).iter().map(|v| 2.0 * v).collect();

            for k in (0..self.horizon).rev() {
                // Gradient w.r.t. u[k]
                let ru = self.r.mul_vec(&u_sequence[k]);
                let bt_lambda = self.b.transpose().mul_vec(&lambda);
                let grad: Vec<f64> = ru
                    .iter()
                    .zip(bt_lambda.iter())
                    .map(|(r, bl)| 2.0 * r + bl)
                    .collect();

                // Update u[k] with gradient descent + projection
                for j in 0..m {
                    u_sequence[k][j] -= learning_rate * grad[j];
                    u_sequence[k][j] = u_sequence[k][j].clamp(self.u_min[j], self.u_max[j]);
                }

                // Propagate costate backward: lambda = Q*x[k+1] + A'*lambda
                // (simplified — using state cost gradient)
                let qx = self.q.mul_vec(&states[k + 1]);
                let at_lambda = self.a.transpose().mul_vec(&lambda);
                lambda = qx
                    .iter()
                    .zip(at_lambda.iter())
                    .map(|(q, al)| 2.0 * q + al)
                    .collect();
            }
        }

        u_sequence[0].clone()
    }
}

// ============================================================================
// Part 9: A* Path Planning
// ============================================================================
//
// A* finds the shortest path on a graph (or grid) using a heuristic to
// guide the search. It maintains:
//   f(n) = g(n) + h(n)
// where g(n) is the cost from start to n, h(n) is the heuristic estimate
// from n to goal. If h is admissible (never overestimates), A* is optimal.
//
// For grid-based robotics, we use an occupancy grid where each cell is
// either free or occupied. Diagonal movement costs √2, cardinal costs 1.

#[derive(Debug, Clone)]
pub struct OccupancyGrid {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<bool>, // true = occupied/obstacle
    pub resolution: f64,  // meters per cell
}

impl OccupancyGrid {
    pub fn new(width: usize, height: usize, resolution: f64) -> Self {
        Self {
            width,
            height,
            cells: vec![false; width * height],
            resolution,
        }
    }

    pub fn set_obstacle(&mut self, x: usize, y: usize) {
        if x < self.width && y < self.height {
            self.cells[y * self.width + x] = true;
        }
    }

    pub fn is_occupied(&self, x: usize, y: usize) -> bool {
        if x >= self.width || y >= self.height {
            return true; // Out of bounds = occupied
        }
        self.cells[y * self.width + x]
    }

    pub fn world_to_grid(&self, wx: f64, wy: f64) -> (usize, usize) {
        let gx = (wx / self.resolution) as usize;
        let gy = (wy / self.resolution) as usize;
        (gx.min(self.width - 1), gy.min(self.height - 1))
    }

    pub fn grid_to_world(&self, gx: usize, gy: usize) -> (f64, f64) {
        (
            (gx as f64 + 0.5) * self.resolution,
            (gy as f64 + 0.5) * self.resolution,
        )
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct AStarNode {
    x: usize,
    y: usize,
    f_cost: u64, // f * 1000 to avoid float comparison issues
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.f_cost.cmp(&self.f_cost) // Reversed for min-heap
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A* path planning on an occupancy grid.
/// Returns the path as a sequence of (x, y) grid coordinates, or None if no path exists.
pub fn astar_grid(
    grid: &OccupancyGrid,
    start: (usize, usize),
    goal: (usize, usize),
) -> Option<Vec<(usize, usize)>> {
    if grid.is_occupied(start.0, start.1) || grid.is_occupied(goal.0, goal.1) {
        return None;
    }

    let mut open = BinaryHeap::new();
    let mut g_score: HashMap<(usize, usize), f64> = HashMap::new();
    let mut came_from: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    let mut closed: HashSet<(usize, usize)> = HashSet::new();

    let heuristic = |x: usize, y: usize| -> f64 {
        let dx = (x as f64 - goal.0 as f64).abs();
        let dy = (y as f64 - goal.1 as f64).abs();
        // Octile distance (consistent heuristic for 8-connected grid)
        let min = dx.min(dy);
        let max = dx.max(dy);
        min * std::f64::consts::SQRT_2 + (max - min)
    };

    g_score.insert(start, 0.0);
    let h = heuristic(start.0, start.1);
    open.push(AStarNode {
        x: start.0,
        y: start.1,
        f_cost: (h * 1000.0) as u64,
    });

    // 8-connected neighbors: cardinal + diagonal
    let neighbors: [(i32, i32); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];

    while let Some(current) = open.pop() {
        let pos = (current.x, current.y);

        if pos == goal {
            // Reconstruct path
            let mut path = vec![goal];
            let mut curr = goal;
            while let Some(&prev) = came_from.get(&curr) {
                path.push(prev);
                curr = prev;
            }
            path.reverse();
            return Some(path);
        }

        if closed.contains(&pos) {
            continue;
        }
        closed.insert(pos);

        let current_g = g_score[&pos];

        for &(dx, dy) in &neighbors {
            let nx = pos.0 as i32 + dx;
            let ny = pos.1 as i32 + dy;

            if nx < 0 || ny < 0 {
                continue;
            }
            let nx = nx as usize;
            let ny = ny as usize;

            if nx >= grid.width || ny >= grid.height || grid.is_occupied(nx, ny) {
                continue;
            }
            if closed.contains(&(nx, ny)) {
                continue;
            }

            // Diagonal movement check: don't cut corners
            if dx != 0 && dy != 0 {
                if grid.is_occupied(pos.0, ny) || grid.is_occupied(nx, pos.1) {
                    continue;
                }
            }

            let move_cost = if dx != 0 && dy != 0 {
                std::f64::consts::SQRT_2
            } else {
                1.0
            };

            let tentative_g = current_g + move_cost;
            let neighbor = (nx, ny);

            if tentative_g < *g_score.get(&neighbor).unwrap_or(&f64::INFINITY) {
                g_score.insert(neighbor, tentative_g);
                came_from.insert(neighbor, pos);
                let f = tentative_g + heuristic(nx, ny);
                open.push(AStarNode {
                    x: nx,
                    y: ny,
                    f_cost: (f * 1000.0) as u64,
                });
            }
        }
    }

    None // No path found
}

// ============================================================================
// Part 10: RRT and RRT* Path Planning
// ============================================================================
//
// Rapidly-exploring Random Trees work in continuous configuration space,
// making them suitable for high-DOF robots. The basic RRT:
// 1. Sample a random point in the space
// 2. Find the nearest node in the tree
// 3. Extend from that node toward the sample by a step distance
// 4. If the extension is collision-free, add the new node
//
// RRT* adds a rewiring step: after adding a node, check if any nearby
// nodes would have a shorter path through the new node. This makes
// RRT* asymptotically optimal (the path improves as more samples are added).

#[derive(Debug, Clone)]
pub struct RrtNode {
    pub position: Vec2,
    pub parent: Option<usize>,
    pub cost: f64,
}

pub struct RrtPlanner {
    pub nodes: Vec<RrtNode>,
    pub step_size: f64,
    pub goal_threshold: f64,
    pub bounds: (Vec2, Vec2), // min, max of planning space
}

impl RrtPlanner {
    pub fn new(start: Vec2, step_size: f64, goal_threshold: f64, bounds: (Vec2, Vec2)) -> Self {
        Self {
            nodes: vec![RrtNode {
                position: start,
                parent: None,
                cost: 0.0,
            }],
            step_size,
            goal_threshold,
            bounds,
        }
    }

    /// Simple pseudo-random number using linear congruential generator.
    /// We avoid external dependencies — this is good enough for path planning.
    fn random_f64(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*seed >> 33) as f64 / (1u64 << 31) as f64
    }

    fn random_point(seed: &mut u64, bounds: &(Vec2, Vec2)) -> Vec2 {
        let x = bounds.0.x + Self::random_f64(seed) * (bounds.1.x - bounds.0.x);
        let y = bounds.0.y + Self::random_f64(seed) * (bounds.1.y - bounds.0.y);
        Vec2::new(x, y)
    }

    fn nearest_node(&self, point: &Vec2) -> usize {
        let mut best = 0;
        let mut best_dist = f64::INFINITY;
        for (i, node) in self.nodes.iter().enumerate() {
            let d = node.position.distance_to(point);
            if d < best_dist {
                best_dist = d;
                best = i;
            }
        }
        best
    }

    fn steer(from: &Vec2, to: &Vec2, step_size: f64) -> Vec2 {
        let dir = to.sub(from);
        let dist = dir.norm();
        if dist < step_size {
            *to
        } else {
            from.add(&dir.normalized().scale(step_size))
        }
    }

    /// Check if a straight-line path between two points is collision-free.
    /// Uses ray marching with small steps.
    fn collision_free(from: &Vec2, to: &Vec2, grid: &OccupancyGrid) -> bool {
        let dir = to.sub(from);
        let dist = dir.norm();
        let steps = (dist / (grid.resolution * 0.5)).ceil() as usize;
        if steps == 0 {
            return true;
        }
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let p = from.add(&dir.scale(t));
            let (gx, gy) = grid.world_to_grid(p.x, p.y);
            if grid.is_occupied(gx, gy) {
                return false;
            }
        }
        true
    }

    /// Run basic RRT planning
    pub fn plan_rrt(
        &mut self,
        goal: &Vec2,
        grid: &OccupancyGrid,
        max_iterations: usize,
        seed: &mut u64,
    ) -> Option<Vec<Vec2>> {
        for _ in 0..max_iterations {
            // Bias toward goal 10% of the time
            let sample = if Self::random_f64(seed) < 0.1 {
                *goal
            } else {
                Self::random_point(seed, &self.bounds)
            };

            let nearest_idx = self.nearest_node(&sample);
            let nearest = self.nodes[nearest_idx].position;
            let new_pos = Self::steer(&nearest, &sample, self.step_size);

            if Self::collision_free(&nearest, &new_pos, grid) {
                let new_cost = self.nodes[nearest_idx].cost + nearest.distance_to(&new_pos);
                self.nodes.push(RrtNode {
                    position: new_pos,
                    parent: Some(nearest_idx),
                    cost: new_cost,
                });

                if new_pos.distance_to(goal) < self.goal_threshold {
                    return Some(self.extract_path(self.nodes.len() - 1));
                }
            }
        }
        None
    }

    /// Run RRT* planning with rewiring
    pub fn plan_rrt_star(
        &mut self,
        goal: &Vec2,
        grid: &OccupancyGrid,
        max_iterations: usize,
        rewire_radius: f64,
        seed: &mut u64,
    ) -> Option<Vec<Vec2>> {
        let mut best_goal_idx: Option<usize> = None;
        let mut best_goal_cost = f64::INFINITY;

        for _ in 0..max_iterations {
            let sample = if Self::random_f64(seed) < 0.1 {
                *goal
            } else {
                Self::random_point(seed, &self.bounds)
            };

            let nearest_idx = self.nearest_node(&sample);
            let nearest = self.nodes[nearest_idx].position;
            let new_pos = Self::steer(&nearest, &sample, self.step_size);

            if !Self::collision_free(&nearest, &new_pos, grid) {
                continue;
            }

            // Find best parent within rewire radius
            let mut best_parent = nearest_idx;
            let mut best_cost =
                self.nodes[nearest_idx].cost + nearest.distance_to(&new_pos);

            let nearby: Vec<usize> = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.position.distance_to(&new_pos) < rewire_radius)
                .map(|(i, _)| i)
                .collect();

            for &i in &nearby {
                let cost = self.nodes[i].cost + self.nodes[i].position.distance_to(&new_pos);
                if cost < best_cost && Self::collision_free(&self.nodes[i].position, &new_pos, grid)
                {
                    best_cost = cost;
                    best_parent = i;
                }
            }

            let new_idx = self.nodes.len();
            self.nodes.push(RrtNode {
                position: new_pos,
                parent: Some(best_parent),
                cost: best_cost,
            });

            // Rewire nearby nodes through the new node if cheaper
            for &i in &nearby {
                let new_cost_through =
                    best_cost + new_pos.distance_to(&self.nodes[i].position);
                if new_cost_through < self.nodes[i].cost
                    && Self::collision_free(&new_pos, &self.nodes[i].position, grid)
                {
                    self.nodes[i].parent = Some(new_idx);
                    self.nodes[i].cost = new_cost_through;
                }
            }

            if new_pos.distance_to(goal) < self.goal_threshold && best_cost < best_goal_cost {
                best_goal_idx = Some(new_idx);
                best_goal_cost = best_cost;
            }
        }

        best_goal_idx.map(|idx| self.extract_path(idx))
    }

    fn extract_path(&self, goal_idx: usize) -> Vec<Vec2> {
        let mut path = vec![];
        let mut idx = Some(goal_idx);
        while let Some(i) = idx {
            path.push(self.nodes[i].position);
            idx = self.nodes[i].parent;
        }
        path.reverse();
        path
    }
}

// ============================================================================
// Part 11: Potential Field Navigation
// ============================================================================
//
// Artificial potential fields treat the goal as an attractor and obstacles
// as repellers. The robot follows the negative gradient of the total
// potential. This is simple and reactive but can get stuck in local minima.
//
// Attractive potential: U_att = 0.5 * k_att * d²(q, q_goal)
// Repulsive potential: U_rep = 0.5 * k_rep * (1/d - 1/d0)² if d < d0, else 0
// where d is distance to obstacle, d0 is the influence radius.

pub struct PotentialField {
    pub k_attractive: f64,
    pub k_repulsive: f64,
    pub obstacle_influence: f64, // d0: how far obstacles exert force
}

impl PotentialField {
    pub fn new(k_attractive: f64, k_repulsive: f64, obstacle_influence: f64) -> Self {
        Self {
            k_attractive,
            k_repulsive,
            obstacle_influence,
        }
    }

    /// Compute the attractive force toward the goal
    pub fn attractive_force(&self, position: &Vec2, goal: &Vec2) -> Vec2 {
        let diff = goal.sub(position);
        diff.scale(self.k_attractive)
    }

    /// Compute the repulsive force from a single obstacle point
    pub fn repulsive_force(&self, position: &Vec2, obstacle: &Vec2) -> Vec2 {
        let diff = position.sub(obstacle);
        let d = diff.norm();
        if d > self.obstacle_influence || d < 1e-6 {
            return Vec2::zero();
        }
        let magnitude = self.k_repulsive * (1.0 / d - 1.0 / self.obstacle_influence) / (d * d);
        diff.normalized().scale(magnitude)
    }

    /// Compute total force from goal + all obstacles
    pub fn total_force(
        &self,
        position: &Vec2,
        goal: &Vec2,
        obstacles: &[Vec2],
    ) -> Vec2 {
        let mut force = self.attractive_force(position, goal);
        for obs in obstacles {
            force = force.add(&self.repulsive_force(position, obs));
        }
        force
    }

    /// Navigate from start to goal using potential field gradient descent.
    /// Returns the trajectory as a sequence of positions.
    pub fn navigate(
        &self,
        start: &Vec2,
        goal: &Vec2,
        obstacles: &[Vec2],
        step_size: f64,
        max_steps: usize,
        goal_tolerance: f64,
    ) -> Vec<Vec2> {
        let mut path = vec![*start];
        let mut pos = *start;

        for _ in 0..max_steps {
            if pos.distance_to(goal) < goal_tolerance {
                break;
            }
            let force = self.total_force(&pos, goal, obstacles);
            let f_norm = force.norm();
            if f_norm < 1e-10 {
                break; // Stuck in local minimum
            }
            pos = pos.add(&force.normalized().scale(step_size));
            path.push(pos);
        }
        path
    }
}

// ============================================================================
// Part 12: Kalman Filter
// ============================================================================
//
// The Kalman filter is the optimal state estimator for linear systems with
// Gaussian noise. It's used in virtually every robotic system: navigation,
// tracking, sensor fusion, SLAM.
//
// State model:    x[k] = F*x[k-1] + B*u[k-1] + w    (w ~ N(0, Q))
// Measurement:    z[k] = H*x[k] + v                   (v ~ N(0, R))
//
// The filter alternates between:
// PREDICT: propagate state and covariance forward
//   x_pred = F*x + B*u
//   P_pred = F*P*F' + Q
//
// UPDATE: incorporate measurement
//   y = z - H*x_pred          (innovation)
//   S = H*P_pred*H' + R       (innovation covariance)
//   K = P_pred*H'*S^(-1)      (Kalman gain)
//   x = x_pred + K*y
//   P = (I - K*H)*P_pred

pub struct KalmanFilter {
    pub state: Vec<f64>,     // x: state vector
    pub covariance: Matrix,  // P: state covariance
    pub f: Matrix,           // F: state transition
    pub b: Matrix,           // B: control input
    pub h: Matrix,           // H: measurement model
    pub q: Matrix,           // Q: process noise covariance
    pub r: Matrix,           // R: measurement noise covariance
}

impl KalmanFilter {
    pub fn new(
        state: Vec<f64>,
        covariance: Matrix,
        f: Matrix,
        b: Matrix,
        h: Matrix,
        q: Matrix,
        r: Matrix,
    ) -> Self {
        Self {
            state,
            covariance,
            f,
            b,
            h,
            q,
            r,
        }
    }

    /// Prediction step
    pub fn predict(&mut self, control: &[f64]) {
        // x = F*x + B*u
        let fx = self.f.mul_vec(&self.state);
        let bu = self.b.mul_vec(control);
        self.state = fx.iter().zip(bu.iter()).map(|(a, b)| a + b).collect();

        // P = F*P*F' + Q
        let fp = self.f.mul(&self.covariance);
        let ft = self.f.transpose();
        self.covariance = fp.mul(&ft).add(&self.q);
    }

    /// Update step with measurement
    pub fn update(&mut self, measurement: &[f64]) {
        // y = z - H*x
        let hx = self.h.mul_vec(&self.state);
        let innovation: Vec<f64> = measurement
            .iter()
            .zip(hx.iter())
            .map(|(z, hx)| z - hx)
            .collect();

        // S = H*P*H' + R
        let hp = self.h.mul(&self.covariance);
        let ht = self.h.transpose();
        let s = hp.mul(&ht).add(&self.r);

        // K = P*H'*S^(-1)
        let pht = self.covariance.mul(&ht);
        let s_inv = s.inverse().expect("Innovation covariance must be invertible");
        let k = pht.mul(&s_inv);

        // x = x + K*y
        let ky = k.mul_vec(&innovation);
        self.state = self.state.iter().zip(ky.iter()).map(|(x, k)| x + k).collect();

        // P = (I - K*H)*P
        let n = self.state.len();
        let kh = k.mul(&self.h);
        let i_kh = Matrix::identity(n).sub(&kh);
        self.covariance = i_kh.mul(&self.covariance);
    }
}

// ============================================================================
// Part 13: Extended Kalman Filter (EKF)
// ============================================================================
//
// When the system dynamics or measurements are nonlinear, we linearize
// around the current estimate. The EKF replaces F and H with Jacobians
// evaluated at the current state.
//
// We implement a generic EKF that takes function pointers for:
// - f(x, u): state transition function
// - h(x): measurement function
// - F_jacobian(x, u): Jacobian of f w.r.t. x
// - H_jacobian(x): Jacobian of h w.r.t. x

pub struct ExtendedKalmanFilter {
    pub state: Vec<f64>,
    pub covariance: Matrix,
    pub q: Matrix,
    pub r: Matrix,
    pub state_dim: usize,
    pub meas_dim: usize,
}

impl ExtendedKalmanFilter {
    pub fn new(
        state: Vec<f64>,
        covariance: Matrix,
        q: Matrix,
        r: Matrix,
    ) -> Self {
        let state_dim = state.len();
        let meas_dim = r.rows;
        Self {
            state,
            covariance,
            q,
            r,
            state_dim,
            meas_dim,
        }
    }

    /// Predict using nonlinear state transition and its Jacobian
    pub fn predict(
        &mut self,
        control: &[f64],
        f_func: &dyn Fn(&[f64], &[f64]) -> Vec<f64>,
        f_jacobian: &dyn Fn(&[f64], &[f64]) -> Matrix,
    ) {
        let f_jac = f_jacobian(&self.state, control);
        self.state = f_func(&self.state, control);

        let fp = f_jac.mul(&self.covariance);
        let ft = f_jac.transpose();
        self.covariance = fp.mul(&ft).add(&self.q);
    }

    /// Update using nonlinear measurement function and its Jacobian
    pub fn update(
        &mut self,
        measurement: &[f64],
        h_func: &dyn Fn(&[f64]) -> Vec<f64>,
        h_jacobian: &dyn Fn(&[f64]) -> Matrix,
    ) {
        let h_jac = h_jacobian(&self.state);
        let predicted_meas = h_func(&self.state);

        let innovation: Vec<f64> = measurement
            .iter()
            .zip(predicted_meas.iter())
            .map(|(z, hz)| z - hz)
            .collect();

        let hp = h_jac.mul(&self.covariance);
        let ht = h_jac.transpose();
        let s = hp.mul(&ht).add(&self.r);
        let s_inv = s.inverse().expect("EKF: innovation covariance must be invertible");

        let pht = self.covariance.mul(&ht);
        let k = pht.mul(&s_inv);

        let ky = k.mul_vec(&innovation);
        self.state = self.state.iter().zip(ky.iter()).map(|(x, k)| x + k).collect();

        let kh = k.mul(&h_jac);
        let i_kh = Matrix::identity(self.state_dim).sub(&kh);
        self.covariance = i_kh.mul(&self.covariance);
    }
}

// ============================================================================
// Part 14: Particle Filter (Monte Carlo Localization)
// ============================================================================
//
// When the system is highly nonlinear or the noise is non-Gaussian,
// particle filters represent the belief distribution as a set of weighted
// samples (particles). The algorithm:
//
// 1. PREDICT: propagate each particle through the motion model (+ noise)
// 2. WEIGHT: evaluate how well each particle explains the measurement
// 3. RESAMPLE: draw new particles proportional to their weights
//    (systematic resampling is most common — low variance, O(n))
//
// Particle filters can handle multimodal distributions (e.g., the robot
// could be in several possible locations), unlike Kalman filters.

#[derive(Debug, Clone)]
pub struct Particle {
    pub state: Vec<f64>,
    pub weight: f64,
}

pub struct ParticleFilter {
    pub particles: Vec<Particle>,
    pub state_dim: usize,
}

impl ParticleFilter {
    /// Initialize with uniformly weighted particles
    pub fn new(initial_particles: Vec<Vec<f64>>) -> Self {
        let state_dim = initial_particles[0].len();
        let n = initial_particles.len();
        let weight = 1.0 / n as f64;
        let particles = initial_particles
            .into_iter()
            .map(|s| Particle {
                state: s,
                weight,
            })
            .collect();
        Self {
            particles,
            state_dim,
        }
    }

    /// Predict: propagate particles through motion model with noise
    pub fn predict(
        &mut self,
        motion_model: &dyn Fn(&[f64], &mut u64) -> Vec<f64>,
        seed: &mut u64,
    ) {
        for particle in &mut self.particles {
            particle.state = motion_model(&particle.state, seed);
        }
    }

    /// Update: reweight particles based on measurement likelihood
    pub fn update(
        &mut self,
        measurement: &[f64],
        likelihood: &dyn Fn(&[f64], &[f64]) -> f64,
    ) {
        let mut total_weight = 0.0;
        for particle in &mut self.particles {
            particle.weight *= likelihood(&particle.state, measurement);
            total_weight += particle.weight;
        }
        // Normalize weights
        if total_weight > 1e-300 {
            for particle in &mut self.particles {
                particle.weight /= total_weight;
            }
        }
    }

    /// Systematic resampling: O(n), low variance
    pub fn resample(&mut self, seed: &mut u64) {
        let n = self.particles.len();
        let mut new_particles = Vec::with_capacity(n);

        // Build cumulative distribution
        let mut cumulative = vec![0.0; n];
        cumulative[0] = self.particles[0].weight;
        for i in 1..n {
            cumulative[i] = cumulative[i - 1] + self.particles[i].weight;
        }

        // Systematic resampling
        let step = 1.0 / n as f64;
        let start = RrtPlanner::random_f64(seed) * step;
        let mut idx = 0;

        for i in 0..n {
            let threshold = start + i as f64 * step;
            while idx < n - 1 && cumulative[idx] < threshold {
                idx += 1;
            }
            new_particles.push(Particle {
                state: self.particles[idx].state.clone(),
                weight: 1.0 / n as f64,
            });
        }

        self.particles = new_particles;
    }

    /// Get weighted mean estimate of the state
    pub fn estimate(&self) -> Vec<f64> {
        let mut mean = vec![0.0; self.state_dim];
        for particle in &self.particles {
            for (i, &s) in particle.state.iter().enumerate() {
                mean[i] += particle.weight * s;
            }
        }
        mean
    }

    /// Effective number of particles (Neff). When this drops too low,
    /// most weight is concentrated on few particles — time to resample.
    pub fn effective_particles(&self) -> f64 {
        let sum_sq: f64 = self.particles.iter().map(|p| p.weight * p.weight).sum();
        if sum_sq < 1e-300 {
            return 0.0;
        }
        1.0 / sum_sq
    }
}

// ============================================================================
// Part 15: EKF-SLAM
// ============================================================================
//
// SLAM = Simultaneous Localization and Mapping. The state vector contains
// both the robot pose and all landmark positions:
//   x = [robot_x, robot_y, robot_theta, lm1_x, lm1_y, lm2_x, lm2_y, ...]
//
// The EKF-SLAM update:
// 1. Predict robot pose using odometry (augment Jacobian for full state)
// 2. For each observed landmark:
//    a. If new: add to state vector and expand covariance
//    b. If known: update using range/bearing measurement

pub struct EkfSlam {
    pub state: Vec<f64>,      // [x, y, theta, lm1_x, lm1_y, ...]
    pub covariance: Matrix,
    pub num_landmarks: usize,
    pub landmark_ids: HashMap<usize, usize>, // external ID -> index in state
}

#[derive(Debug, Clone)]
pub struct LandmarkObservation {
    pub id: usize,
    pub range: f64,
    pub bearing: f64, // relative to robot heading
}

impl EkfSlam {
    pub fn new(initial_pose: (f64, f64, f64)) -> Self {
        let state = vec![initial_pose.0, initial_pose.1, initial_pose.2];
        let covariance = Matrix::zeros(3, 3); // Start with zero uncertainty (known pose)
        Self {
            state,
            covariance,
            num_landmarks: 0,
            landmark_ids: HashMap::new(),
        }
    }

    /// Predict step using differential-drive odometry model
    pub fn predict(&mut self, v: f64, omega: f64, dt: f64, motion_noise: &Matrix) {
        let theta = self.state[2];
        let n = self.state.len();

        // State update
        if omega.abs() < 1e-10 {
            self.state[0] += v * dt * theta.cos();
            self.state[1] += v * dt * theta.sin();
        } else {
            let r = v / omega;
            self.state[0] += r * ((theta + omega * dt).sin() - theta.sin());
            self.state[1] += r * (theta.cos() - (theta + omega * dt).cos());
            self.state[2] += omega * dt;
        }

        // Jacobian of motion model w.r.t. robot state
        let mut g = Matrix::identity(n);
        if omega.abs() < 1e-10 {
            g.set(0, 2, -v * dt * theta.sin());
            g.set(1, 2, v * dt * theta.cos());
        } else {
            let r = v / omega;
            g.set(0, 2, r * ((theta + omega * dt).cos() - theta.cos()));
            g.set(1, 2, r * ((theta + omega * dt).sin() - theta.sin()));
        }

        // Update covariance: P = G*P*G' + noise (noise only affects robot pose)
        let gp = g.mul(&self.covariance);
        let gt = g.transpose();
        self.covariance = gp.mul(&gt);

        // Add motion noise to robot pose block
        for i in 0..3 {
            for j in 0..3 {
                let val = self.covariance.get(i, j) + motion_noise.get(i, j);
                self.covariance.set(i, j, val);
            }
        }
    }

    /// Update step with landmark observations
    pub fn update(&mut self, observations: &[LandmarkObservation], meas_noise: &Matrix) {
        for obs in observations {
            if let Some(&lm_idx) = self.landmark_ids.get(&obs.id) {
                // Known landmark — EKF update
                let lm_state_idx = 3 + lm_idx * 2;
                self.update_known_landmark(lm_state_idx, obs, meas_noise);
            } else {
                // New landmark — initialize
                self.add_landmark(obs);
                // Then update with the measurement
                let lm_idx = self.num_landmarks - 1;
                let lm_state_idx = 3 + lm_idx * 2;
                self.update_known_landmark(lm_state_idx, obs, meas_noise);
            }
        }
    }

    fn add_landmark(&mut self, obs: &LandmarkObservation) {
        let theta = self.state[2];
        let lm_x = self.state[0] + obs.range * (theta + obs.bearing).cos();
        let lm_y = self.state[1] + obs.range * (theta + obs.bearing).sin();

        self.state.push(lm_x);
        self.state.push(lm_y);

        let n = self.state.len();
        let mut new_cov = Matrix::zeros(n, n);
        // Copy existing covariance
        for i in 0..(n - 2) {
            for j in 0..(n - 2) {
                new_cov.set(i, j, self.covariance.get(i, j));
            }
        }
        // Initialize new landmark with large uncertainty
        new_cov.set(n - 2, n - 2, 1000.0);
        new_cov.set(n - 1, n - 1, 1000.0);
        self.covariance = new_cov;

        self.landmark_ids.insert(obs.id, self.num_landmarks);
        self.num_landmarks += 1;
    }

    fn update_known_landmark(
        &mut self,
        lm_idx: usize,
        obs: &LandmarkObservation,
        meas_noise: &Matrix,
    ) {
        let n = self.state.len();
        let rx = self.state[0];
        let ry = self.state[1];
        let rtheta = self.state[2];
        let lx = self.state[lm_idx];
        let ly = self.state[lm_idx + 1];

        let dx = lx - rx;
        let dy = ly - ry;
        let q = dx * dx + dy * dy;
        let dist = q.sqrt();

        if dist < 1e-10 {
            return;
        }

        // Expected measurement
        let expected_range = dist;
        let expected_bearing = dy.atan2(dx) - rtheta;

        // Innovation
        let range_innov = obs.range - expected_range;
        let mut bearing_innov = obs.bearing - expected_bearing;
        // Normalize angle
        while bearing_innov > PI {
            bearing_innov -= 2.0 * PI;
        }
        while bearing_innov < -PI {
            bearing_innov += 2.0 * PI;
        }

        // Jacobian H (2 x n): measurement w.r.t. full state
        let mut h = Matrix::zeros(2, n);
        h.set(0, 0, -dx / dist);
        h.set(0, 1, -dy / dist);
        h.set(0, lm_idx, dx / dist);
        h.set(0, lm_idx + 1, dy / dist);

        h.set(1, 0, dy / q);
        h.set(1, 1, -dx / q);
        h.set(1, 2, -1.0);
        h.set(1, lm_idx, -dy / q);
        h.set(1, lm_idx + 1, dx / q);

        // S = H*P*H' + R
        let hp = h.mul(&self.covariance);
        let ht = h.transpose();
        let s = hp.mul(&ht).add(meas_noise);
        let s_inv = match s.inverse() {
            Some(inv) => inv,
            None => return,
        };

        // K = P*H'*S^(-1)
        let pht = self.covariance.mul(&ht);
        let k = pht.mul(&s_inv);

        // Update state
        let innov = vec![range_innov, bearing_innov];
        let correction = k.mul_vec(&innov);
        for i in 0..n {
            self.state[i] += correction[i];
        }

        // Update covariance
        let kh = k.mul(&h);
        let i_kh = Matrix::identity(n).sub(&kh);
        self.covariance = i_kh.mul(&self.covariance);
    }

    pub fn robot_pose(&self) -> (f64, f64, f64) {
        (self.state[0], self.state[1], self.state[2])
    }

    pub fn landmark_position(&self, id: usize) -> Option<(f64, f64)> {
        self.landmark_ids.get(&id).map(|&idx| {
            let si = 3 + idx * 2;
            (self.state[si], self.state[si + 1])
        })
    }
}

// ============================================================================
// Part 16: Trajectory Generation
// ============================================================================
//
// Generating smooth, feasible trajectories is essential for robot motion.
// We implement two common methods:
//
// Trapezoidal velocity profile: accelerate, cruise, decelerate. Simple and
// widely used in industrial robots. Guarantees bounded velocity and acceleration.
//
// Cubic spline: pass through waypoints with smooth velocity. C1 continuous
// (continuous position and velocity). Natural spline boundary conditions.

/// Trapezoidal velocity profile for point-to-point motion.
/// Returns (position, velocity) at time t.
pub fn trapezoidal_profile(
    start: f64,
    end: f64,
    max_vel: f64,
    max_accel: f64,
    t: f64,
) -> (f64, f64) {
    let distance = (end - start).abs();
    let sign = if end > start { 1.0 } else { -1.0 };

    if distance < 1e-10 {
        return (start, 0.0);
    }

    // Time to reach max velocity
    let t_accel = max_vel / max_accel;
    // Distance during acceleration phase
    let d_accel = 0.5 * max_accel * t_accel * t_accel;

    let (t_accel, t_cruise, t_total) = if 2.0 * d_accel >= distance {
        // Triangle profile: never reach max velocity
        let t_accel = (distance / max_accel).sqrt();
        (t_accel, 0.0, 2.0 * t_accel)
    } else {
        // Trapezoidal: has cruise phase
        let d_cruise = distance - 2.0 * d_accel;
        let t_cruise = d_cruise / max_vel;
        (t_accel, t_cruise, 2.0 * t_accel + t_cruise)
    };

    if t <= 0.0 {
        (start, 0.0)
    } else if t >= t_total {
        (end, 0.0)
    } else if t < t_accel {
        // Acceleration phase
        let pos = start + sign * 0.5 * max_accel * t * t;
        let vel = sign * max_accel * t;
        (pos, vel)
    } else if t < t_accel + t_cruise {
        // Cruise phase
        let actual_max_vel = max_accel * t_accel;
        let pos = start + sign * (0.5 * max_accel * t_accel * t_accel + actual_max_vel * (t - t_accel));
        let vel = sign * actual_max_vel;
        (pos, vel)
    } else {
        // Deceleration phase
        let t_decel = t - t_accel - t_cruise;
        let actual_max_vel = max_accel * t_accel;
        let d_accel_phase = 0.5 * max_accel * t_accel * t_accel;
        let d_cruise_phase = actual_max_vel * t_cruise;
        let pos = start
            + sign
                * (d_accel_phase + d_cruise_phase + actual_max_vel * t_decel
                    - 0.5 * max_accel * t_decel * t_decel);
        let vel = sign * (actual_max_vel - max_accel * t_decel);
        (pos, vel)
    }
}

/// Natural cubic spline interpolation through waypoints.
/// Returns spline coefficients for evaluation.
pub struct CubicSpline {
    pub xs: Vec<f64>,     // knot x-coordinates (parameter values)
    pub ys: Vec<f64>,     // knot y-values
    pub a: Vec<f64>,      // coefficients
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub d: Vec<f64>,
}

impl CubicSpline {
    /// Construct a natural cubic spline through the given (x, y) points.
    /// x values must be strictly increasing.
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> Self {
        let n = xs.len() - 1;
        assert!(n >= 1);

        let mut h = vec![0.0; n];
        for i in 0..n {
            h[i] = xs[i + 1] - xs[i];
        }

        // Solve tridiagonal system for second derivatives
        let mut alpha = vec![0.0; n + 1];
        for i in 1..n {
            alpha[i] = 3.0 / h[i] * (ys[i + 1] - ys[i]) - 3.0 / h[i - 1] * (ys[i] - ys[i - 1]);
        }

        let mut c = vec![0.0; n + 1];
        let mut l = vec![1.0; n + 1];
        let mut mu = vec![0.0; n + 1];
        let mut z = vec![0.0; n + 1];

        for i in 1..n {
            l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        for j in (0..n).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
        }

        let mut a = ys.clone();
        let mut b = vec![0.0; n];
        let mut d = vec![0.0; n];

        for i in 0..n {
            b[i] = (ys[i + 1] - ys[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0;
            d[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
        }

        a.truncate(n);
        c.truncate(n);

        Self {
            xs,
            ys,
            a,
            b,
            c,
            d,
        }
    }

    /// Evaluate spline at parameter value t
    pub fn evaluate(&self, t: f64) -> f64 {
        let n = self.a.len();
        // Find the right segment
        let mut i = 0;
        for j in 0..n {
            if t >= self.xs[j] {
                i = j;
            }
        }
        if i >= n {
            i = n - 1;
        }

        let dx = t - self.xs[i];
        self.a[i] + self.b[i] * dx + self.c[i] * dx * dx + self.d[i] * dx * dx * dx
    }

    /// Evaluate derivative at parameter value t
    pub fn derivative(&self, t: f64) -> f64 {
        let n = self.a.len();
        let mut i = 0;
        for j in 0..n {
            if t >= self.xs[j] {
                i = j;
            }
        }
        if i >= n {
            i = n - 1;
        }

        let dx = t - self.xs[i];
        self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx * dx
    }
}

// ============================================================================
// Part 17: Sensor Models
// ============================================================================
//
// Real robots perceive the world through noisy, imperfect sensors.
// Modeling sensor noise correctly is essential for estimation algorithms.
//
// We implement noise models for common sensors:
// - Lidar: range + bearing with Gaussian noise
// - IMU: accelerometer + gyroscope with bias and noise
// - Wheel encoder: tick-based odometry with slip noise
// - GPS: position with Gaussian noise and occasional outliers

/// Simple pseudo-random Gaussian noise using Box-Muller transform
pub fn gaussian_noise(mean: f64, std_dev: f64, seed: &mut u64) -> f64 {
    let u1 = RrtPlanner::random_f64(seed).max(1e-10);
    let u2 = RrtPlanner::random_f64(seed);
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    mean + std_dev * z
}

#[derive(Debug, Clone)]
pub struct LidarReading {
    pub range: f64,
    pub bearing: f64,
}

/// Simulate a 2D lidar scan from a given pose in an occupancy grid
pub fn simulate_lidar(
    pose: &Transform2D,
    grid: &OccupancyGrid,
    num_beams: usize,
    max_range: f64,
    range_noise_std: f64,
    seed: &mut u64,
) -> Vec<LidarReading> {
    let mut readings = Vec::with_capacity(num_beams);
    let angle_step = 2.0 * PI / num_beams as f64;

    for i in 0..num_beams {
        let bearing = -PI + i as f64 * angle_step;
        let global_angle = pose.theta + bearing;

        // Ray march
        let step = grid.resolution * 0.5;
        let mut range = 0.0;
        let mut hit = false;

        while range < max_range {
            range += step;
            let x = pose.x + range * global_angle.cos();
            let y = pose.y + range * global_angle.sin();
            let (gx, gy) = grid.world_to_grid(x, y);
            if grid.is_occupied(gx, gy) {
                hit = true;
                break;
            }
        }

        if !hit {
            range = max_range;
        }

        // Add noise
        let noisy_range = (range + gaussian_noise(0.0, range_noise_std, seed)).max(0.0);
        readings.push(LidarReading {
            range: noisy_range,
            bearing,
        });
    }
    readings
}

#[derive(Debug, Clone)]
pub struct ImuReading {
    pub accel: Vec3,    // m/s² (includes gravity)
    pub gyro: Vec3,     // rad/s
}

/// Simulate IMU reading given true acceleration and angular velocity
pub fn simulate_imu(
    true_accel: &Vec3,
    true_gyro: &Vec3,
    accel_noise_std: f64,
    gyro_noise_std: f64,
    accel_bias: &Vec3,
    gyro_bias: &Vec3,
    seed: &mut u64,
) -> ImuReading {
    ImuReading {
        accel: Vec3::new(
            true_accel.x + accel_bias.x + gaussian_noise(0.0, accel_noise_std, seed),
            true_accel.y + accel_bias.y + gaussian_noise(0.0, accel_noise_std, seed),
            true_accel.z + accel_bias.z + gaussian_noise(0.0, accel_noise_std, seed),
        ),
        gyro: Vec3::new(
            true_gyro.x + gyro_bias.x + gaussian_noise(0.0, gyro_noise_std, seed),
            true_gyro.y + gyro_bias.y + gaussian_noise(0.0, gyro_noise_std, seed),
            true_gyro.z + gyro_bias.z + gaussian_noise(0.0, gyro_noise_std, seed),
        ),
    }
}

#[derive(Debug, Clone)]
pub struct EncoderReading {
    pub left_ticks: i64,
    pub right_ticks: i64,
}

/// Differential drive odometry from encoder ticks
pub fn encoder_odometry(
    reading: &EncoderReading,
    ticks_per_rev: f64,
    wheel_radius: f64,
    wheel_base: f64,
) -> (f64, f64) {
    // Returns (linear_distance, angular_change)
    let left_dist = (reading.left_ticks as f64 / ticks_per_rev) * 2.0 * PI * wheel_radius;
    let right_dist = (reading.right_ticks as f64 / ticks_per_rev) * 2.0 * PI * wheel_radius;
    let linear = (left_dist + right_dist) / 2.0;
    let angular = (right_dist - left_dist) / wheel_base;
    (linear, angular)
}

/// Simulate GPS reading with Gaussian noise
pub fn simulate_gps(
    true_x: f64,
    true_y: f64,
    noise_std: f64,
    seed: &mut u64,
) -> (f64, f64) {
    (
        true_x + gaussian_noise(0.0, noise_std, seed),
        true_y + gaussian_noise(0.0, noise_std, seed),
    )
}

// ============================================================================
// Part 18: Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // --- Vector math tests ---

    #[test]
    fn test_vec2_operations() {
        let a = Vec2::new(3.0, 4.0);
        assert!(approx_eq(a.norm(), 5.0, EPS));
        assert!(approx_eq(a.normalized().norm(), 1.0, EPS));

        let b = Vec2::new(1.0, 2.0);
        assert!(approx_eq(a.dot(&b), 11.0, EPS));
        assert!(approx_eq(a.cross(&b), 2.0, EPS)); // 3*2 - 4*1 = 2

        let rotated = Vec2::new(1.0, 0.0).rotate(PI / 2.0);
        assert!(approx_eq(rotated.x, 0.0, EPS));
        assert!(approx_eq(rotated.y, 1.0, EPS));
    }

    #[test]
    fn test_vec3_operations() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = a.cross(&b);
        assert!(approx_eq(c.x, 0.0, EPS));
        assert!(approx_eq(c.y, 0.0, EPS));
        assert!(approx_eq(c.z, 1.0, EPS));
    }

    #[test]
    fn test_matrix_operations() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.mul(&b);
        assert!(approx_eq(c.get(0, 0), 19.0, EPS));
        assert!(approx_eq(c.get(0, 1), 22.0, EPS));
        assert!(approx_eq(c.get(1, 0), 43.0, EPS));
        assert!(approx_eq(c.get(1, 1), 50.0, EPS));
    }

    #[test]
    fn test_matrix_solve() {
        // Solve: 2x + y = 5, x + 3y = 7 => x = 1.6, y = 1.8
        let a = Matrix::new(2, 2, vec![2.0, 1.0, 1.0, 3.0]);
        let b = vec![5.0, 7.0];
        let x = a.solve(&b).unwrap();
        assert!(approx_eq(x[0], 1.6, EPS));
        assert!(approx_eq(x[1], 1.8, EPS));
    }

    #[test]
    fn test_matrix_inverse() {
        let a = Matrix::new(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
        let inv = a.inverse().unwrap();
        let prod = a.mul(&inv);
        assert!(approx_eq(prod.get(0, 0), 1.0, EPS));
        assert!(approx_eq(prod.get(0, 1), 0.0, EPS));
        assert!(approx_eq(prod.get(1, 0), 0.0, EPS));
        assert!(approx_eq(prod.get(1, 1), 1.0, EPS));
    }

    // --- Quaternion tests ---

    #[test]
    fn test_quaternion_rotation() {
        // Rotate (1, 0, 0) by 90° around z-axis => (0, 1, 0)
        let q = Quaternion::from_axis_angle(&Vec3::new(0.0, 0.0, 1.0), PI / 2.0);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = q.rotate_vec(&v);
        assert!(approx_eq(rotated.x, 0.0, EPS));
        assert!(approx_eq(rotated.y, 1.0, EPS));
        assert!(approx_eq(rotated.z, 0.0, EPS));
    }

    #[test]
    fn test_quaternion_slerp() {
        let q0 = Quaternion::identity();
        let q1 = Quaternion::from_axis_angle(&Vec3::new(0.0, 0.0, 1.0), PI / 2.0);
        let q_half = q0.slerp(&q1, 0.5);
        // Should be 45° rotation around z
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = q_half.rotate_vec(&v);
        let expected_angle = PI / 4.0;
        assert!(approx_eq(rotated.x, expected_angle.cos(), 1e-4));
        assert!(approx_eq(rotated.y, expected_angle.sin(), 1e-4));
    }

    #[test]
    fn test_quaternion_to_rotation_matrix() {
        let q = Quaternion::from_axis_angle(&Vec3::new(0.0, 0.0, 1.0), PI / 2.0);
        let r = q.to_rotation_matrix();
        // Should be [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        assert!(approx_eq(r.get(0, 0), 0.0, EPS));
        assert!(approx_eq(r.get(0, 1), -1.0, EPS));
        assert!(approx_eq(r.get(1, 0), 1.0, EPS));
        assert!(approx_eq(r.get(1, 1), 0.0, EPS));
        assert!(approx_eq(r.get(2, 2), 1.0, EPS));
    }

    // --- Transform tests ---

    #[test]
    fn test_transform_2d() {
        let t1 = Transform2D::new(1.0, 0.0, PI / 2.0);
        let p = Vec2::new(1.0, 0.0);
        let transformed = t1.transform_point(&p);
        assert!(approx_eq(transformed.x, 1.0, EPS));
        assert!(approx_eq(transformed.y, 1.0, EPS));

        // Compose with identity should be same
        let t2 = Transform2D::identity();
        let composed = t1.compose(&t2);
        assert!(approx_eq(composed.x, t1.x, EPS));
        assert!(approx_eq(composed.y, t1.y, EPS));
    }

    #[test]
    fn test_transform_2d_inverse() {
        let t = Transform2D::new(2.0, 3.0, PI / 4.0);
        let t_inv = t.inverse();
        let composed = t.compose(&t_inv);
        assert!(approx_eq(composed.x, 0.0, 1e-10));
        assert!(approx_eq(composed.y, 0.0, 1e-10));
    }

    #[test]
    fn test_transform_3d() {
        let rot = Matrix::identity(3);
        let trans = Vec3::new(1.0, 2.0, 3.0);
        let t = Transform3D::from_rotation_translation(&rot, &trans);
        let p = Vec3::new(0.0, 0.0, 0.0);
        let transformed = t.transform_point(&p);
        assert!(approx_eq(transformed.x, 1.0, EPS));
        assert!(approx_eq(transformed.y, 2.0, EPS));
        assert!(approx_eq(transformed.z, 3.0, EPS));
    }

    #[test]
    fn test_transform_3d_inverse() {
        let q = Quaternion::from_axis_angle(&Vec3::new(0.0, 1.0, 0.0), PI / 3.0);
        let rot = q.to_rotation_matrix();
        let trans = Vec3::new(5.0, -3.0, 2.0);
        let t = Transform3D::from_rotation_translation(&rot, &trans);
        let t_inv = t.inverse();
        let composed = t.compose(&t_inv);
        // Should be approximately identity
        assert!(approx_eq(composed.matrix[0], 1.0, 1e-10));
        assert!(approx_eq(composed.matrix[5], 1.0, 1e-10));
        assert!(approx_eq(composed.matrix[10], 1.0, 1e-10));
        assert!(approx_eq(composed.matrix[3], 0.0, 1e-10));
        assert!(approx_eq(composed.matrix[7], 0.0, 1e-10));
        assert!(approx_eq(composed.matrix[11], 0.0, 1e-10));
    }

    // --- Forward kinematics tests ---

    #[test]
    fn test_forward_kinematics_2link() {
        // Simple 2-link planar arm with link lengths 1.0 each
        // DH parameters for planar: alpha=0, d=0, a=link_length
        let robot = SerialRobot::new(vec![
            DHParameter {
                joint_type: JointType::Revolute,
                theta: 0.0,
                d: 0.0,
                a: 1.0,
                alpha: 0.0,
            },
            DHParameter {
                joint_type: JointType::Revolute,
                theta: 0.0,
                d: 0.0,
                a: 1.0,
                alpha: 0.0,
            },
        ]);

        // Both joints at 0: end effector at (2, 0, 0)
        let t = robot.forward_kinematics(&[0.0, 0.0]);
        let pos = t.translation();
        assert!(approx_eq(pos.x, 2.0, EPS));
        assert!(approx_eq(pos.y, 0.0, EPS));

        // First joint at 90°: end effector at (0, 2, 0)
        let t = robot.forward_kinematics(&[PI / 2.0, 0.0]);
        let pos = t.translation();
        assert!(approx_eq(pos.x, 0.0, 1e-4));
        assert!(approx_eq(pos.y, 2.0, 1e-4));

        // Both joints at 90°: end effector at (-1, 1, 0)
        let t = robot.forward_kinematics(&[PI / 2.0, PI / 2.0]);
        let pos = t.translation();
        assert!(approx_eq(pos.x, -1.0, 1e-4));
        assert!(approx_eq(pos.y, 1.0, 1e-4));
    }

    // --- Inverse kinematics tests ---

    #[test]
    fn test_ik_ccd_2d() {
        let link_lengths = vec![1.0, 1.0, 1.0];
        let target = Vec2::new(2.0, 1.0);
        let initial = vec![0.0, 0.0, 0.0];
        let (angles, error) = ik_ccd_2d(&link_lengths, &target, &initial, 100, 0.01);
        assert!(error < 0.01, "CCD should converge, error={}", error);

        // Verify the solution: forward kinematics should reach target
        let end = forward_2d(&link_lengths, &angles);
        assert!(end.distance_to(&target) < 0.01);
    }

    #[test]
    fn test_ik_jacobian_transpose() {
        let robot = SerialRobot::new(vec![
            DHParameter {
                joint_type: JointType::Revolute,
                theta: 0.0,
                d: 0.0,
                a: 1.0,
                alpha: 0.0,
            },
            DHParameter {
                joint_type: JointType::Revolute,
                theta: 0.0,
                d: 0.0,
                a: 1.0,
                alpha: 0.0,
            },
        ]);
        let target = Vec3::new(1.0, 1.0, 0.0);
        let (_, error) = ik_jacobian_transpose(&robot, &target, &[0.1, 0.1], 500, 0.05);
        assert!(error < 0.05, "IK should converge, error={}", error);
    }

    // --- PID controller tests ---

    #[test]
    fn test_pid_step_response() {
        let mut pid = PidController::new(2.0, 0.5, 0.1)
            .with_output_limits(-10.0, 10.0)
            .with_integral_limits(-5.0, 5.0);

        let setpoint = 1.0;
        let dt = 0.01;
        let mut pv = 0.0;

        // Simulate a simple first-order system: pv += u * dt
        for _ in 0..1000 {
            let u = pid.update(setpoint, pv, dt);
            pv += u * dt;
        }

        // Should converge close to setpoint
        assert!(
            approx_eq(pv, setpoint, 0.05),
            "PID should converge to setpoint, pv={}",
            pv
        );
    }

    #[test]
    fn test_pid_anti_windup() {
        let mut pid = PidController::new(1.0, 10.0, 0.0)
            .with_integral_limits(-1.0, 1.0)
            .with_output_limits(-5.0, 5.0);

        let dt = 0.01;
        // Large error for many steps should not cause integral windup
        for _ in 0..1000 {
            pid.update(100.0, 0.0, dt);
        }
        // Integral should be clamped
        assert!(pid.integral <= 1.0);
        assert!(pid.integral >= -1.0);
    }

    // --- LQR tests ---

    #[test]
    fn test_lqr_double_integrator() {
        // Double integrator: x = [position, velocity], u = acceleration
        // x[k+1] = [[1, dt], [0, 1]] x[k] + [[0.5*dt², dt]]' u[k]
        let dt = 0.1;
        let a = Matrix::new(2, 2, vec![1.0, dt, 0.0, 1.0]);
        let b = Matrix::new(2, 1, vec![0.5 * dt * dt, dt]);
        let q = Matrix::new(2, 2, vec![10.0, 0.0, 0.0, 1.0]);
        let r = Matrix::new(1, 1, vec![1.0]);

        let lqr = LqrController::new(&a, &b, &q, &r);

        // Simulate: starting at x = [5, 0], should drive to origin
        let mut state = vec![5.0, 0.0];
        for _ in 0..200 {
            let u = lqr.control(&state);
            let new_pos = state[0] + dt * state[1] + 0.5 * dt * dt * u[0];
            let new_vel = state[1] + dt * u[0];
            state = vec![new_pos, new_vel];
        }
        assert!(
            state[0].abs() < 0.1,
            "LQR should drive position to 0, pos={}",
            state[0]
        );
        assert!(
            state[1].abs() < 0.1,
            "LQR should drive velocity to 0, vel={}",
            state[1]
        );
    }

    // --- MPC tests ---

    #[test]
    fn test_mpc_simple() {
        let dt = 0.1;
        let a = Matrix::new(2, 2, vec![1.0, dt, 0.0, 1.0]);
        let b = Matrix::new(2, 1, vec![0.5 * dt * dt, dt]);
        let q = Matrix::new(2, 2, vec![10.0, 0.0, 0.0, 1.0]);
        let r = Matrix::new(1, 1, vec![0.1]);

        let mpc = MpcController::new(a.clone(), b.clone(), q, r, 10, vec![-5.0], vec![5.0]);

        let mut state = vec![3.0, 0.0];
        for _ in 0..100 {
            let u = mpc.solve(&state);
            // Clamp u within limits
            let u_clamped = u[0].clamp(-5.0, 5.0);
            let new_pos = state[0] + dt * state[1] + 0.5 * dt * dt * u_clamped;
            let new_vel = state[1] + dt * u_clamped;
            state = vec![new_pos, new_vel];
        }
        assert!(
            state[0].abs() < 0.5,
            "MPC should drive state toward origin, pos={}",
            state[0]
        );
    }

    // --- A* path planning tests ---

    #[test]
    fn test_astar_simple() {
        let mut grid = OccupancyGrid::new(10, 10, 1.0);
        // Add a wall with a gap
        for y in 0..8 {
            grid.set_obstacle(5, y);
        }

        let path = astar_grid(&grid, (0, 5), (9, 5));
        assert!(path.is_some(), "A* should find a path");
        let path = path.unwrap();
        assert_eq!(path[0], (0, 5));
        assert_eq!(*path.last().unwrap(), (9, 5));

        // Verify no path goes through obstacles
        for &(x, y) in &path {
            assert!(!grid.is_occupied(x, y), "Path should not go through obstacles");
        }
    }

    #[test]
    fn test_astar_no_path() {
        let mut grid = OccupancyGrid::new(10, 10, 1.0);
        // Complete wall
        for y in 0..10 {
            grid.set_obstacle(5, y);
        }
        let path = astar_grid(&grid, (0, 5), (9, 5));
        assert!(path.is_none(), "A* should return None when no path exists");
    }

    #[test]
    fn test_astar_straight_line() {
        let grid = OccupancyGrid::new(10, 10, 1.0);
        let path = astar_grid(&grid, (0, 0), (9, 0)).unwrap();
        assert_eq!(path.len(), 10); // Straight line should be 10 cells
    }

    // --- RRT tests ---

    #[test]
    fn test_rrt_basic() {
        let grid = OccupancyGrid::new(20, 20, 0.5);
        let start = Vec2::new(1.0, 1.0);
        let goal = Vec2::new(8.0, 8.0);
        let bounds = (Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));

        let mut rrt = RrtPlanner::new(start, 0.5, 0.5, bounds);
        let mut seed = 42u64;
        let path = rrt.plan_rrt(&goal, &grid, 2000, &mut seed);
        assert!(path.is_some(), "RRT should find a path in open space");
    }

    #[test]
    fn test_rrt_star_better_than_rrt() {
        let grid = OccupancyGrid::new(20, 20, 0.5);
        let start = Vec2::new(1.0, 1.0);
        let goal = Vec2::new(8.0, 8.0);
        let bounds = (Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));

        let mut rrt = RrtPlanner::new(start, 0.5, 0.5, bounds.clone());
        let mut seed = 42u64;
        let path_rrt = rrt.plan_rrt(&goal, &grid, 2000, &mut seed);
        assert!(path_rrt.is_some());

        let mut rrt_star = RrtPlanner::new(start, 0.5, 0.5, bounds);
        let mut seed2 = 42u64;
        let path_star = rrt_star.plan_rrt_star(&goal, &grid, 2000, 2.0, &mut seed2);
        assert!(path_star.is_some());
    }

    // --- Potential field tests ---

    #[test]
    fn test_potential_field_no_obstacles() {
        let pf = PotentialField::new(1.0, 1.0, 2.0);
        let start = Vec2::new(0.0, 0.0);
        let goal = Vec2::new(5.0, 5.0);
        let path = pf.navigate(&start, &goal, &[], 0.1, 1000, 0.2);

        // Should reach goal
        let end = path.last().unwrap();
        assert!(end.distance_to(&goal) < 0.3, "Should reach goal");
    }

    #[test]
    fn test_potential_field_with_obstacle() {
        let pf = PotentialField::new(1.0, 5.0, 3.0);
        let start = Vec2::new(0.0, 0.0);
        let goal = Vec2::new(10.0, 0.0);
        let obstacles = vec![Vec2::new(5.0, 0.0)];
        let path = pf.navigate(&start, &goal, &obstacles, 0.1, 2000, 0.5);

        // Path should avoid the obstacle
        for p in &path {
            let d = p.distance_to(&obstacles[0]);
            assert!(d > 0.5, "Path should keep distance from obstacle");
        }
    }

    // --- Kalman filter tests ---

    #[test]
    fn test_kalman_filter_constant_velocity() {
        // Track a 1D object moving at constant velocity
        let dt = 0.1;
        let f = Matrix::new(2, 2, vec![1.0, dt, 0.0, 1.0]);
        let b = Matrix::zeros(2, 1);
        let h = Matrix::new(1, 2, vec![1.0, 0.0]); // observe position only
        let q = Matrix::new(2, 2, vec![0.01, 0.0, 0.0, 0.01]);
        let r = Matrix::new(1, 1, vec![1.0]); // noisy measurement

        let state = vec![0.0, 0.0];
        let cov = Matrix::new(2, 2, vec![10.0, 0.0, 0.0, 10.0]);
        let mut kf = KalmanFilter::new(state, cov, f, b, h, q, r);

        // True state: position = 5*t, velocity = 5
        let true_vel = 5.0;
        let mut seed = 123u64;

        for i in 0..50 {
            let t = (i + 1) as f64 * dt;
            let true_pos = true_vel * t;

            kf.predict(&[0.0]);
            let noisy_pos = true_pos + gaussian_noise(0.0, 1.0, &mut seed);
            kf.update(&[noisy_pos]);
        }

        // After 50 steps, should have good estimate
        let est_pos = kf.state[0];
        let est_vel = kf.state[1];
        let true_final_pos = true_vel * 50.0 * dt;

        assert!(
            (est_pos - true_final_pos).abs() < 2.0,
            "KF position estimate should be close, est={}, true={}",
            est_pos,
            true_final_pos
        );
        assert!(
            (est_vel - true_vel).abs() < 2.0,
            "KF velocity estimate should be close, est={}, true={}",
            est_vel,
            true_vel
        );
    }

    // --- EKF tests ---

    #[test]
    fn test_ekf_nonlinear_bearing() {
        // Robot at some position, measuring bearing to a fixed landmark
        let landmark = Vec2::new(5.0, 5.0);

        let state = vec![0.0, 0.0]; // robot [x, y]
        let cov = Matrix::new(2, 2, vec![10.0, 0.0, 0.0, 10.0]);
        let q = Matrix::new(2, 2, vec![0.01, 0.0, 0.0, 0.01]);
        let r = Matrix::new(1, 1, vec![0.01]);

        let mut ekf = ExtendedKalmanFilter::new(state, cov, q, r);

        // Motion model: simple constant position
        let f_func = |x: &[f64], _u: &[f64]| -> Vec<f64> { x.to_vec() };
        let f_jac = |_x: &[f64], _u: &[f64]| -> Matrix { Matrix::identity(2) };

        // Measurement: bearing to landmark
        let lm = landmark;
        let h_func = move |x: &[f64]| -> Vec<f64> {
            vec![(lm.y - x[1]).atan2(lm.x - x[0])]
        };
        let h_jac = move |x: &[f64]| -> Matrix {
            let dx = lm.x - x[0];
            let dy = lm.y - x[1];
            let d2 = dx * dx + dy * dy;
            Matrix::new(1, 2, vec![dy / d2, -dx / d2])
        };

        // True position is (1, 1), take many measurements
        let true_pos = Vec2::new(1.0, 1.0);
        let true_bearing = (landmark.y - true_pos.y).atan2(landmark.x - true_pos.x);
        let mut seed = 456u64;

        for _ in 0..100 {
            ekf.predict(&[], &f_func, &f_jac);
            let noisy_bearing = true_bearing + gaussian_noise(0.0, 0.1, &mut seed);
            ekf.update(&[noisy_bearing], &h_func, &h_jac);
        }

        // With only bearing measurements from one landmark, we can determine
        // the angle but not the exact position. The state should at least
        // move toward the correct bearing line.
        let est_bearing = (landmark.y - ekf.state[1]).atan2(landmark.x - ekf.state[0]);
        assert!(
            (est_bearing - true_bearing).abs() < 0.3,
            "EKF should estimate correct bearing"
        );
    }

    // --- Particle filter tests ---

    #[test]
    fn test_particle_filter_convergence() {
        let mut seed = 789u64;

        // Create particles spread around the true state
        let true_state = vec![5.0, 3.0];
        let mut initial_particles = Vec::new();
        for _ in 0..200 {
            initial_particles.push(vec![
                gaussian_noise(5.0, 3.0, &mut seed),
                gaussian_noise(3.0, 3.0, &mut seed),
            ]);
        }

        let mut pf = ParticleFilter::new(initial_particles);

        // Likelihood: Gaussian centered at true state
        let ts = true_state.clone();
        let likelihood = move |state: &[f64], _meas: &[f64]| -> f64 {
            let dx = state[0] - ts[0];
            let dy = state[1] - ts[1];
            (-0.5 * (dx * dx + dy * dy)).exp()
        };

        // Motion model: stay in place with small noise
        let motion_model = |state: &[f64], seed: &mut u64| -> Vec<f64> {
            vec![
                state[0] + gaussian_noise(0.0, 0.1, seed),
                state[1] + gaussian_noise(0.0, 0.1, seed),
            ]
        };

        // Run several predict-update-resample cycles
        for _ in 0..20 {
            pf.predict(&motion_model, &mut seed);
            pf.update(&[], &likelihood);
            pf.resample(&mut seed);
        }

        let estimate = pf.estimate();
        assert!(
            (estimate[0] - true_state[0]).abs() < 1.0,
            "PF x estimate should be close, est={}, true={}",
            estimate[0],
            true_state[0]
        );
        assert!(
            (estimate[1] - true_state[1]).abs() < 1.0,
            "PF y estimate should be close, est={}, true={}",
            estimate[1],
            true_state[1]
        );
    }

    // --- EKF-SLAM tests ---

    #[test]
    fn test_ekf_slam_basic() {
        let mut slam = EkfSlam::new((0.0, 0.0, 0.0));
        let motion_noise = Matrix::new(3, 3, vec![0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.001]);
        let meas_noise = Matrix::new(2, 2, vec![0.1, 0.0, 0.0, 0.01]);

        // Move forward
        slam.predict(1.0, 0.0, 1.0, &motion_noise);
        let pose = slam.robot_pose();
        assert!(approx_eq(pose.0, 1.0, 0.1));

        // Observe a landmark at range 3, bearing 0 (directly ahead)
        let obs = vec![LandmarkObservation {
            id: 0,
            range: 3.0,
            bearing: 0.0,
        }];
        slam.update(&obs, &meas_noise);

        // Landmark should be approximately at (4, 0) = robot_x + range
        let lm = slam.landmark_position(0).unwrap();
        assert!(
            (lm.0 - 4.0).abs() < 1.0,
            "Landmark x should be ~4, got {}",
            lm.0
        );
    }

    // --- Trajectory generation tests ---

    #[test]
    fn test_trapezoidal_profile() {
        let (pos, vel) = trapezoidal_profile(0.0, 10.0, 2.0, 1.0, 0.0);
        assert!(approx_eq(pos, 0.0, EPS));
        assert!(approx_eq(vel, 0.0, EPS));

        // At the end, should be at target with zero velocity
        let (pos, vel) = trapezoidal_profile(0.0, 10.0, 2.0, 1.0, 100.0);
        assert!(approx_eq(pos, 10.0, EPS));
        assert!(approx_eq(vel, 0.0, EPS));

        // In the middle, velocity should be positive
        let (_, vel) = trapezoidal_profile(0.0, 10.0, 2.0, 1.0, 3.0);
        assert!(vel > 0.0, "Velocity should be positive during motion");
    }

    #[test]
    fn test_cubic_spline() {
        // Interpolate through known points
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let spline = CubicSpline::new(xs, ys);

        // Should pass through all knot points
        assert!(approx_eq(spline.evaluate(0.0), 0.0, 1e-4));
        assert!(approx_eq(spline.evaluate(1.0), 1.0, 1e-4));
        assert!(approx_eq(spline.evaluate(2.0), 0.0, 1e-4));
        assert!(approx_eq(spline.evaluate(3.0), 1.0, 1e-4));
        assert!(approx_eq(spline.evaluate(4.0), 0.0, 1e-4));

        // Midpoint should be smooth (not equal to either neighbor)
        let mid = spline.evaluate(0.5);
        assert!(mid > 0.0 && mid < 1.0, "Midpoint should be between knots");
    }

    // --- Sensor model tests ---

    #[test]
    fn test_encoder_odometry() {
        let reading = EncoderReading {
            left_ticks: 100,
            right_ticks: 100,
        };
        let (linear, angular) = encoder_odometry(&reading, 360.0, 0.05, 0.3);
        // Both wheels same => straight line, no rotation
        assert!(angular.abs() < EPS);
        assert!(linear > 0.0);
    }

    #[test]
    fn test_encoder_turning() {
        let reading = EncoderReading {
            left_ticks: 0,
            right_ticks: 100,
        };
        let (linear, angular) = encoder_odometry(&reading, 360.0, 0.05, 0.3);
        // Only right wheel moves => turns left
        assert!(angular > 0.0);
        assert!(linear > 0.0); // Still moves forward somewhat
    }

    #[test]
    fn test_lidar_simulation() {
        let mut grid = OccupancyGrid::new(20, 20, 0.5);
        // Place a wall at x=5
        for y in 0..20 {
            grid.set_obstacle(10, y); // grid x=10 => world x=5.25
        }

        let pose = Transform2D::new(2.5, 5.0, 0.0);
        let mut seed = 42u64;
        let readings = simulate_lidar(&pose, &grid, 36, 10.0, 0.01, &mut seed);

        assert_eq!(readings.len(), 36);
        // At least some readings should detect the wall (range < max)
        let close_readings: Vec<_> = readings.iter().filter(|r| r.range < 5.0).collect();
        assert!(
            !close_readings.is_empty(),
            "Lidar should detect the wall"
        );
    }

    #[test]
    fn test_gaussian_noise_distribution() {
        let mut seed = 999u64;
        let n = 10000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let v = gaussian_noise(5.0, 2.0, &mut seed);
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n as f64;
        let variance = sum_sq / n as f64 - mean * mean;
        assert!(
            (mean - 5.0).abs() < 0.2,
            "Mean should be ~5.0, got {}",
            mean
        );
        assert!(
            (variance - 4.0).abs() < 0.5,
            "Variance should be ~4.0, got {}",
            variance
        );
    }

    // --- Integration test: full robotics pipeline ---

    #[test]
    fn test_full_pipeline_navigate_and_estimate() {
        // Create environment
        let mut grid = OccupancyGrid::new(20, 20, 0.5);
        for y in 4..16 {
            grid.set_obstacle(10, y); // Wall in the middle with gaps at top/bottom
        }

        // Plan a path with A*
        let start = (2, 10);
        let goal = (18, 10);
        let path = astar_grid(&grid, start, goal);
        assert!(path.is_some(), "Should find path around wall");
        let path = path.unwrap();
        assert!(path.len() > 2, "Path should have multiple waypoints");

        // Use Kalman filter to track estimated position along path
        let dt = 0.1;
        let f = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let b = Matrix::new(2, 2, vec![dt, 0.0, 0.0, dt]);
        let h = Matrix::identity(2);
        let q = Matrix::new(2, 2, vec![0.01, 0.0, 0.0, 0.01]);
        let r = Matrix::new(2, 2, vec![0.1, 0.0, 0.0, 0.1]);

        let state = vec![
            (start.0 as f64 + 0.5) * grid.resolution,
            (start.1 as f64 + 0.5) * grid.resolution,
        ];
        let cov = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let mut kf = KalmanFilter::new(state, cov, f, b, h, q, r);

        let mut seed = 42u64;

        // Follow the first few waypoints
        for i in 1..path.len().min(5) {
            let (wx, wy) = grid.grid_to_world(path[i].0, path[i].1);
            let (cx, cy) = (kf.state[0], kf.state[1]);
            let vx = (wx - cx) / dt;
            let vy = (wy - cy) / dt;

            kf.predict(&[vx, vy]);
            let noisy_x = wx + gaussian_noise(0.0, 0.3, &mut seed);
            let noisy_y = wy + gaussian_noise(0.0, 0.3, &mut seed);
            kf.update(&[noisy_x, noisy_y]);
        }

        // KF should give reasonable position estimate
        assert!(kf.state[0] > 0.0 && kf.state[0] < 10.0);
        assert!(kf.state[1] > 0.0 && kf.state[1] < 10.0);
    }
}
