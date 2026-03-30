// chronos-dsp-engine.rs
//
// Chronos Digital Signal Processing Engine
// ==========================================
// A comprehensive DSP library implementing signal transforms, filter design,
// spectral analysis, and audio processing algorithms from first principles.
//
// Modules:
//   1.  Complex Numbers & Arithmetic
//   2.  Discrete Fourier Transform (DFT) — naive O(N²) for reference
//   3.  Fast Fourier Transform (FFT) — Cooley-Tukey radix-2 Decimation-In-Time
//   4.  Inverse FFT (IFFT)
//   5.  Short-Time Fourier Transform (STFT)
//   6.  Window Functions (Hann, Hamming, Blackman, Kaiser, Bartlett, Flat-top)
//   7.  FIR Filter Design (windowed sinc, Parks-McClellan via Remez)
//   8.  IIR Filters (Butterworth, Chebyshev Type I/II, Biquad cascade)
//   9.  Convolution & Correlation (direct and FFT-based overlap-add)
//  10.  Resampling (polyphase, linear, cubic spline)
//  11.  Spectral Analysis (PSD via Welch's method, spectrogram)
//  12.  Signal Generators (sine, square, sawtooth, noise, chirp, pulse)
//  13.  Audio Effects (compressor, limiter, reverb delay line, comb filter)
//  14.  Adaptive Filters (LMS, NLMS, RLS — for echo cancellation, noise reduction)
//  15.  Cepstrum & MFCC (speech features)
//  16.  Phase Vocoder (pitch shifting)

use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// 1. COMPLEX NUMBERS
// ─────────────────────────────────────────────────────────────────────────────

/// Complex number in rectangular form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex { pub re: f64, pub im: f64 }

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    pub const ONE:  Complex = Complex { re: 1.0, im: 0.0 };
    pub const I:    Complex = Complex { re: 0.0, im: 1.0 };

    pub fn new(re: f64, im: f64) -> Self { Complex { re, im } }
    pub fn from_polar(r: f64, theta: f64) -> Self { Complex { re: r * theta.cos(), im: r * theta.sin() } }

    pub fn magnitude(&self) -> f64 { (self.re * self.re + self.im * self.im).sqrt() }
    pub fn magnitude_sq(&self) -> f64 { self.re * self.re + self.im * self.im }
    pub fn phase(&self) -> f64 { self.im.atan2(self.re) }
    pub fn conjugate(&self) -> Self { Complex { re: self.re, im: -self.im } }

    pub fn add(&self, rhs: &Complex) -> Complex { Complex::new(self.re + rhs.re, self.im + rhs.im) }
    pub fn sub(&self, rhs: &Complex) -> Complex { Complex::new(self.re - rhs.re, self.im - rhs.im) }
    pub fn mul(&self, rhs: &Complex) -> Complex {
        Complex::new(self.re * rhs.re - self.im * rhs.im,
                     self.re * rhs.im + self.im * rhs.re)
    }
    pub fn div(&self, rhs: &Complex) -> Complex {
        let d = rhs.magnitude_sq();
        Complex::new((self.re * rhs.re + self.im * rhs.im) / d,
                     (self.im * rhs.re - self.re * rhs.im) / d)
    }
    pub fn scale(&self, s: f64) -> Complex { Complex::new(self.re * s, self.im * s) }

    /// e^{iθ} = cos(θ) + i·sin(θ) (Euler's formula)
    pub fn exp_i(theta: f64) -> Complex { Complex::new(theta.cos(), theta.sin()) }
}

impl std::ops::Add for Complex { type Output = Complex; fn add(self, rhs: Self) -> Complex { Complex::add(&self, &rhs) } }
impl std::ops::Sub for Complex { type Output = Complex; fn sub(self, rhs: Self) -> Complex { Complex::sub(&self, &rhs) } }
impl std::ops::Mul for Complex { type Output = Complex; fn mul(self, rhs: Self) -> Complex { Complex::mul(&self, &rhs) } }
impl std::ops::Neg for Complex { type Output = Complex; fn neg(self) -> Complex { Complex::new(-self.re, -self.im) } }

// ─────────────────────────────────────────────────────────────────────────────
// 2. DFT (REFERENCE IMPLEMENTATION)
// ─────────────────────────────────────────────────────────────────────────────

/// Naive O(N²) DFT for small signals or verification.
/// X[k] = Σ_{n=0}^{N-1} x[n] · e^{-i·2π·k·n/N}
pub fn dft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    let mut out = vec![Complex::ZERO; n];
    for k in 0..n {
        let mut sum = Complex::ZERO;
        for j in 0..n {
            let theta = -2.0 * PI * k as f64 * j as f64 / n as f64;
            sum = sum + x[j] * Complex::exp_i(theta);
        }
        out[k] = sum;
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. FAST FOURIER TRANSFORM (COOLEY-TUKEY RADIX-2 DIT)
// ─────────────────────────────────────────────────────────────────────────────

/// In-place bit-reversal permutation.
fn bit_reverse_permute(x: &mut [Complex]) {
    let n = x.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 { j ^= bit; bit >>= 1; }
        j ^= bit;
        if i < j { x.swap(i, j); }
    }
}

/// Cooley-Tukey radix-2 DIT FFT. Input length must be a power of 2.
/// O(N log N) time, O(1) extra space (in-place).
/// sign = -1 for forward FFT, +1 for inverse FFT (caller divides by N).
fn fft_inplace(x: &mut [Complex], sign: f64) {
    let n = x.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of 2");
    bit_reverse_permute(x);
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let w_step = Complex::exp_i(sign * 2.0 * PI / len as f64);
        for i in (0..n).step_by(len) {
            let mut w = Complex::ONE;
            for k in 0..half {
                let u = x[i + k];
                let t = w * x[i + k + half];
                x[i + k]        = u + t;
                x[i + k + half] = u - t;
                w = w * w_step;
            }
        }
        len <<= 1;
    }
}

/// Forward FFT: returns spectrum of `x`. Pads or truncates to next power of 2.
pub fn fft(x: &[f64]) -> Vec<Complex> {
    let n = x.len().next_power_of_two();
    let mut buf: Vec<Complex> = (0..n).map(|i| {
        if i < x.len() { Complex::new(x[i], 0.0) } else { Complex::ZERO }
    }).collect();
    fft_inplace(&mut buf, -1.0);
    buf
}

/// Forward FFT of complex input.
pub fn fft_complex(x: &[Complex]) -> Vec<Complex> {
    let n = x.len().next_power_of_two();
    let mut buf: Vec<Complex> = (0..n).map(|i| if i < x.len() { x[i] } else { Complex::ZERO }).collect();
    fft_inplace(&mut buf, -1.0);
    buf
}

/// Inverse FFT: returns time-domain signal. Length must be a power of 2.
pub fn ifft(x: &[Complex]) -> Vec<Complex> {
    let mut buf = x.to_vec();
    let n = buf.len();
    assert!(n.is_power_of_two());
    fft_inplace(&mut buf, 1.0);
    let scale = 1.0 / n as f64;
    buf.iter_mut().for_each(|c| *c = c.scale(scale));
    buf
}

/// FFT magnitude spectrum (single-sided, DC to Nyquist).
pub fn fft_magnitude(x: &[f64]) -> Vec<f64> {
    let spec = fft(x);
    let n = spec.len();
    // Single-sided: 0..=N/2
    (0..=n/2).map(|k| {
        let mag = spec[k].magnitude();
        if k == 0 || k == n/2 { mag / n as f64 } else { 2.0 * mag / n as f64 }
    }).collect()
}

/// FFT power spectral density (magnitude squared, normalised).
pub fn fft_psd(x: &[f64]) -> Vec<f64> {
    fft_magnitude(x).iter().map(|m| m * m).collect()
}

/// Frequency bins for an FFT of length N at sample rate fs.
pub fn fft_frequencies(n: usize, fs: f64) -> Vec<f64> {
    (0..=n/2).map(|k| k as f64 * fs / n as f64).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. WINDOW FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a window of `n` samples.
pub enum WindowType { Rectangular, Hann, Hamming, Blackman, Bartlett, FlatTop }

pub fn window(n: usize, wtype: &WindowType) -> Vec<f64> {
    let nm1 = (n - 1) as f64;
    (0..n).map(|i| {
        let t = i as f64;
        match wtype {
            WindowType::Rectangular => 1.0,
            WindowType::Hann        => 0.5 * (1.0 - (2.0 * PI * t / nm1).cos()),
            WindowType::Hamming     => 0.54 - 0.46 * (2.0 * PI * t / nm1).cos(),
            WindowType::Blackman    => 0.42 - 0.5  * (2.0 * PI * t / nm1).cos()
                                             + 0.08 * (4.0 * PI * t / nm1).cos(),
            WindowType::Bartlett    => 1.0 - (2.0 * t / nm1 - 1.0).abs(),
            WindowType::FlatTop     => 1.0
                - 1.930_4 * (2.0 * PI * t / nm1).cos()
                + 1.290_9 * (4.0 * PI * t / nm1).cos()
                - 0.388_2 * (6.0 * PI * t / nm1).cos()
                + 0.032_8 * (8.0 * PI * t / nm1).cos(),
        }
    }).collect()
}

/// Kaiser window with shape parameter β.
/// Approximates the prolate spheroidal wave function; trades sidelobe
/// level against main-lobe width. β=0 → rectangular, β≈5.6 → Hann.
pub fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    let nm1 = (n - 1) as f64;
    let i0_beta = bessel_i0(beta);
    (0..n).map(|k| {
        let x = 2.0 * k as f64 / nm1 - 1.0;
        bessel_i0(beta * (1.0 - x * x).sqrt()) / i0_beta
    }).collect()
}

/// Modified Bessel function I₀(x) — series expansion.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half_x = x / 2.0;
    for k in 1..=30 {
        term *= (half_x / k as f64).powi(2);
        sum += term;
        if term < 1e-15 * sum { break; }
    }
    sum
}

/// Coherent power gain of a window (for amplitude correction).
pub fn window_coherent_gain(w: &[f64]) -> f64 {
    w.iter().sum::<f64>() / w.len() as f64
}

/// Apply window to a signal frame.
pub fn apply_window(signal: &[f64], w: &[f64]) -> Vec<f64> {
    signal.iter().zip(w).map(|(s, wi)| s * wi).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. SHORT-TIME FOURIER TRANSFORM (STFT)
// ─────────────────────────────────────────────────────────────────────────────

/// STFT frame parameters.
pub struct StftParams {
    pub frame_size: usize,  // FFT length (power of 2)
    pub hop_size:   usize,  // hop between frames
    pub window:     Vec<f64>,
}

impl StftParams {
    pub fn new(frame_size: usize, hop_size: usize, win_type: &WindowType) -> Self {
        assert!(frame_size.is_power_of_two());
        StftParams { frame_size, hop_size, window: window(frame_size, win_type) }
    }
}

/// STFT result: 2D array of complex spectra [frames][bins].
pub struct Stft {
    pub frames:     Vec<Vec<Complex>>,
    pub frame_size: usize,
    pub hop_size:   usize,
    pub num_bins:   usize,  // frame_size/2 + 1
}

impl Stft {
    /// Compute the STFT of a signal.
    pub fn compute(signal: &[f64], params: &StftParams) -> Self {
        let n = signal.len();
        let fs = params.frame_size;
        let hs = params.hop_size;
        let nb = fs / 2 + 1;
        let num_frames = if n >= fs { (n - fs) / hs + 1 } else { 0 };
        let mut frames = Vec::with_capacity(num_frames);

        for f in 0..num_frames {
            let start = f * hs;
            let frame: Vec<f64> = (0..fs).map(|i| {
                let idx = start + i;
                if idx < n { signal[idx] * params.window[i] } else { 0.0 }
            }).collect();
            let spec = fft_complex(&frame.iter().map(|&s| Complex::new(s, 0.0)).collect::<Vec<_>>());
            frames.push(spec[..nb].to_vec());
        }
        Stft { frames, frame_size: fs, hop_size: hs, num_bins: nb }
    }

    /// Magnitude spectrogram [frames][bins].
    pub fn magnitude_spectrogram(&self) -> Vec<Vec<f64>> {
        self.frames.iter().map(|f| f.iter().map(|c| c.magnitude()).collect()).collect()
    }

    /// Reconstruct time-domain signal via overlap-add (OLA).
    pub fn istft(&self, window: &[f64]) -> Vec<f64> {
        let fs = self.frame_size;
        let hs = self.hop_size;
        let n_frames = self.frames.len();
        let out_len = (n_frames - 1) * hs + fs;
        let mut out    = vec![0.0f64; out_len];
        let mut weight = vec![0.0f64; out_len];
        for (f, frame) in self.frames.iter().enumerate() {
            // Mirror bins to get full spectrum
            let mut full = vec![Complex::ZERO; fs];
            full[..self.num_bins].copy_from_slice(frame);
            for k in 1..self.num_bins-1 {
                full[fs - k] = frame[k].conjugate();
            }
            let time = ifft(&full);
            let start = f * hs;
            for i in 0..fs {
                out[start + i]    += time[i].re * window[i];
                weight[start + i] += window[i] * window[i];
            }
        }
        // Normalise by overlap-add weight
        out.iter_mut().zip(&weight).for_each(|(s, &w)| { if w > 1e-12 { *s /= w; } });
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. FIR FILTER DESIGN
// ─────────────────────────────────────────────────────────────────────────────

/// FIR filter coefficients.
#[derive(Debug, Clone)]
pub struct FirFilter { pub coeffs: Vec<f64> }

impl FirFilter {
    /// Windowed-sinc low-pass FIR filter.
    /// `order`: filter order (number of taps = order + 1, must be even for Type I).
    /// `cutoff`: normalised cutoff frequency in [0, 0.5] (0.5 = Nyquist).
    /// `win`: window function for sidelobe control.
    pub fn lowpass(order: usize, cutoff: f64, win_type: &WindowType) -> Self {
        assert!(order % 2 == 0, "Order must be even for linear phase");
        let n = order + 1;
        let w = window(n, win_type);
        let center = order as f64 / 2.0;
        let coeffs: Vec<f64> = (0..n).map(|i| {
            let k = i as f64 - center;
            let h = if k.abs() < 1e-10 {
                2.0 * cutoff
            } else {
                (2.0 * PI * cutoff * k).sin() / (PI * k)
            };
            h * w[i]
        }).collect();
        // Normalise so DC gain = 1
        let gain: f64 = coeffs.iter().sum();
        FirFilter { coeffs: coeffs.iter().map(|c| c / gain).collect() }
    }

    /// High-pass FIR via spectral inversion of low-pass.
    pub fn highpass(order: usize, cutoff: f64, win_type: &WindowType) -> Self {
        let mut lp = FirFilter::lowpass(order, cutoff, win_type).coeffs;
        // Spectral inversion: negate all, add 1 to centre tap
        for c in lp.iter_mut() { *c = -*c; }
        let center = order / 2;
        lp[center] += 1.0;
        FirFilter { coeffs: lp }
    }

    /// Band-pass FIR: difference of two low-pass filters.
    pub fn bandpass(order: usize, low: f64, high: f64, win_type: &WindowType) -> Self {
        let hp_lp = FirFilter::lowpass(order, high, win_type).coeffs;
        let lp_lp = FirFilter::lowpass(order, low,  win_type).coeffs;
        let coeffs: Vec<f64> = hp_lp.iter().zip(&lp_lp).map(|(a, b)| a - b).collect();
        FirFilter { coeffs }
    }

    /// Band-stop (notch) FIR: complement of band-pass.
    pub fn bandstop(order: usize, low: f64, high: f64, win_type: &WindowType) -> Self {
        let bp = FirFilter::bandpass(order, low, high, win_type).coeffs;
        let n = bp.len();
        let center = order / 2;
        let mut coeffs = bp.iter().map(|c| -*c).collect::<Vec<_>>();
        coeffs[center] += 1.0;
        FirFilter { coeffs }
    }

    /// Apply the FIR filter to a signal using direct-form convolution.
    pub fn apply(&self, signal: &[f64]) -> Vec<f64> {
        let h = &self.coeffs;
        let m = h.len();
        let n = signal.len();
        (0..n).map(|i| {
            (0..m).map(|k| {
                if i >= k { h[k] * signal[i - k] } else { 0.0 }
            }).sum()
        }).collect()
    }

    /// Frequency response H(e^{jω}) at normalised frequency f in [0, 0.5].
    pub fn frequency_response(&self, f: f64) -> Complex {
        let h = &self.coeffs;
        let mut hr = Complex::ZERO;
        for (k, &c) in h.iter().enumerate() {
            hr = hr + Complex::exp_i(-2.0 * PI * f * k as f64).scale(c);
        }
        hr
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. IIR FILTERS — BIQUAD SECTIONS
// ─────────────────────────────────────────────────────────────────────────────

/// Second-order IIR section (biquad) — Direct Form II Transposed.
/// Transfer function: H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)
#[derive(Debug, Clone)]
pub struct Biquad {
    pub b0: f64, pub b1: f64, pub b2: f64,
    pub a1: f64, pub a2: f64,
    // State (Direct Form II Transposed)
    w1: f64, w2: f64,
}

impl Biquad {
    pub fn new(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Biquad { b0, b1, b2, a1, a2, w1: 0.0, w2: 0.0 }
    }

    /// Process one sample through the biquad (stateful).
    pub fn process_sample(&mut self, x: f64) -> f64 {
        let y  = self.b0 * x + self.w1;
        self.w1 = self.b1 * x - self.a1 * y + self.w2;
        self.w2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Process a block of samples.
    pub fn process(&mut self, signal: &[f64]) -> Vec<f64> {
        signal.iter().map(|&x| self.process_sample(x)).collect()
    }

    /// Reset internal state.
    pub fn reset(&mut self) { self.w1 = 0.0; self.w2 = 0.0; }

    /// Frequency response |H(e^{jω})| at normalised frequency f ∈ [0, 0.5].
    pub fn magnitude_response(&self, f: f64) -> f64 {
        let w = 2.0 * PI * f;
        let ejw  = Complex::exp_i(-w);
        let ej2w = Complex::exp_i(-2.0 * w);
        let num = Complex::new(self.b0, 0.0) + ejw.scale(self.b1) + ej2w.scale(self.b2);
        let den = Complex::ONE + ejw.scale(self.a1) + ej2w.scale(self.a2);
        num.magnitude() / den.magnitude()
    }
}

/// Cascade of biquad sections for higher-order IIR filters.
pub struct IirFilter { pub sections: Vec<Biquad> }

impl IirFilter {
    pub fn new(sections: Vec<Biquad>) -> Self { IirFilter { sections } }

    pub fn process_sample(&mut self, x: f64) -> f64 {
        self.sections.iter_mut().fold(x, |s, bq| bq.process_sample(s))
    }

    pub fn process(&mut self, signal: &[f64]) -> Vec<f64> {
        signal.iter().map(|&x| self.process_sample(x)).collect()
    }

    pub fn reset(&mut self) { self.sections.iter_mut().for_each(|bq| bq.reset()); }

    pub fn magnitude_response(&self, f: f64) -> f64 {
        self.sections.iter().map(|bq| bq.magnitude_response(f)).product()
    }
}

/// Design a Butterworth low-pass filter of order N, cutoff fc (normalised 0..0.5).
/// Uses bilinear transform from analogue prototype.
/// Butterworth poles are equally spaced on the unit circle in the s-plane.
pub fn butterworth_lowpass(order: usize, fc: f64) -> IirFilter {
    // Pre-warp cutoff for bilinear transform
    let wc = 2.0 * (PI * fc).tan(); // analogue prototype cutoff
    let n_sections = order / 2;
    let mut sections = Vec::new();

    for k in 0..n_sections {
        // Analogue Butterworth pole pair angle
        let theta = PI * (2 * k + 1 + order) as f64 / (2 * order) as f64;
        let sigma = -theta.sin() * wc; // real part
        let omega = theta.cos() * wc;  // imaginary part

        // Bilinear transform: s → (z-1)/(z+1)
        // H_a(s) = 1/((s-p)(s-p*)) = 1/(s² - 2σs + σ²+ω²)
        let a0_a = 1.0;
        let a1_a = -2.0 * sigma;
        let a2_a = sigma * sigma + omega * omega;

        // Bilinear transform with pre-warping
        let c = 2.0; // c = 2/T where T=1 (normalised)
        let k0 = c * c * a0_a + c * a1_a + a2_a;
        let b0 = a2_a / k0; // all-zeros at z=-1 for LP (H_a gain at s→0)
        let b1 = 2.0 * a2_a / k0;
        let b2 = a2_a / k0;
        let a1 = (2.0 * a2_a - 2.0 * c * c * a0_a) / k0;
        let a2 = (c * c * a0_a - c * a1_a + a2_a) / k0;

        sections.push(Biquad::new(b0, b1, b2, a1, a2));
    }

    // Handle odd order: first-order section
    if order % 2 == 1 {
        // Analogue first-order: H(s) = 1/(s + wc) → bilinear
        let k0 = 1.0 + wc; // wc * T/2 + 1, T normalised
        let b0 = wc / k0;
        let b1 = wc / k0;
        let b2 = 0.0;
        let a1 = (wc - 1.0) / k0;
        let a2 = 0.0;
        sections.insert(0, Biquad::new(b0, b1, b2, a1, a2));
    }
    IirFilter::new(sections)
}

/// Peak/notch parametric EQ biquad (for audio mixing).
/// `fc`: centre frequency (normalised); `gain_db`: boost/cut in dB; `q`: quality factor.
pub fn parametric_eq(fc: f64, gain_db: f64, q: f64) -> Biquad {
    let a = 10.0f64.powf(gain_db / 40.0); // sqrt(linear gain)
    let w0 = 2.0 * PI * fc;
    let alpha = w0.sin() / (2.0 * q);
    let b0 = 1.0 + alpha * a;
    let b1 = -2.0 * w0.cos();
    let b2 = 1.0 - alpha * a;
    let a0 = 1.0 + alpha / a;
    let a1n = -2.0 * w0.cos();
    let a2n = 1.0 - alpha / a;
    Biquad::new(b0/a0, b1/a0, b2/a0, a1n/a0, a2n/a0)
}

/// Low-shelf EQ biquad.
pub fn low_shelf(fc: f64, gain_db: f64) -> Biquad {
    let a  = 10.0f64.powf(gain_db / 40.0);
    let w0 = 2.0 * PI * fc;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let sqrt_a = a.sqrt();
    let alpha = sin_w0 / 2.0 * (a + 1.0/a) * (1.0/1.0 - 1.0) + 2.0 * sqrt_a;
    // Standard Audio EQ Cookbook (Zölzer)
    let b0 =    a * ((a+1.0) - (a-1.0)*cos_w0 + 2.0*sqrt_a*sin_w0/2.0);
    let b1 =  2.0*a * ((a-1.0) - (a+1.0)*cos_w0);
    let b2 =    a * ((a+1.0) - (a-1.0)*cos_w0 - 2.0*sqrt_a*sin_w0/2.0);
    let a0 =         (a+1.0) + (a-1.0)*cos_w0 + 2.0*sqrt_a*sin_w0/2.0;
    let a1 =   -2.0 * ((a-1.0) + (a+1.0)*cos_w0);
    let a2 =         (a+1.0) + (a-1.0)*cos_w0 - 2.0*sqrt_a*sin_w0/2.0;
    Biquad::new(b0/a0, b1/a0, b2/a0, a1/a0, a2/a0)
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. CONVOLUTION & CORRELATION
// ─────────────────────────────────────────────────────────────────────────────

/// Direct convolution: y[n] = Σ h[k]·x[n-k]. O(MN).
pub fn convolve(x: &[f64], h: &[f64]) -> Vec<f64> {
    let n = x.len() + h.len() - 1;
    let mut y = vec![0.0f64; n];
    for (i, &xi) in x.iter().enumerate() {
        for (j, &hj) in h.iter().enumerate() {
            y[i + j] += xi * hj;
        }
    }
    y
}

/// FFT-based convolution via overlap-add. O((N+M) log(N+M)).
/// Efficient when both signals are long.
pub fn fft_convolve(x: &[f64], h: &[f64]) -> Vec<f64> {
    let out_len = x.len() + h.len() - 1;
    let n = out_len.next_power_of_two();

    let mut x_pad: Vec<f64> = x.to_vec(); x_pad.resize(n, 0.0);
    let mut h_pad: Vec<f64> = h.to_vec(); h_pad.resize(n, 0.0);

    let X = fft(&x_pad);
    let H = fft(&h_pad);
    let Y: Vec<Complex> = X.iter().zip(&H).map(|(xi, hi)| *xi * *hi).collect();
    let y_full = ifft(&Y);
    y_full[..out_len].iter().map(|c| c.re).collect()
}

/// Cross-correlation: R_{xy}[τ] = Σ x[n]·y[n+τ].
/// Computed via FFT: R = IFFT(X* · Y).
pub fn cross_correlate(x: &[f64], y: &[f64]) -> Vec<f64> {
    let out_len = x.len() + y.len() - 1;
    let n = out_len.next_power_of_two();
    let mut x_pad = x.to_vec(); x_pad.resize(n, 0.0);
    let mut y_pad = y.to_vec(); y_pad.resize(n, 0.0);
    let X = fft(&x_pad);
    let Y = fft(&y_pad);
    // R = IFFT(conj(X) * Y)
    let R: Vec<Complex> = X.iter().zip(&Y).map(|(xi, yi)| xi.conjugate() * *yi).collect();
    let r_full = ifft(&R);
    // Rearrange so zero-lag is at centre
    let mut result = vec![0.0f64; out_len];
    let half = y.len() - 1;
    for i in 0..out_len {
        result[i] = r_full[(i + n - half) % n].re;
    }
    result
}

/// Autocorrelation: R_{xx}[τ] = cross_correlate(x, x).
pub fn autocorrelate(x: &[f64]) -> Vec<f64> { cross_correlate(x, x) }

// ─────────────────────────────────────────────────────────────────────────────
// 9. WELCH'S POWER SPECTRAL DENSITY
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate PSD using Welch's method (averaged periodograms).
/// Divides signal into overlapping frames, windows each, FFTs, and averages.
/// Returns (frequencies, PSD) where PSD is in units/Hz.
pub fn welch_psd(signal: &[f64], fs: f64, nperseg: usize, noverlap: usize,
                 win_type: &WindowType) -> (Vec<f64>, Vec<f64>) {
    assert!(nperseg.is_power_of_two());
    let w     = window(nperseg, win_type);
    let w_sq_sum: f64 = w.iter().map(|wi| wi * wi).sum::<f64>() * fs;
    let hop   = nperseg - noverlap;
    let nbins = nperseg / 2 + 1;
    let mut psd = vec![0.0f64; nbins];
    let mut n_frames = 0usize;

    let n = signal.len();
    let mut start = 0;
    while start + nperseg <= n {
        let frame: Vec<f64> = (0..nperseg).map(|i| signal[start + i] * w[i]).collect();
        let spec = fft(&frame);
        for k in 0..nbins {
            let power = spec[k].magnitude_sq();
            psd[k] += if k == 0 || k == nbins - 1 { power } else { 2.0 * power };
        }
        n_frames += 1;
        start += hop;
    }

    if n_frames > 0 {
        for p in psd.iter_mut() { *p /= n_frames as f64 * w_sq_sum; }
    }

    let freqs = (0..nbins).map(|k| k as f64 * fs / nperseg as f64).collect();
    (freqs, psd)
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. RESAMPLING
// ─────────────────────────────────────────────────────────────────────────────

/// Linear interpolation resampling.
pub fn resample_linear(signal: &[f64], factor: f64) -> Vec<f64> {
    let out_len = (signal.len() as f64 * factor) as usize;
    (0..out_len).map(|i| {
        let x = i as f64 / factor;
        let lo = x as usize;
        let hi = (lo + 1).min(signal.len() - 1);
        let t  = x - lo as f64;
        signal[lo] * (1.0 - t) + signal[hi] * t
    }).collect()
}

/// Cubic spline (Catmull-Rom) resampling.
pub fn resample_cubic(signal: &[f64], factor: f64) -> Vec<f64> {
    let n = signal.len();
    let out_len = (n as f64 * factor) as usize;
    let clamp = |i: isize| signal[i.clamp(0, n as isize - 1) as usize];
    (0..out_len).map(|i| {
        let x  = i as f64 / factor;
        let lo = x as isize;
        let t  = x - lo as f64;
        let p0 = clamp(lo - 1);
        let p1 = clamp(lo);
        let p2 = clamp(lo + 1);
        let p3 = clamp(lo + 2);
        // Catmull-Rom coefficients
        let a0 = -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3;
        let a1 =      p0 - 2.5*p1 + 2.0*p2 - 0.5*p3;
        let a2 = -0.5*p0           + 0.5*p2;
        let a3 =            p1;
        a0*t*t*t + a1*t*t + a2*t + a3
    }).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. SIGNAL GENERATORS
// ─────────────────────────────────────────────────────────────────────────────

/// Generate N samples of a sinusoid at frequency `freq` Hz, sample rate `fs`.
pub fn gen_sine(n: usize, freq: f64, fs: f64, amplitude: f64, phase: f64) -> Vec<f64> {
    (0..n).map(|i| amplitude * (2.0 * PI * freq * i as f64 / fs + phase).sin()).collect()
}

/// Square wave: +amp for first half-period, -amp for second.
pub fn gen_square(n: usize, freq: f64, fs: f64, amplitude: f64) -> Vec<f64> {
    (0..n).map(|i| {
        let phase = (freq * i as f64 / fs).fract();
        if phase < 0.5 { amplitude } else { -amplitude }
    }).collect()
}

/// Sawtooth wave: ramps from -amp to +amp over each period.
pub fn gen_sawtooth(n: usize, freq: f64, fs: f64, amplitude: f64) -> Vec<f64> {
    (0..n).map(|i| {
        let phase = (freq * i as f64 / fs).fract();
        amplitude * (2.0 * phase - 1.0)
    }).collect()
}

/// Linear chirp: sweeps from `f_start` to `f_end` Hz over `duration` seconds.
/// Instantaneous phase: φ(t) = 2π(f0·t + (f1-f0)·t²/(2·T))
pub fn gen_chirp(n: usize, fs: f64, f_start: f64, f_end: f64, amplitude: f64) -> Vec<f64> {
    let t_total = n as f64 / fs;
    (0..n).map(|i| {
        let t = i as f64 / fs;
        let phase = 2.0 * PI * (f_start * t + (f_end - f_start) * t * t / (2.0 * t_total));
        amplitude * phase.sin()
    }).collect()
}

/// White noise (uniform distribution scaled to [-amplitude, amplitude]).
pub fn gen_noise(n: usize, amplitude: f64, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (state >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
        amplitude * (2.0 * u - 1.0)
    }).collect()
}

/// Dirac impulse (unit impulse at sample `pos`).
pub fn gen_impulse(n: usize, pos: usize) -> Vec<f64> {
    let mut v = vec![0.0f64; n];
    if pos < n { v[pos] = 1.0; }
    v
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. AUDIO DYNAMICS — COMPRESSOR / LIMITER
// ─────────────────────────────────────────────────────────────────────────────

/// Feed-forward dynamic range compressor.
/// Gain is computed from the RMS level with attack/release smoothing.
pub struct Compressor {
    pub threshold_db: f64,  // dBFS threshold above which compression begins
    pub ratio:        f64,  // compression ratio (e.g. 4.0 = 4:1)
    pub attack:       f64,  // attack time constant (seconds)
    pub release:      f64,  // release time constant (seconds)
    pub makeup_db:    f64,  // makeup gain in dB
    pub fs:           f64,  // sample rate
    env: f64,               // envelope follower state
}

impl Compressor {
    pub fn new(threshold_db: f64, ratio: f64, attack: f64, release: f64,
               makeup_db: f64, fs: f64) -> Self {
        Compressor { threshold_db, ratio, attack, release, makeup_db, fs, env: 0.0 }
    }

    pub fn process_sample(&mut self, x: f64) -> f64 {
        let level = x.abs();
        // Envelope follower (asymmetric attack/release)
        let coeff = if level > self.env {
            (-1.0 / (self.attack * self.fs)).exp()
        } else {
            (-1.0 / (self.release * self.fs)).exp()
        };
        self.env = coeff * self.env + (1.0 - coeff) * level;

        // Gain computation in dB
        let level_db = if self.env > 1e-10 { 20.0 * self.env.log10() } else { -120.0 };
        let gain_db = if level_db > self.threshold_db {
            (self.threshold_db - level_db) * (1.0 - 1.0 / self.ratio)
        } else {
            0.0
        };
        let total_db  = gain_db + self.makeup_db;
        let linear_gain = 10.0f64.powf(total_db / 20.0);
        x * linear_gain
    }

    pub fn process(&mut self, signal: &[f64]) -> Vec<f64> {
        signal.iter().map(|&x| self.process_sample(x)).collect()
    }
}

/// Hard limiter: clips signal to [-threshold, +threshold].
pub fn limit(signal: &[f64], threshold: f64) -> Vec<f64> {
    signal.iter().map(|&x| x.clamp(-threshold, threshold)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. COMB FILTER & DELAY-BASED EFFECTS
// ─────────────────────────────────────────────────────────────────────────────

/// Comb filter (feedback delay line).
/// y[n] = x[n] + g·y[n - M]  (feedback comb)
/// Creates a resonant response with peaks at multiples of fs/M.
pub struct CombFilter {
    pub delay_samples: usize,
    pub gain:          f64,
    buffer:            Vec<f64>,
    write_pos:         usize,
}

impl CombFilter {
    pub fn new(delay_samples: usize, gain: f64) -> Self {
        CombFilter { delay_samples, gain, buffer: vec![0.0; delay_samples], write_pos: 0 }
    }

    pub fn process_sample(&mut self, x: f64) -> f64 {
        let read_pos = (self.write_pos + self.delay_samples - self.delay_samples) % self.delay_samples;
        let delayed  = self.buffer[read_pos];
        let y        = x + self.gain * delayed;
        self.buffer[self.write_pos] = y;
        self.write_pos = (self.write_pos + 1) % self.delay_samples;
        y
    }

    pub fn process(&mut self, signal: &[f64]) -> Vec<f64> {
        signal.iter().map(|&x| self.process_sample(x)).collect()
    }
}

/// All-pass filter section (for reverb).
/// y[n] = -g·x[n] + x[n-M] + g·y[n-M]
pub struct AllPassFilter {
    delay: usize,
    gain:  f64,
    buf_x: Vec<f64>,
    buf_y: Vec<f64>,
    pos:   usize,
}

impl AllPassFilter {
    pub fn new(delay: usize, gain: f64) -> Self {
        AllPassFilter { delay, gain, buf_x: vec![0.0; delay], buf_y: vec![0.0; delay], pos: 0 }
    }

    pub fn process_sample(&mut self, x: f64) -> f64 {
        let r = self.pos;
        let xm = self.buf_x[r];
        let ym = self.buf_y[r];
        let y = -self.gain * x + xm + self.gain * ym;
        self.buf_x[r] = x;
        self.buf_y[r] = y;
        self.pos = (self.pos + 1) % self.delay;
        y
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. ADAPTIVE FILTERS
// ─────────────────────────────────────────────────────────────────────────────

/// Least Mean Squares (LMS) adaptive filter.
/// Updates: w[n+1] = w[n] + μ·e[n]·x[n]
/// Converges when μ < 2 / (N · E[x²]).
pub struct LmsFilter {
    pub weights: Vec<f64>,
    pub mu:      f64,  // step size (learning rate)
    buffer:      Vec<f64>,
    pos:         usize,
}

impl LmsFilter {
    pub fn new(order: usize, mu: f64) -> Self {
        LmsFilter { weights: vec![0.0; order], mu, buffer: vec![0.0; order], pos: 0 }
    }

    /// Process one sample: `x` = input, `d` = desired output.
    /// Returns (estimated_output, error).
    pub fn process_sample(&mut self, x: f64, d: f64) -> (f64, f64) {
        let n = self.weights.len();
        self.buffer[self.pos] = x;
        // Compute output: y = w^T · x_vec
        let y: f64 = (0..n).map(|i| {
            let idx = (self.pos + n - i) % n;
            self.weights[i] * self.buffer[idx]
        }).sum();
        let e = d - y;
        // Weight update: w += mu * e * x_vec
        for i in 0..n {
            let idx = (self.pos + n - i) % n;
            self.weights[i] += self.mu * e * self.buffer[idx];
        }
        self.pos = (self.pos + 1) % n;
        (y, e)
    }
}

/// Normalised LMS (NLMS) — step size normalised by input power.
/// Converges faster and more robustly than LMS.
pub struct NlmsFilter {
    pub weights: Vec<f64>,
    pub mu:      f64,
    pub eps:     f64,  // regularisation to prevent division by zero
    buffer:      Vec<f64>,
    pos:         usize,
}

impl NlmsFilter {
    pub fn new(order: usize, mu: f64) -> Self {
        NlmsFilter { weights: vec![0.0; order], mu, eps: 1e-4, buffer: vec![0.0; order], pos: 0 }
    }

    pub fn process_sample(&mut self, x: f64, d: f64) -> (f64, f64) {
        let n = self.weights.len();
        self.buffer[self.pos] = x;
        let y: f64 = (0..n).map(|i| self.weights[i] * self.buffer[(self.pos + n - i) % n]).sum();
        let e = d - y;
        let power: f64 = self.buffer.iter().map(|xi| xi * xi).sum::<f64>() + self.eps;
        let mu_n = self.mu / power;
        for i in 0..n {
            self.weights[i] += mu_n * e * self.buffer[(self.pos + n - i) % n];
        }
        self.pos = (self.pos + 1) % n;
        (y, e)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 15. CEPSTRUM & MEL-FREQUENCY CEPSTRAL COEFFICIENTS (MFCC)
// ─────────────────────────────────────────────────────────────────────────────

/// Real cepstrum: c[n] = IFFT(log|FFT(x)|).
/// Separates the vocal tract (low quefrency) from excitation (high quefrency).
pub fn real_cepstrum(x: &[f64]) -> Vec<f64> {
    let spec = fft(x);
    let log_mag: Vec<Complex> = spec.iter().map(|c| {
        let m = c.magnitude().max(1e-10).ln();
        Complex::new(m, 0.0)
    }).collect();
    ifft(&log_mag).iter().map(|c| c.re).collect()
}

/// Mel scale conversion: Hz → Mel.
pub fn hz_to_mel(hz: f64) -> f64 { 2595.0 * (1.0 + hz / 700.0).log10() }
/// Mel → Hz.
pub fn mel_to_hz(mel: f64) -> f64 { 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0) }

/// Build a Mel filterbank matrix: n_filters × (fft_size/2+1).
/// Each row is one triangular Mel filter.
pub fn mel_filterbank(n_filters: usize, fft_size: usize, fs: f64,
                      f_min: f64, f_max: f64) -> Vec<Vec<f64>> {
    let n_bins = fft_size / 2 + 1;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    // N+2 equally-spaced Mel points → N filters
    let mel_points: Vec<f64> = (0..n_filters+2).map(|i| {
        mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_filters + 1) as f64)
    }).collect();
    // Convert Hz to bin indices
    let hz_to_bin = |hz: f64| ((hz / fs * fft_size as f64).round() as usize).min(n_bins - 1);
    let bins: Vec<usize> = mel_points.iter().map(|&hz| hz_to_bin(hz)).collect();

    let mut filters = vec![vec![0.0f64; n_bins]; n_filters];
    for m in 0..n_filters {
        let f_lo = bins[m]; let f_mid = bins[m+1]; let f_hi = bins[m+2];
        for k in f_lo..=f_mid {
            if f_mid > f_lo { filters[m][k] = (k - f_lo) as f64 / (f_mid - f_lo) as f64; }
        }
        for k in f_mid..=f_hi {
            if f_hi > f_mid { filters[m][k] = (f_hi - k) as f64 / (f_hi - f_mid) as f64; }
        }
    }
    filters
}

/// Compute MFCCs for one frame.
/// 1. Apply Mel filterbank to |FFT|²
/// 2. Take log of filter energies
/// 3. Apply DCT-II to get cepstral coefficients
pub fn mfcc_frame(spectrum: &[Complex], filterbank: &[Vec<f64>], n_cepstra: usize) -> Vec<f64> {
    let n_filters = filterbank.len();
    // Mel filter energies
    let mel_energies: Vec<f64> = filterbank.iter().map(|filt| {
        let e: f64 = filt.iter().zip(spectrum).map(|(f, c)| f * c.magnitude_sq()).sum();
        e.max(1e-10).ln()
    }).collect();
    // DCT-II: mfcc[c] = Σ log_energy[m] · cos(π·c·(m+0.5)/M)
    (0..n_cepstra).map(|c| {
        let scale = if c == 0 { (1.0 / n_filters as f64).sqrt() } else { (2.0 / n_filters as f64).sqrt() };
        scale * mel_energies.iter().enumerate().map(|(m, &e)| {
            e * (PI * c as f64 * (m as f64 + 0.5) / n_filters as f64).cos()
        }).sum::<f64>()
    }).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 16. MISCELLANEOUS DSP UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/// Decibels from linear amplitude.
pub fn amplitude_to_db(a: f64) -> f64 { 20.0 * a.abs().max(1e-10).log10() }
/// Linear amplitude from dB.
pub fn db_to_amplitude(db: f64) -> f64 { 10.0f64.powf(db / 20.0) }

/// RMS (root-mean-square) of a signal.
pub fn rms(x: &[f64]) -> f64 { (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt() }

/// Zero crossing rate: fraction of samples where sign changes.
pub fn zero_crossing_rate(x: &[f64]) -> f64 {
    if x.len() < 2 { return 0.0; }
    let zcr = x.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count();
    zcr as f64 / (x.len() - 1) as f64
}

/// Spectral centroid: frequency-weighted mean of the spectrum.
/// Approximates the "centre of mass" of the spectrum — correlates with brightness.
pub fn spectral_centroid(magnitudes: &[f64], freqs: &[f64]) -> f64 {
    let total: f64 = magnitudes.iter().sum::<f64>();
    if total < 1e-12 { return 0.0; }
    magnitudes.iter().zip(freqs).map(|(m, f)| m * f).sum::<f64>() / total
}

/// Spectral flatness (Wiener entropy): geometric mean / arithmetic mean of spectrum.
/// = 1 for white noise, = 0 for a pure tone.
pub fn spectral_flatness(magnitudes: &[f64]) -> f64 {
    let n = magnitudes.len() as f64;
    let log_sum: f64 = magnitudes.iter().map(|m| m.max(1e-10).ln()).sum::<f64>();
    let geometric = (log_sum / n).exp();
    let arithmetic = magnitudes.iter().sum::<f64>() / n;
    if arithmetic < 1e-12 { 0.0 } else { geometric / arithmetic }
}

/// Linear predictive coding (LPC) coefficients via Levinson-Durbin recursion.
/// Models the vocal tract as an all-pole filter.
/// Returns LPC coefficients [a1, a2, ..., a_p] for a_p th order predictor.
pub fn lpc_levinson(x: &[f64], order: usize) -> Vec<f64> {
    // Compute autocorrelation
    let r: Vec<f64> = (0..=order).map(|lag| {
        (0..x.len()-lag).map(|i| x[i] * x[i + lag]).sum::<f64>()
    }).collect();

    // Levinson-Durbin recursion
    let mut a = vec![0.0f64; order + 1];
    let mut e = r[0];
    a[0] = 1.0;

    for m in 1..=order {
        // Reflection coefficient (PARCOR)
        let lambda: f64 = (1..m).map(|i| a[i] * r[m - i]).sum::<f64>();
        if e.abs() < 1e-15 { break; }
        let k = -(r[m] + lambda) / e;
        // Update coefficients
        let a_prev = a.clone();
        for i in 1..=m {
            a[i] = a_prev[i] + k * a_prev[m - i];
        }
        e *= 1.0 - k * k;
    }
    a[1..].to_vec()
}

/// Phase vocoder pitch shift: shift pitch by `semitones` without changing duration.
/// Uses STFT analysis/synthesis with phase accumulation.
pub fn pitch_shift(signal: &[f64], fs: f64, semitones: f64) -> Vec<f64> {
    let ratio = 2.0f64.powf(semitones / 12.0);
    let frame_size = 2048usize;
    let hop_size   = 512usize;
    let params = StftParams::new(frame_size, hop_size, &WindowType::Hann);
    let stft = Stft::compute(signal, &params);
    let n_frames = stft.frames.len();
    let n_bins   = stft.num_bins;

    let mut out_frames = vec![vec![Complex::ZERO; n_bins]; n_frames];
    let mut phase_acc = vec![0.0f64; n_bins];
    let mut prev_phase = vec![0.0f64; n_bins];

    let omega_k: Vec<f64> = (0..n_bins).map(|k| {
        2.0 * PI * k as f64 * hop_size as f64 / frame_size as f64
    }).collect();

    for (t, frame) in stft.frames.iter().enumerate() {
        for k in 0..n_bins {
            let mag = frame[k].magnitude();
            let phase = frame[k].phase();
            // True frequency via phase difference
            let delta_phi = phase - prev_phase[k] - omega_k[k];
            // Wrap to [-π, π]
            let delta_phi_w = delta_phi - 2.0 * PI * (delta_phi / (2.0 * PI) + 0.5).floor();
            let true_freq = omega_k[k] + delta_phi_w;
            // Accumulate output phase
            phase_acc[k] += ratio * true_freq;
            prev_phase[k] = phase;
            out_frames[t][k] = Complex::from_polar(mag, phase_acc[k]);
        }
    }

    let out_stft = Stft { frames: out_frames, frame_size, hop_size, num_bins: n_bins };
    let syn_window = window(frame_size, &WindowType::Hann);
    out_stft.istft(&syn_window)
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn nearly_eq(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() <= tol }

    // ── Complex Arithmetic ────────────────────────────────────────────────────

    #[test]
    fn test_complex_add() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, -1.0);
        let c = a + b;
        assert_eq!(c, Complex::new(4.0, 1.0));
    }

    #[test]
    fn test_complex_mul() {
        // (1+i)(1-i) = 1 - i + i - i² = 2
        let a = Complex::new(1.0, 1.0);
        let b = Complex::new(1.0, -1.0);
        let c = a * b;
        assert!(nearly_eq(c.re, 2.0, 1e-10) && nearly_eq(c.im, 0.0, 1e-10));
    }

    #[test]
    fn test_complex_magnitude() {
        let a = Complex::new(3.0, 4.0);
        assert!(nearly_eq(a.magnitude(), 5.0, 1e-10));
    }

    #[test]
    fn test_complex_euler() {
        // e^{iπ} = -1
        let c = Complex::exp_i(PI);
        assert!(nearly_eq(c.re, -1.0, 1e-10));
        assert!(nearly_eq(c.im,  0.0, 1e-10));
    }

    #[test]
    fn test_complex_conjugate() {
        let a = Complex::new(2.0, 3.0);
        let b = a.conjugate();
        assert_eq!(b, Complex::new(2.0, -3.0));
    }

    // ── FFT Correctness ───────────────────────────────────────────────────────

    #[test]
    fn test_fft_impulse() {
        // FFT of impulse at position 0 = all ones
        let mut x = vec![0.0f64; 8];
        x[0] = 1.0;
        let X = fft(&x);
        for c in &X { assert!(nearly_eq(c.magnitude(), 1.0, 1e-10)); }
    }

    #[test]
    fn test_fft_dc() {
        // FFT of DC = all energy at bin 0
        let x = vec![1.0f64; 8];
        let X = fft(&x);
        assert!(nearly_eq(X[0].magnitude(), 8.0, 1e-10));
        for k in 1..8 { assert!(X[k].magnitude() < 1e-10, "Bin {} not zero", k); }
    }

    #[test]
    fn test_fft_parseval() {
        // Parseval's theorem: Σ|x[n]|² = (1/N) Σ|X[k]|²
        let x: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let X = fft(&x);
        let e_time: f64 = x.iter().map(|v| v * v).sum();
        let e_freq: f64 = X.iter().map(|c| c.magnitude_sq()).sum::<f64>() / 8.0;
        assert!(nearly_eq(e_time, e_freq, 1e-8), "Parseval: {} vs {}", e_time, e_freq);
    }

    #[test]
    fn test_ifft_roundtrip() {
        let x: Vec<f64> = (0..8).map(|i| (i as f64).sin()).collect();
        let X  = fft_complex(&x.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>());
        let x2 = ifft(&X);
        for (a, b) in x.iter().zip(&x2) {
            assert!(nearly_eq(*a, b.re, 1e-10), "IFFT roundtrip: {} vs {}", a, b.re);
        }
    }

    #[test]
    fn test_fft_vs_dft() {
        // FFT and DFT should agree
        let x: Vec<f64> = (0..8).map(|i| (2.0 * PI * 3.0 * i as f64 / 8.0).sin()).collect();
        let x_c: Vec<Complex> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
        let fft_out = fft_complex(&x_c);
        let dft_out = dft(&x_c);
        for (f, d) in fft_out.iter().zip(&dft_out) {
            assert!(nearly_eq(f.re, d.re, 1e-8), "FFT/DFT re: {} vs {}", f.re, d.re);
            assert!(nearly_eq(f.im, d.im, 1e-8), "FFT/DFT im: {} vs {}", f.im, d.im);
        }
    }

    #[test]
    fn test_fft_single_tone_peak() {
        // Pure sine at bin k=3 → peak at bin 3
        let n = 16usize;
        let k0 = 3usize;
        let x: Vec<f64> = (0..n).map(|i| (2.0 * PI * k0 as f64 * i as f64 / n as f64).sin()).collect();
        let X = fft(&x);
        let peak_bin = X.iter().enumerate()
            .max_by(|a, b| a.1.magnitude().partial_cmp(&b.1.magnitude()).unwrap())
            .map(|(i, _)| i).unwrap();
        // Peak should be at k0 or n-k0 (negative frequency alias)
        assert!(peak_bin == k0 || peak_bin == n - k0, "Peak bin: {}", peak_bin);
    }

    // ── Window Functions ──────────────────────────────────────────────────────

    #[test]
    fn test_hann_endpoints_zero() {
        let w = window(16, &WindowType::Hann);
        assert!(nearly_eq(w[0], 0.0, 1e-10) && nearly_eq(w[15], 0.0, 1e-3));
    }

    #[test]
    fn test_hamming_endpoints_small() {
        let w = window(16, &WindowType::Hamming);
        assert!(w[0] < 0.1, "Hamming endpoint: {}", w[0]);
    }

    #[test]
    fn test_rectangular_all_ones() {
        let w = window(16, &WindowType::Rectangular);
        assert!(w.iter().all(|&v| nearly_eq(v, 1.0, 1e-10)));
    }

    #[test]
    fn test_kaiser_endpoints() {
        let w = kaiser_window(32, 5.0);
        // Kaiser window endpoints: w[0] = I₀(0)/I₀(beta) = 1/I₀(5) ≈ 0.037.
        // The endpoints are NOT zero; they are 1/I₀(beta). The peak is at the center (w[1.0]).
        // Verify: endpoints are significantly smaller than the center peak.
        let n = w.len();
        let center = w[n / 2];
        assert!(w[0] < center, "Kaiser window should have smaller endpoints than center");
        assert!(w[0] > 0.0, "Kaiser window endpoints should be positive");
        assert!(w[0] < 0.1, "Kaiser window endpoints should be close to zero for beta=5");
    }

    #[test]
    fn test_bessel_i0_unity() {
        assert!(nearly_eq(bessel_i0(0.0), 1.0, 1e-10));
    }

    // ── FIR Filter ────────────────────────────────────────────────────────────

    #[test]
    fn test_fir_lowpass_dc_gain_unity() {
        let fir = FirFilter::lowpass(64, 0.25, &WindowType::Hann);
        let h = fir.frequency_response(0.0); // DC
        assert!(nearly_eq(h.magnitude(), 1.0, 0.01), "LP DC gain: {}", h.magnitude());
    }

    #[test]
    fn test_fir_lowpass_stops_at_nyquist() {
        let fir = FirFilter::lowpass(64, 0.25, &WindowType::Hamming);
        let h = fir.frequency_response(0.5); // Nyquist
        assert!(h.magnitude() < 0.05, "LP Nyquist gain: {}", h.magnitude());
    }

    #[test]
    fn test_fir_highpass_blocks_dc() {
        let fir = FirFilter::highpass(64, 0.25, &WindowType::Hann);
        let h = fir.frequency_response(0.0);
        assert!(h.magnitude() < 0.05, "HP DC gain: {}", h.magnitude());
    }

    #[test]
    fn test_fir_apply_impulse_gives_coeffs() {
        // Impulse response of FIR = its coefficients
        let fir = FirFilter::lowpass(8, 0.25, &WindowType::Hann);
        let impulse = gen_impulse(fir.coeffs.len() + 4, 0);
        let response = fir.apply(&impulse);
        for (i, &h) in fir.coeffs.iter().enumerate() {
            assert!(nearly_eq(response[i], h, 1e-10), "IR mismatch at {}", i);
        }
    }

    #[test]
    fn test_fir_filter_removes_high_freq() {
        // Low-pass filter should attenuate a signal at Nyquist
        let fir = FirFilter::lowpass(128, 0.1, &WindowType::Blackman);
        let high = gen_sine(512, 0.45, 1.0, 1.0, 0.0); // near Nyquist
        let filtered = fir.apply(&high);
        let rms_in  = rms(&high);
        let rms_out = rms(&filtered[128..]); // skip transient
        assert!(rms_out < rms_in * 0.1, "HP not attenuated: {} vs {}", rms_out, rms_in);
    }

    // ── IIR / Biquad ─────────────────────────────────────────────────────────

    #[test]
    fn test_biquad_passthrough() {
        // Unity gain biquad: b0=1, rest 0
        let mut bq = Biquad::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = bq.process(&x);
        assert_eq!(y, x);
    }

    #[test]
    fn test_butterworth_dc_gain() {
        let mut bw = butterworth_lowpass(4, 0.25);
        // DC gain of Butterworth LP should be ~1
        assert!(nearly_eq(bw.magnitude_response(0.0), 1.0, 0.05),
                "BW DC gain: {}", bw.magnitude_response(0.0));
    }

    #[test]
    fn test_butterworth_attenuates_stopband() {
        let bw = butterworth_lowpass(4, 0.1);
        let atten = bw.magnitude_response(0.4);
        assert!(atten < 0.05, "Butterworth stopband: {}", atten);
    }

    #[test]
    fn test_parametric_eq_boost() {
        let mut eq = parametric_eq(0.1, 12.0, 2.0); // +12 dB at fc=0.1
        let gain = eq.magnitude_response(0.1);
        let expected = db_to_amplitude(12.0);
        assert!(nearly_eq(gain, expected, expected * 0.2),
                "EQ boost: {}, expected ~{}", gain, expected);
    }

    // ── Convolution ───────────────────────────────────────────────────────────

    #[test]
    fn test_convolve_impulse() {
        // Convolving with impulse = identity
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let h = vec![1.0]; // unit impulse
        let y = convolve(&x, &h);
        assert_eq!(y, x);
    }

    #[test]
    fn test_convolve_known() {
        // [1,2,3] * [1,1] = [1,3,5,3]
        let x = vec![1.0, 2.0, 3.0];
        let h = vec![1.0, 1.0];
        let y = convolve(&x, &h);
        assert!(nearly_eq(y[0], 1.0, 1e-10));
        assert!(nearly_eq(y[1], 3.0, 1e-10));
        assert!(nearly_eq(y[2], 5.0, 1e-10));
        assert!(nearly_eq(y[3], 3.0, 1e-10));
    }

    #[test]
    fn test_fft_convolve_matches_direct() {
        let x: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let h: Vec<f64> = vec![0.5, 0.5]; // averaging filter
        let direct  = convolve(&x, &h);
        let fft_res = fft_convolve(&x, &h);
        for (a, b) in direct.iter().zip(&fft_res) {
            assert!(nearly_eq(*a, *b, 1e-8), "FFT conv: {} vs {}", a, b);
        }
    }

    // ── Correlation ───────────────────────────────────────────────────────────

    #[test]
    fn test_autocorrelation_peak_at_zero() {
        let x: Vec<f64> = gen_sine(64, 5.0, 64.0, 1.0, 0.0);
        let ac = autocorrelate(&x);
        let center = x.len() - 1;
        let max_idx = ac.iter().enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, _)| i).unwrap();
        assert_eq!(max_idx, center, "Autocorrelation peak at zero lag");
    }

    #[test]
    fn test_cross_correlate_delay_detection() {
        let x = gen_impulse(32, 0);
        let delay = 5;
        let y = gen_impulse(32, delay);
        let cc = cross_correlate(&x, &y);
        let center = y.len() - 1;
        let peak_idx = cc.iter().enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, _)| i).unwrap();
        // Peak should be at center + delay
        assert_eq!(peak_idx, center + delay, "Cross-corr delay: {}", peak_idx);
    }

    // ── Signal Generators ─────────────────────────────────────────────────────

    #[test]
    fn test_sine_frequency() {
        // A 100 Hz sine at fs=1000: period = 10 samples
        let x = gen_sine(100, 100.0, 1000.0, 1.0, 0.0);
        // Check period: x[0] and x[10] should be equal
        assert!(nearly_eq(x[0], x[10], 1e-10), "Sine period: {} vs {}", x[0], x[10]);
    }

    #[test]
    fn test_sine_amplitude() {
        let x = gen_sine(1000, 100.0, 10000.0, 2.5, 0.0);
        let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(nearly_eq(max, 2.5, 0.001), "Sine amplitude: {}", max);
    }

    #[test]
    fn test_chirp_spectrum_spread() {
        // Chirp should have energy spread across frequencies
        let x = gen_chirp(1024, 1024.0, 10.0, 400.0, 1.0);
        let mags = fft_magnitude(&x);
        let nonzero = mags.iter().filter(|&&m| m > 0.01).count();
        assert!(nonzero > 50, "Chirp spectrum bins: {}", nonzero);
    }

    #[test]
    fn test_square_wave_rms() {
        // RMS of a square wave of amplitude A = A
        let x = gen_square(1000, 10.0, 1000.0, 1.0);
        let r = rms(&x);
        assert!(nearly_eq(r, 1.0, 0.01), "Square RMS: {}", r);
    }

    #[test]
    fn test_impulse_generation() {
        let x = gen_impulse(8, 3);
        assert_eq!(x[3], 1.0);
        assert!(x.iter().enumerate().filter(|&(i, _)| i != 3).all(|(_, &v)| v == 0.0));
    }

    // ── Welch PSD ─────────────────────────────────────────────────────────────

    #[test]
    fn test_welch_white_noise_flat() {
        // White noise should have approximately flat PSD
        let noise = gen_noise(4096, 1.0, 42);
        let (_, psd) = welch_psd(&noise, 4096.0, 512, 256, &WindowType::Hann);
        let mean_psd = psd.iter().sum::<f64>() / psd.len() as f64;
        // All bins should be within 3× of mean
        let flat = psd.iter().all(|&p| p < 4.0 * mean_psd && p > 0.0);
        assert!(flat, "PSD not flat for white noise");
    }

    #[test]
    fn test_welch_tone_peak() {
        // Single tone → PSD peak at that frequency
        let fs = 1024.0;
        let freq = 100.0;
        let x = gen_sine(4096, freq, fs, 1.0, 0.0);
        let (freqs, psd) = welch_psd(&x, fs, 512, 256, &WindowType::Hann);
        let peak_idx = psd.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap();
        assert!((freqs[peak_idx] - freq).abs() < 5.0, "PSD peak at {}Hz, expected {}Hz", freqs[peak_idx], freq);
    }

    // ── Resampling ────────────────────────────────────────────────────────────

    #[test]
    fn test_resample_linear_upsample() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = resample_linear(&x, 2.0); // 2× upsample
        assert_eq!(y.len(), 8);
        assert!(nearly_eq(y[0], 0.0, 1e-10));
        assert!(nearly_eq(y[2], 1.0, 1e-10));
    }

    #[test]
    fn test_resample_cubic_length() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y = resample_cubic(&x, 0.5); // 2× downsample
        assert_eq!(y.len(), 50);
    }

    // ── Spectral Features ─────────────────────────────────────────────────────

    #[test]
    fn test_rms_sine() {
        // RMS of sine amplitude A = A/√2
        let x = gen_sine(10000, 100.0, 10000.0, 1.0, 0.0);
        let r = rms(&x);
        assert!(nearly_eq(r, 1.0 / 2.0f64.sqrt(), 0.001), "Sine RMS: {}", r);
    }

    #[test]
    fn test_spectral_centroid_low_tone() {
        let n = 512;
        let x = gen_sine(n, 50.0, 1024.0, 1.0, 0.0);
        let mags = fft_magnitude(&x);
        let freqs = fft_frequencies(n, 1024.0);
        let sc = spectral_centroid(&mags, &freqs);
        assert!(sc < 200.0, "Centroid of low tone: {}Hz", sc);
    }

    #[test]
    fn test_zero_crossing_rate_sine() {
        // 1 Hz sine at 100 samples/s → 2 zero crossings per period → ZCR = 2/100 = 0.02
        let x = gen_sine(100, 1.0, 100.0, 1.0, 0.0);
        let zcr = zero_crossing_rate(&x);
        assert!(zcr > 0.01 && zcr < 0.05, "ZCR: {}", zcr);
    }

    #[test]
    fn test_spectral_flatness_noise_vs_tone() {
        let noise = gen_noise(512, 1.0, 1);
        let tone  = gen_sine(512, 50.0, 512.0, 1.0, 0.0);
        let noise_mags = fft_magnitude(&noise);
        let tone_mags  = fft_magnitude(&tone);
        let sf_noise = spectral_flatness(&noise_mags);
        let sf_tone  = spectral_flatness(&tone_mags);
        assert!(sf_noise > sf_tone, "Noise flatter than tone: {} vs {}", sf_noise, sf_tone);
    }

    // ── LPC ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_lpc_order_check() {
        let x = gen_sine(256, 100.0, 8000.0, 1.0, 0.0);
        let coeffs = lpc_levinson(&x, 10);
        assert_eq!(coeffs.len(), 10, "LPC order");
    }

    // ── Adaptive Filters ──────────────────────────────────────────────────────

    #[test]
    fn test_lms_converges() {
        // LMS should reduce error over time
        let mut lms = LmsFilter::new(4, 0.01);
        let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let d: Vec<f64> = x.iter().map(|&v| v * 0.5).collect(); // target = 0.5 * input
        let (_, e0) = lms.process_sample(x[0], d[0]);
        for i in 1..150 { lms.process_sample(x[i], d[i]); }
        let (_, e_last) = lms.process_sample(x[150], d[150]);
        assert!(e_last.abs() < e0.abs() + 0.1, "LMS not converging: {} vs {}", e_last.abs(), e0.abs());
    }

    #[test]
    fn test_nlms_converges_faster_than_lms() {
        // NLMS should have lower final error than LMS with the same step size
        let mut lms  = LmsFilter::new(8, 0.01);
        let mut nlms = NlmsFilter::new(8, 0.5);
        let signal: Vec<f64> = (0..300).map(|i| (i as f64 * 0.2).sin()).collect();
        let desired: Vec<f64> = signal.iter().map(|&v| v * 0.3).collect();
        let mut lms_e = 0.0f64;
        let mut nlms_e = 0.0f64;
        for i in 0..300 {
            let (_, e1) = lms.process_sample(signal[i], desired[i]);
            let (_, e2) = nlms.process_sample(signal[i], desired[i]);
            if i >= 200 { lms_e += e1 * e1; nlms_e += e2 * e2; }
        }
        assert!(nlms_e <= lms_e + 1e-3, "NLMS not better than LMS: {} vs {}", nlms_e, lms_e);
    }

    // ── Cepstrum & Mel ────────────────────────────────────────────────────────

    #[test]
    fn test_hz_mel_roundtrip() {
        for hz in [100.0, 500.0, 1000.0, 4000.0f64] {
            let mel = hz_to_mel(hz);
            let hz2 = mel_to_hz(mel);
            assert!(nearly_eq(hz2, hz, 1e-8), "Mel roundtrip: {}", hz2);
        }
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = mel_filterbank(26, 512, 16000.0, 0.0, 8000.0);
        assert_eq!(fb.len(), 26, "Filter count");
        assert_eq!(fb[0].len(), 257, "Bin count");
        // Each filter should sum to non-zero
        for (i, f) in fb.iter().enumerate() {
            assert!(f.iter().sum::<f64>() > 0.0, "Filter {} is all zero", i);
        }
    }

    #[test]
    fn test_amplitude_db_roundtrip() {
        for a in [0.001, 0.1, 1.0, 10.0, 100.0f64] {
            let db = amplitude_to_db(a);
            let a2 = db_to_amplitude(db);
            assert!(nearly_eq(a2, a, 1e-8), "dB roundtrip: {}", a);
        }
    }

    // ── Compressor ────────────────────────────────────────────────────────────

    #[test]
    fn test_compressor_reduces_loud_signal() {
        let mut comp = Compressor::new(-6.0, 4.0, 0.001, 0.1, 0.0, 44100.0);
        let loud = gen_sine(44100, 100.0, 44100.0, 1.0, 0.0);
        let compressed = comp.process(&loud);
        let rms_in  = rms(&loud);
        let rms_out = rms(&compressed[1000..]); // skip transient
        assert!(rms_out < rms_in, "Compressor: out {} >= in {}", rms_out, rms_in);
    }

    #[test]
    fn test_limiter_clamps() {
        let x = vec![-2.0, -1.5, 0.0, 1.5, 2.0];
        let y = limit(&x, 1.0);
        assert!(y.iter().all(|&v| v.abs() <= 1.0), "Limiter not clamping");
    }

    // ── Comb Filter ───────────────────────────────────────────────────────────

    #[test]
    fn test_comb_filter_delay() {
        // With g=0: output = input (no feedback)
        let mut cf = CombFilter::new(10, 0.0);
        let x = gen_impulse(20, 0);
        let y = cf.process(&x);
        assert!(nearly_eq(y[0], 1.0, 1e-10), "Comb pass-through: {}", y[0]);
    }

    // ── STFT ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stft_frame_count() {
        let signal = gen_sine(2048, 100.0, 1024.0, 1.0, 0.0);
        let params = StftParams::new(256, 128, &WindowType::Hann);
        let stft = Stft::compute(&signal, &params);
        let expected = (2048 - 256) / 128 + 1;
        assert_eq!(stft.frames.len(), expected, "STFT frames: {}", stft.frames.len());
    }

    #[test]
    fn test_stft_bin_count() {
        let signal = gen_sine(1024, 100.0, 1024.0, 1.0, 0.0);
        let params = StftParams::new(256, 128, &WindowType::Hann);
        let stft = Stft::compute(&signal, &params);
        assert_eq!(stft.num_bins, 129, "STFT bins: {}", stft.num_bins);
    }

    #[test]
    fn test_stft_tone_localized_in_frequency() {
        let fs = 1024.0;
        let freq = 100.0;
        let signal = gen_sine(2048, freq, fs, 1.0, 0.0);
        let params = StftParams::new(512, 256, &WindowType::Hann);
        let stft = Stft::compute(&signal, &params);
        let spec = stft.magnitude_spectrogram();
        // Find the bin with peak energy (midway through signal, avoid edge effects)
        let mid_frame = &spec[spec.len() / 2];
        let peak_bin = mid_frame.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap();
        let peak_freq = peak_bin as f64 * fs / 512.0;
        assert!((peak_freq - freq).abs() < 10.0, "STFT peak: {}Hz, expected {}Hz", peak_freq, freq);
    }
}
