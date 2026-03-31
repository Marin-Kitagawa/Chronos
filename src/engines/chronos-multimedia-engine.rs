// ============================================================================
// CHRONOS MULTIMEDIA ENGINE
// ============================================================================
//
// HOW DIGITAL AUDIO & VIDEO ACTUALLY WORK (and how this code models it):
//
// All multimedia is fundamentally about SAMPLING continuous signals at
// discrete intervals. Audio samples pressure waves (typically 44,100 or
// 48,000 times per second), and video samples light (typically 24-60
// frames per second, each frame a grid of pixels). Everything in this
// engine — synthesis, filtering, encoding, effects — operates on these
// discrete samples.
//
// AUDIO: A digital audio signal is a sequence of floating-point numbers
// between -1.0 and 1.0, where 0.0 is silence. The sample rate determines
// the highest frequency that can be represented (Nyquist theorem: max
// frequency = sample_rate / 2). Multiple channels (stereo, surround)
// are interleaved: [L0, R0, L1, R1, L2, R2, ...].
//
// VIDEO: A digital video frame is a 2D grid of pixels. Each pixel is
// typically 3 or 4 channels (RGB or RGBA), each 8 bits (0-255). Video
// compression exploits temporal redundancy (most pixels don't change
// between frames) and spatial redundancy (nearby pixels are similar).
//
// IMAGE PROCESSING: Convolution with small kernels (3x3, 5x5) is the
// fundamental operation — blur, sharpen, edge detect, emboss are all
// just different kernel matrices applied to each pixel neighborhood.
//
// WHAT THIS ENGINE IMPLEMENTS (real, working logic with tests):
//   1.  Audio sample buffer with multi-channel support
//   2.  Oscillators: sine, square, sawtooth, triangle, white noise
//   3.  ADSR envelope generator (Attack-Decay-Sustain-Release)
//   4.  Biquad filter (low-pass, high-pass, band-pass, notch, peaking)
//   5.  FFT (Fast Fourier Transform) — Cooley-Tukey radix-2
//   6.  Audio effects: delay, reverb (Schroeder), chorus, distortion,
//       compressor, ring modulation
//   7.  MIDI message parsing (Note On/Off, CC, Pitch Bend, etc.)
//   8.  Audio codec framework (PCM, μ-law, A-law encoding/decoding)
//   9.  Image buffer with pixel operations
//  10.  Image convolution kernels (blur, sharpen, edge detect, emboss)
//  11.  Color space conversion (RGB ↔ HSV ↔ YUV ↔ grayscale)
//  12.  Image transformations (resize nearest-neighbor/bilinear, rotate,
//       flip, crop)
//  13.  Histogram computation and equalization
//  14.  Video frame buffer with inter-frame operations
//  15.  Motion detection (frame differencing)
//  16.  Simple video transition effects (crossfade, wipe)
//  17.  Spatial audio (stereo panning, distance attenuation)
// ============================================================================

use std::collections::HashMap;
use std::f64::consts::PI;

// ============================================================================
// PART 1: AUDIO SAMPLE BUFFER
// ============================================================================
// The audio buffer is the fundamental data structure for all audio
// processing. It stores interleaved floating-point samples for one or
// more channels. All audio operations consume and produce AudioBuffers.

/// An audio sample buffer.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Interleaved samples: [ch0_s0, ch1_s0, ch0_s1, ch1_s1, ...]
    pub samples: Vec<f64>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioBuffer {
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        AudioBuffer { samples: Vec::new(), sample_rate, channels }
    }

    pub fn from_samples(samples: Vec<f64>, sample_rate: u32, channels: u16) -> Self {
        AudioBuffer { samples, sample_rate, channels }
    }

    /// Create a mono buffer of silence with the given duration in seconds.
    pub fn silence(sample_rate: u32, channels: u16, duration_secs: f64) -> Self {
        let num_samples = (sample_rate as f64 * duration_secs) as usize * channels as usize;
        AudioBuffer {
            samples: vec![0.0; num_samples],
            sample_rate,
            channels,
        }
    }

    /// Number of sample frames (each frame has one sample per channel).
    pub fn num_frames(&self) -> usize {
        if self.channels == 0 { return 0; }
        self.samples.len() / self.channels as usize
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f64 {
        self.num_frames() as f64 / self.sample_rate as f64
    }

    /// Get a sample at the given frame and channel.
    pub fn get(&self, frame: usize, channel: u16) -> f64 {
        let idx = frame * self.channels as usize + channel as usize;
        self.samples.get(idx).copied().unwrap_or(0.0)
    }

    /// Set a sample at the given frame and channel.
    pub fn set(&mut self, frame: usize, channel: u16, value: f64) {
        let idx = frame * self.channels as usize + channel as usize;
        if idx < self.samples.len() {
            self.samples[idx] = value;
        }
    }

    /// Push a frame (one sample per channel).
    pub fn push_frame(&mut self, frame: &[f64]) {
        self.samples.extend_from_slice(&frame[..self.channels as usize]);
    }

    /// Mix (add) another buffer into this one, with a gain factor.
    pub fn mix(&mut self, other: &AudioBuffer, gain: f64) {
        let len = self.samples.len().min(other.samples.len());
        for i in 0..len {
            self.samples[i] += other.samples[i] * gain;
        }
    }

    /// Apply a gain (volume) to all samples.
    pub fn apply_gain(&mut self, gain: f64) {
        for s in &mut self.samples {
            *s *= gain;
        }
    }

    /// Clip all samples to the [-1.0, 1.0] range.
    pub fn clip(&mut self) {
        for s in &mut self.samples {
            *s = s.clamp(-1.0, 1.0);
        }
    }

    /// Compute RMS (Root Mean Square) level — the standard measure of
    /// perceived loudness. Returns one value per channel.
    pub fn rms(&self) -> Vec<f64> {
        let mut sums = vec![0.0f64; self.channels as usize];
        let frames = self.num_frames();
        if frames == 0 { return sums; }
        for frame in 0..frames {
            for ch in 0..self.channels as usize {
                let s = self.get(frame, ch as u16);
                sums[ch] += s * s;
            }
        }
        sums.iter().map(|sum| (sum / frames as f64).sqrt()).collect()
    }

    /// Peak absolute level per channel.
    pub fn peak(&self) -> Vec<f64> {
        let mut peaks = vec![0.0f64; self.channels as usize];
        for frame in 0..self.num_frames() {
            for ch in 0..self.channels as usize {
                let abs = self.get(frame, ch as u16).abs();
                if abs > peaks[ch] { peaks[ch] = abs; }
            }
        }
        peaks
    }
}

// ============================================================================
// PART 2: OSCILLATORS
// ============================================================================
// An oscillator generates a periodic waveform at a given frequency.
// These are the building blocks of all synthesized sound. Each waveform
// has a distinct timbre (harmonic content):
//   - Sine: pure tone, single frequency (no harmonics)
//   - Square: odd harmonics (1, 3, 5, ...) with 1/n amplitude — hollow sound
//   - Sawtooth: all harmonics (1, 2, 3, ...) with 1/n amplitude — bright/buzzy
//   - Triangle: odd harmonics with 1/n² amplitude — softer than square
//   - White noise: random, all frequencies equally — used for percussion, wind

/// Waveform type for oscillators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Waveform {
    Sine,
    Square,
    Sawtooth,
    Triangle,
    WhiteNoise,
}

/// A basic oscillator.
pub struct Oscillator {
    pub waveform: Waveform,
    pub frequency: f64,      // Hz
    pub amplitude: f64,      // 0.0 to 1.0
    pub phase: f64,          // current phase in radians
    pub sample_rate: u32,
    /// Simple pseudo-random state for noise generation
    noise_state: u64,
}

impl Oscillator {
    pub fn new(waveform: Waveform, frequency: f64, amplitude: f64, sample_rate: u32) -> Self {
        Oscillator {
            waveform, frequency, amplitude, phase: 0.0, sample_rate,
            noise_state: 12345,
        }
    }

    /// Generate the next sample and advance the phase.
    pub fn next_sample(&mut self) -> f64 {
        let value = match self.waveform {
            Waveform::Sine => self.phase.sin(),
            Waveform::Square => {
                if self.phase.sin() >= 0.0 { 1.0 } else { -1.0 }
            }
            Waveform::Sawtooth => {
                // Map phase [0, 2π) to [-1, 1)
                let t = self.phase / (2.0 * PI);
                2.0 * (t - (t + 0.5).floor())
            }
            Waveform::Triangle => {
                let t = self.phase / (2.0 * PI);
                2.0 * (2.0 * (t - (t + 0.5).floor())).abs() - 1.0
            }
            Waveform::WhiteNoise => {
                // xorshift64 PRNG
                self.noise_state ^= self.noise_state << 13;
                self.noise_state ^= self.noise_state >> 7;
                self.noise_state ^= self.noise_state << 17;
                (self.noise_state as f64 / u64::MAX as f64) * 2.0 - 1.0
            }
        };

        // Advance phase
        let phase_increment = 2.0 * PI * self.frequency / self.sample_rate as f64;
        self.phase += phase_increment;
        if self.phase >= 2.0 * PI {
            self.phase -= 2.0 * PI;
        }

        value * self.amplitude
    }

    /// Generate a buffer of samples for a given duration.
    pub fn generate(&mut self, duration_secs: f64) -> AudioBuffer {
        let num_samples = (self.sample_rate as f64 * duration_secs) as usize;
        let mut samples = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            samples.push(self.next_sample());
        }
        AudioBuffer::from_samples(samples, self.sample_rate, 1)
    }
}

// ============================================================================
// PART 3: ADSR ENVELOPE
// ============================================================================
// The ADSR envelope shapes the amplitude of a sound over time:
//   Attack:  0 → 1.0 (how fast the sound reaches full volume)
//   Decay:   1.0 → sustain_level (initial drop after the attack)
//   Sustain: held at sustain_level (while the key is held)
//   Release: sustain_level → 0 (fade out after key release)
//
// This is the fundamental tool for making synthesized sounds feel natural.
// A piano has a fast attack, quick decay, low sustain, and medium release.
// A pad has a slow attack, no decay, high sustain, and slow release.

#[derive(Debug, Clone)]
pub struct AdsrEnvelope {
    pub attack_time: f64,    // seconds
    pub decay_time: f64,     // seconds
    pub sustain_level: f64,  // 0.0 to 1.0
    pub release_time: f64,   // seconds
    sample_rate: u32,
}

impl AdsrEnvelope {
    pub fn new(attack: f64, decay: f64, sustain: f64, release: f64, sample_rate: u32) -> Self {
        AdsrEnvelope {
            attack_time: attack,
            decay_time: decay,
            sustain_level: sustain.clamp(0.0, 1.0),
            release_time: release,
            sample_rate,
        }
    }

    /// Generate the envelope curve for a note with the given gate time
    /// (how long the key is held). Returns a buffer of gain values.
    pub fn generate(&self, gate_time: f64) -> Vec<f64> {
        let total_time = gate_time + self.release_time;
        let num_samples = (self.sample_rate as f64 * total_time) as usize;
        let mut envelope = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = i as f64 / self.sample_rate as f64;
            let value = if t < self.attack_time {
                // Attack phase: linear ramp from 0 to 1
                t / self.attack_time
            } else if t < self.attack_time + self.decay_time {
                // Decay phase: linear ramp from 1 to sustain_level
                let decay_progress = (t - self.attack_time) / self.decay_time;
                1.0 - decay_progress * (1.0 - self.sustain_level)
            } else if t < gate_time {
                // Sustain phase: hold at sustain_level
                self.sustain_level
            } else {
                // Release phase: linear ramp from sustain_level to 0
                let release_progress = (t - gate_time) / self.release_time;
                self.sustain_level * (1.0 - release_progress).max(0.0)
            };
            envelope.push(value.clamp(0.0, 1.0));
        }

        envelope
    }

    /// Apply the envelope to an audio buffer in place.
    pub fn apply(&self, buffer: &mut AudioBuffer, gate_time: f64) {
        let envelope = self.generate(gate_time);
        let frames = buffer.num_frames();
        for frame in 0..frames {
            let gain = if frame < envelope.len() { envelope[frame] } else { 0.0 };
            for ch in 0..buffer.channels {
                let old = buffer.get(frame, ch);
                buffer.set(frame, ch, old * gain);
            }
        }
    }
}

// ============================================================================
// PART 4: BIQUAD FILTER
// ============================================================================
// The biquad filter is the workhorse of digital audio processing. It's a
// second-order IIR (Infinite Impulse Response) filter that can implement
// low-pass, high-pass, band-pass, notch, peaking EQ, and shelving filters
// by changing its five coefficients (b0, b1, b2, a1, a2).
//
// The difference equation:
//   y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
//
// The coefficients are computed from the desired filter type, cutoff
// frequency, Q factor (resonance), and gain using the Robert Bristow-
// Johnson audio EQ cookbook formulas.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    Notch,
    PeakingEQ,
    LowShelf,
    HighShelf,
}

#[derive(Debug, Clone)]
pub struct BiquadFilter {
    b0: f64, b1: f64, b2: f64,
    a1: f64, a2: f64,
    // State variables (previous input and output samples)
    x1: f64, x2: f64,
    y1: f64, y2: f64,
}

impl BiquadFilter {
    /// Design a biquad filter using the Audio EQ Cookbook formulas.
    /// freq: center/cutoff frequency in Hz
    /// q: Q factor (resonance), typically 0.5-10
    /// gain_db: gain for peaking/shelving filters
    /// sample_rate: in Hz
    pub fn design(filter_type: FilterType, freq: f64, q: f64,
                  gain_db: f64, sample_rate: u32) -> Self
    {
        let w0 = 2.0 * PI * freq / sample_rate as f64;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);
        let a = 10.0f64.powf(gain_db / 40.0); // for peaking/shelving

        let (b0, b1, b2, a0, a1, a2) = match filter_type {
            FilterType::LowPass => {
                let b1 = 1.0 - cos_w0;
                let b0 = b1 / 2.0;
                let b2 = b0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::HighPass => {
                let b1 = -(1.0 + cos_w0);
                let b0 = (1.0 + cos_w0) / 2.0;
                let b2 = b0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::BandPass => {
                let b0 = alpha;
                let b1 = 0.0;
                let b2 = -alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::Notch => {
                let b0 = 1.0;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::PeakingEQ => {
                let b0 = 1.0 + alpha * a;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 - alpha * a;
                let a0 = 1.0 + alpha / a;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha / a;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::LowShelf => {
                let sq = 2.0 * a.sqrt() * alpha;
                let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + sq);
                let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - sq);
                let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + sq;
                let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - sq;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::HighShelf => {
                let sq = 2.0 * a.sqrt() * alpha;
                let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + sq);
                let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - sq);
                let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + sq;
                let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - sq;
                (b0, b1, b2, a0, a1, a2)
            }
        };

        // Normalize coefficients by a0
        BiquadFilter {
            b0: b0 / a0, b1: b1 / a0, b2: b2 / a0,
            a1: a1 / a0, a2: a2 / a0,
            x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0,
        }
    }

    /// Process a single sample through the filter.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
                   - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;
        output
    }

    /// Process an entire audio buffer in place.
    pub fn process(&mut self, buffer: &mut AudioBuffer) {
        for i in 0..buffer.samples.len() {
            buffer.samples[i] = self.process_sample(buffer.samples[i]);
        }
    }

    /// Reset the filter state (clear delay line).
    pub fn reset(&mut self) {
        self.x1 = 0.0; self.x2 = 0.0;
        self.y1 = 0.0; self.y2 = 0.0;
    }
}

// ============================================================================
// PART 5: FFT (Fast Fourier Transform)
// ============================================================================
// The FFT converts a time-domain signal into its frequency-domain
// representation. This is arguably the most important algorithm in all
// of signal processing — it's used for spectral analysis, convolution,
// pitch detection, compression, and countless other applications.
//
// The Cooley-Tukey algorithm computes the DFT in O(N log N) instead of
// the naive O(N²). It works by recursively splitting the DFT into two
// half-sized DFTs (one for even-indexed samples, one for odd-indexed),
// computing them separately, and combining the results with "twiddle
// factors" (complex exponentials). The input size must be a power of 2.

/// A complex number for FFT computation.
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self { Complex { re, im } }
    pub fn zero() -> Self { Complex { re: 0.0, im: 0.0 } }
    pub fn from_polar(mag: f64, phase: f64) -> Self {
        Complex { re: mag * phase.cos(), im: mag * phase.sin() }
    }
    pub fn magnitude(&self) -> f64 { (self.re * self.re + self.im * self.im).sqrt() }
    pub fn phase(&self) -> f64 { self.im.atan2(self.re) }
    pub fn add(&self, other: &Complex) -> Complex {
        Complex { re: self.re + other.re, im: self.im + other.im }
    }
    pub fn sub(&self, other: &Complex) -> Complex {
        Complex { re: self.re - other.re, im: self.im - other.im }
    }
    pub fn mul(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
    pub fn scale(&self, factor: f64) -> Complex {
        Complex { re: self.re * factor, im: self.im * factor }
    }
}

/// Compute the FFT of a complex signal using the Cooley-Tukey algorithm.
/// Input length MUST be a power of 2.
pub fn fft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    if n <= 1 { return input.to_vec(); }

    assert!(n.is_power_of_two(), "FFT input length must be a power of 2, got {}", n);

    // Bit-reversal permutation (iterative for efficiency)
    let mut output: Vec<Complex> = vec![Complex::zero(); n];
    let bits = (n as f64).log2() as u32;
    for i in 0..n {
        let rev = bit_reverse(i as u32, bits) as usize;
        output[rev] = input[i];
    }

    // Butterfly operations
    let mut size = 2;
    while size <= n {
        let half = size / 2;
        let w_n = Complex::from_polar(1.0, -2.0 * PI / size as f64);
        let mut i = 0;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..half {
                let u = output[i + j];
                let t = w.mul(&output[i + j + half]);
                output[i + j] = u.add(&t);
                output[i + j + half] = u.sub(&t);
                w = w.mul(&w_n);
            }
            i += size;
        }
        size *= 2;
    }

    output
}

/// Compute the inverse FFT (frequency domain → time domain).
pub fn ifft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    // Conjugate, apply FFT, conjugate again, and scale by 1/N
    let conjugated: Vec<Complex> = input.iter()
        .map(|c| Complex::new(c.re, -c.im))
        .collect();
    let transformed = fft(&conjugated);
    transformed.iter()
        .map(|c| Complex::new(c.re / n as f64, -c.im / n as f64))
        .collect()
}

/// Bit-reverse an integer with the given number of bits.
fn bit_reverse(x: u32, bits: u32) -> u32 {
    let mut result = 0u32;
    let mut x = x;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Compute the magnitude spectrum from FFT output.
pub fn magnitude_spectrum(fft_output: &[Complex]) -> Vec<f64> {
    fft_output.iter().map(|c| c.magnitude()).collect()
}

/// Compute the power spectrum (magnitude squared, in dB).
pub fn power_spectrum_db(fft_output: &[Complex]) -> Vec<f64> {
    fft_output.iter()
        .map(|c| {
            let power = c.re * c.re + c.im * c.im;
            if power > 1e-20 { 10.0 * power.log10() } else { -200.0 }
        })
        .collect()
}

/// Find the dominant frequency in an audio buffer.
pub fn detect_pitch(buffer: &AudioBuffer) -> Option<f64> {
    let n = buffer.num_frames().next_power_of_two();
    if n < 4 { return None; }

    let mut input: Vec<Complex> = (0..n)
        .map(|i| {
            let sample = if i < buffer.num_frames() { buffer.get(i, 0) } else { 0.0 };
            // Apply Hann window to reduce spectral leakage
            let window = 0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos());
            Complex::new(sample * window, 0.0)
        })
        .collect();

    let spectrum = fft(&input);
    let magnitudes = magnitude_spectrum(&spectrum);

    // Find the peak in the first half of the spectrum (positive frequencies)
    let half = n / 2;
    let mut max_mag = 0.0;
    let mut max_bin = 1; // skip DC (bin 0)
    for i in 1..half {
        if magnitudes[i] > max_mag {
            max_mag = magnitudes[i];
            max_bin = i;
        }
    }

    let freq = max_bin as f64 * buffer.sample_rate as f64 / n as f64;
    if max_mag > 1e-6 { Some(freq) } else { None }
}

// ============================================================================
// PART 6: AUDIO EFFECTS
// ============================================================================
// Audio effects transform audio buffers to create richer, more interesting
// sounds. Each effect is a real, working DSP algorithm.

/// Delay effect: repeats the signal after a fixed time interval.
pub struct Delay {
    buffer: Vec<f64>,
    write_pos: usize,
    delay_samples: usize,
    feedback: f64,   // 0.0 to <1.0 (how much of the delayed signal feeds back)
    mix: f64,        // 0.0 = dry only, 1.0 = wet only
}

impl Delay {
    pub fn new(delay_ms: f64, feedback: f64, mix: f64, sample_rate: u32) -> Self {
        let delay_samples = (delay_ms / 1000.0 * sample_rate as f64) as usize;
        Delay {
            buffer: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
            delay_samples,
            feedback: feedback.clamp(0.0, 0.99),
            mix: mix.clamp(0.0, 1.0),
        }
    }

    pub fn process_sample(&mut self, input: f64) -> f64 {
        let read_pos = (self.write_pos + self.buffer.len() - self.delay_samples) % self.buffer.len();
        let delayed = self.buffer[read_pos];
        self.buffer[self.write_pos] = input + delayed * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        input * (1.0 - self.mix) + delayed * self.mix
    }

    pub fn process(&mut self, buffer: &mut AudioBuffer) {
        for i in 0..buffer.samples.len() {
            buffer.samples[i] = self.process_sample(buffer.samples[i]);
        }
    }
}

/// Schroeder reverb: simulates room acoustics using a network of
/// comb filters and all-pass filters. This is the classic algorithmic
/// reverb design from Manfred Schroeder (1961).
pub struct SchroederReverb {
    comb_filters: Vec<CombFilter>,
    allpass_filters: Vec<AllPassFilter>,
    mix: f64,
}

struct CombFilter {
    buffer: Vec<f64>,
    pos: usize,
    feedback: f64,
}

impl CombFilter {
    fn new(delay_samples: usize, feedback: f64) -> Self {
        CombFilter {
            buffer: vec![0.0; delay_samples.max(1)],
            pos: 0,
            feedback,
        }
    }

    fn process(&mut self, input: f64) -> f64 {
        let output = self.buffer[self.pos];
        self.buffer[self.pos] = input + output * self.feedback;
        self.pos = (self.pos + 1) % self.buffer.len();
        output
    }
}

struct AllPassFilter {
    buffer: Vec<f64>,
    pos: usize,
    feedback: f64,
}

impl AllPassFilter {
    fn new(delay_samples: usize, feedback: f64) -> Self {
        AllPassFilter {
            buffer: vec![0.0; delay_samples.max(1)],
            pos: 0,
            feedback,
        }
    }

    fn process(&mut self, input: f64) -> f64 {
        let delayed = self.buffer[self.pos];
        let output = -input + delayed;
        self.buffer[self.pos] = input + delayed * self.feedback;
        self.pos = (self.pos + 1) % self.buffer.len();
        output
    }
}

impl SchroederReverb {
    pub fn new(room_size: f64, damping: f64, mix: f64, sample_rate: u32) -> Self {
        // Classic Schroeder reverb: 4 parallel comb filters + 2 series all-pass filters
        // Delay times are chosen to be mutually prime to maximize echo density
        let scale = room_size * sample_rate as f64 / 44100.0;
        let fb = 0.7 + damping * 0.28; // feedback based on damping

        let comb_filters = vec![
            CombFilter::new((1557.0 * scale / 44100.0 * sample_rate as f64) as usize, fb),
            CombFilter::new((1617.0 * scale / 44100.0 * sample_rate as f64) as usize, fb - 0.02),
            CombFilter::new((1491.0 * scale / 44100.0 * sample_rate as f64) as usize, fb - 0.04),
            CombFilter::new((1422.0 * scale / 44100.0 * sample_rate as f64) as usize, fb - 0.06),
        ];

        let allpass_filters = vec![
            AllPassFilter::new((225.0 * scale / 44100.0 * sample_rate as f64) as usize, 0.5),
            AllPassFilter::new((556.0 * scale / 44100.0 * sample_rate as f64) as usize, 0.5),
        ];

        SchroederReverb { comb_filters, allpass_filters, mix: mix.clamp(0.0, 1.0) }
    }

    pub fn process_sample(&mut self, input: f64) -> f64 {
        // Sum the outputs of all comb filters (parallel)
        let mut comb_sum = 0.0;
        for comb in &mut self.comb_filters {
            comb_sum += comb.process(input);
        }
        comb_sum /= self.comb_filters.len() as f64;

        // Pass through all-pass filters (series)
        let mut output = comb_sum;
        for ap in &mut self.allpass_filters {
            output = ap.process(output);
        }

        input * (1.0 - self.mix) + output * self.mix
    }

    pub fn process(&mut self, buffer: &mut AudioBuffer) {
        for i in 0..buffer.samples.len() {
            buffer.samples[i] = self.process_sample(buffer.samples[i]);
        }
    }
}

/// Chorus effect: slight pitch modulation creates a thicker sound.
pub struct Chorus {
    buffer: Vec<f64>,
    write_pos: usize,
    rate: f64,       // LFO rate in Hz
    depth: f64,      // modulation depth in samples
    mix: f64,
    phase: f64,
    sample_rate: u32,
}

impl Chorus {
    pub fn new(rate: f64, depth_ms: f64, mix: f64, sample_rate: u32) -> Self {
        let max_delay = (depth_ms * 2.0 / 1000.0 * sample_rate as f64) as usize + 1024;
        Chorus {
            buffer: vec![0.0; max_delay],
            write_pos: 0,
            rate,
            depth: depth_ms / 1000.0 * sample_rate as f64,
            mix: mix.clamp(0.0, 1.0),
            phase: 0.0,
            sample_rate,
        }
    }

    pub fn process_sample(&mut self, input: f64) -> f64 {
        self.buffer[self.write_pos] = input;

        // LFO modulates the delay time
        let lfo = self.phase.sin();
        let delay = self.depth * (1.0 + lfo) / 2.0 + 1.0;
        let read_pos = self.write_pos as f64 - delay;
        let read_pos = if read_pos < 0.0 { read_pos + self.buffer.len() as f64 } else { read_pos };

        // Linear interpolation for fractional delay
        let idx0 = read_pos.floor() as usize % self.buffer.len();
        let idx1 = (idx0 + 1) % self.buffer.len();
        let frac = read_pos.fract();
        let delayed = self.buffer[idx0] * (1.0 - frac) + self.buffer[idx1] * frac;

        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        self.phase += 2.0 * PI * self.rate / self.sample_rate as f64;
        if self.phase >= 2.0 * PI { self.phase -= 2.0 * PI; }

        input * (1.0 - self.mix) + delayed * self.mix
    }

    pub fn process(&mut self, buffer: &mut AudioBuffer) {
        for i in 0..buffer.samples.len() {
            buffer.samples[i] = self.process_sample(buffer.samples[i]);
        }
    }
}

/// Distortion/overdrive effect: soft clipping using tanh saturation.
pub fn distortion(buffer: &mut AudioBuffer, drive: f64) {
    let gain = 1.0 + drive * 10.0; // drive 0-1 maps to gain 1-11
    for s in &mut buffer.samples {
        *s = (*s * gain).tanh();
    }
}

/// Dynamic range compressor: reduces the volume of loud sounds.
pub struct Compressor {
    threshold_db: f64,
    ratio: f64,         // e.g., 4.0 means 4:1 compression
    attack_coeff: f64,
    release_coeff: f64,
    envelope: f64,
    makeup_gain: f64,
}

impl Compressor {
    pub fn new(threshold_db: f64, ratio: f64, attack_ms: f64,
               release_ms: f64, makeup_db: f64, sample_rate: u32) -> Self
    {
        let attack_coeff = (-1.0 / (attack_ms / 1000.0 * sample_rate as f64)).exp();
        let release_coeff = (-1.0 / (release_ms / 1000.0 * sample_rate as f64)).exp();
        let makeup_gain = 10.0f64.powf(makeup_db / 20.0);
        Compressor {
            threshold_db, ratio, attack_coeff, release_coeff,
            envelope: 0.0, makeup_gain,
        }
    }

    pub fn process_sample(&mut self, input: f64) -> f64 {
        let abs_input = input.abs();
        let input_db = if abs_input > 1e-6 { 20.0 * abs_input.log10() } else { -120.0 };

        // Envelope follower
        let coeff = if abs_input > self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope = coeff * self.envelope + (1.0 - coeff) * abs_input;

        // Compute gain reduction
        let env_db = if self.envelope > 1e-6 { 20.0 * self.envelope.log10() } else { -120.0 };
        let gain_db = if env_db > self.threshold_db {
            self.threshold_db + (env_db - self.threshold_db) / self.ratio - env_db
        } else {
            0.0
        };

        let gain = 10.0f64.powf(gain_db / 20.0) * self.makeup_gain;
        input * gain
    }

    pub fn process(&mut self, buffer: &mut AudioBuffer) {
        for i in 0..buffer.samples.len() {
            buffer.samples[i] = self.process_sample(buffer.samples[i]);
        }
    }
}

/// Ring modulation: multiplies two signals, creating sum and difference
/// frequencies. Classic "robot voice" / "bell" effect.
pub fn ring_modulate(buffer: &mut AudioBuffer, carrier_freq: f64) {
    let sr = buffer.sample_rate as f64;
    for i in 0..buffer.samples.len() {
        let t = i as f64 / sr;
        let carrier = (2.0 * PI * carrier_freq * t).sin();
        buffer.samples[i] *= carrier;
    }
}

// ============================================================================
// PART 7: MIDI
// ============================================================================
// MIDI (Musical Instrument Digital Interface) is the standard protocol
// for communicating musical events between instruments, controllers, and
// software. A MIDI message is 1-3 bytes:
//   - Status byte (high bit set): identifies the message type and channel
//   - Data bytes (high bit clear): 0-127 values
//
// The most important messages:
//   - Note On (0x90):  start playing a note (velocity = how hard)
//   - Note Off (0x80): stop playing a note
//   - CC (0xB0):       continuous controller (mod wheel, expression, etc.)
//   - Pitch Bend (0xE0): 14-bit pitch bend value

#[derive(Debug, Clone, PartialEq)]
pub enum MidiMessage {
    NoteOn { channel: u8, note: u8, velocity: u8 },
    NoteOff { channel: u8, note: u8, velocity: u8 },
    ControlChange { channel: u8, controller: u8, value: u8 },
    ProgramChange { channel: u8, program: u8 },
    PitchBend { channel: u8, value: i16 },  // -8192 to +8191
    Aftertouch { channel: u8, note: u8, pressure: u8 },
    ChannelPressure { channel: u8, pressure: u8 },
    SystemExclusive(Vec<u8>),
    Unknown(Vec<u8>),
}

impl MidiMessage {
    /// Parse a MIDI message from raw bytes.
    pub fn parse(data: &[u8]) -> Result<MidiMessage, String> {
        if data.is_empty() {
            return Err("Empty MIDI message".to_string());
        }

        let status = data[0];
        let msg_type = status & 0xF0;
        let channel = status & 0x0F;

        match msg_type {
            0x90 => {
                if data.len() < 3 { return Err("Note On requires 3 bytes".to_string()); }
                if data[2] == 0 {
                    // Velocity 0 is treated as Note Off (common optimization)
                    Ok(MidiMessage::NoteOff { channel, note: data[1], velocity: 0 })
                } else {
                    Ok(MidiMessage::NoteOn { channel, note: data[1], velocity: data[2] })
                }
            }
            0x80 => {
                if data.len() < 3 { return Err("Note Off requires 3 bytes".to_string()); }
                Ok(MidiMessage::NoteOff { channel, note: data[1], velocity: data[2] })
            }
            0xB0 => {
                if data.len() < 3 { return Err("CC requires 3 bytes".to_string()); }
                Ok(MidiMessage::ControlChange { channel, controller: data[1], value: data[2] })
            }
            0xC0 => {
                if data.len() < 2 { return Err("Program Change requires 2 bytes".to_string()); }
                Ok(MidiMessage::ProgramChange { channel, program: data[1] })
            }
            0xE0 => {
                if data.len() < 3 { return Err("Pitch Bend requires 3 bytes".to_string()); }
                let value = ((data[2] as i16) << 7 | data[1] as i16) - 8192;
                Ok(MidiMessage::PitchBend { channel, value })
            }
            0xA0 => {
                if data.len() < 3 { return Err("Aftertouch requires 3 bytes".to_string()); }
                Ok(MidiMessage::Aftertouch { channel, note: data[1], pressure: data[2] })
            }
            0xD0 => {
                if data.len() < 2 { return Err("Channel Pressure requires 2 bytes".to_string()); }
                Ok(MidiMessage::ChannelPressure { channel, pressure: data[1] })
            }
            0xF0 if status == 0xF0 => {
                // System Exclusive: read until 0xF7
                let end = data.iter().position(|&b| b == 0xF7).unwrap_or(data.len());
                Ok(MidiMessage::SystemExclusive(data[1..end].to_vec()))
            }
            _ => Ok(MidiMessage::Unknown(data.to_vec())),
        }
    }

    /// Serialize a MIDI message to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            MidiMessage::NoteOn { channel, note, velocity } =>
                vec![0x90 | channel, *note, *velocity],
            MidiMessage::NoteOff { channel, note, velocity } =>
                vec![0x80 | channel, *note, *velocity],
            MidiMessage::ControlChange { channel, controller, value } =>
                vec![0xB0 | channel, *controller, *value],
            MidiMessage::ProgramChange { channel, program } =>
                vec![0xC0 | channel, *program],
            MidiMessage::PitchBend { channel, value } => {
                let raw = (*value + 8192) as u16;
                vec![0xE0 | channel, (raw & 0x7F) as u8, ((raw >> 7) & 0x7F) as u8]
            }
            MidiMessage::Aftertouch { channel, note, pressure } =>
                vec![0xA0 | channel, *note, *pressure],
            MidiMessage::ChannelPressure { channel, pressure } =>
                vec![0xD0 | channel, *pressure],
            MidiMessage::SystemExclusive(data) => {
                let mut bytes = vec![0xF0];
                bytes.extend(data);
                bytes.push(0xF7);
                bytes
            }
            MidiMessage::Unknown(data) => data.clone(),
        }
    }

    /// Convert a MIDI note number to frequency (A4 = 440 Hz = note 69).
    pub fn note_to_freq(note: u8) -> f64 {
        440.0 * 2.0f64.powf((note as f64 - 69.0) / 12.0)
    }

    /// Convert a MIDI note number to a name (e.g., 60 → "C4").
    pub fn note_to_name(note: u8) -> String {
        let names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
        let octave = (note / 12) as i32 - 1;
        let name = names[(note % 12) as usize];
        format!("{}{}", name, octave)
    }
}

// ============================================================================
// PART 8: AUDIO CODECS
// ============================================================================
// Audio codecs encode and decode audio samples for storage and transmission.
// PCM (Pulse-Code Modulation) is the raw, uncompressed format used by
// WAV files. μ-law and A-law are logarithmic companding algorithms that
// compress 16-bit samples into 8 bits with perceptually better quality
// than linear quantization — they're used in telephone systems worldwide
// (μ-law in North America/Japan, A-law in Europe/rest of world).

/// Encode a floating-point sample (-1.0 to 1.0) to 16-bit PCM.
pub fn encode_pcm16(sample: f64) -> i16 {
    let clamped = sample.clamp(-1.0, 1.0);
    (clamped * 32767.0) as i16
}

/// Decode a 16-bit PCM sample to floating-point.
pub fn decode_pcm16(sample: i16) -> f64 {
    sample as f64 / 32768.0
}

/// μ-law encode: compress 16-bit linear to 8-bit logarithmic.
/// This is ITU-T G.711 μ-law, used in North American telephone networks.
pub fn encode_ulaw(sample: i16) -> u8 {
    const MU: f64 = 255.0;
    const BIAS: i16 = 0x84;
    const CLIP: i16 = 32635;

    let mut sample = sample;
    let sign = if sample < 0 { sample = -sample; 0x80u8 } else { 0x00u8 };
    if sample > CLIP { sample = CLIP; }
    sample += BIAS;

    let exponent = match sample {
        s if s >= 0x4000 => 7,
        s if s >= 0x2000 => 6,
        s if s >= 0x1000 => 5,
        s if s >= 0x0800 => 4,
        s if s >= 0x0400 => 3,
        s if s >= 0x0200 => 2,
        s if s >= 0x0100 => 1,
        _ => 0,
    };

    let mantissa = (sample >> (exponent + 3)) & 0x0F;
    let encoded = !(sign | ((exponent as u8) << 4) | mantissa as u8);
    encoded
}

/// μ-law decode: expand 8-bit logarithmic to 16-bit linear.
pub fn decode_ulaw(encoded: u8) -> i16 {
    let encoded = !encoded;
    let sign = encoded & 0x80;
    let exponent = ((encoded >> 4) & 0x07) as i16;
    let mantissa = (encoded & 0x0F) as i16;
    let mut sample = ((mantissa << 3) + 0x84) << exponent;
    sample -= 0x84;
    if sign != 0 { -sample } else { sample }
}

/// A-law encode: ITU-T G.711 A-law, used in European telephone networks.
pub fn encode_alaw(sample: i16) -> u8 {
    let mut sample = sample;
    let sign = if sample < 0 { sample = -sample; 0x80u8 } else { 0x00u8 };

    let (exponent, mantissa) = if sample < 256 {
        (0u8, (sample >> 4) as u8)
    } else {
        let exp = match sample {
            s if s >= 0x2000 => 7,
            s if s >= 0x1000 => 6,
            s if s >= 0x0800 => 5,
            s if s >= 0x0400 => 4,
            s if s >= 0x0200 => 3,
            s if s >= 0x0100 => 2,
            _ => 1,
        };
        (exp, ((sample >> (exp + 3)) & 0x0F) as u8)
    };

    let encoded = sign | (exponent << 4) | mantissa;
    encoded ^ 0x55 // A-law uses alternating bit inversion
}

/// A-law decode.
pub fn decode_alaw(encoded: u8) -> i16 {
    let encoded = encoded ^ 0x55;
    let sign = encoded & 0x80;
    let exponent = ((encoded >> 4) & 0x07) as i16;
    let mantissa = (encoded & 0x0F) as i16;

    let sample = if exponent == 0 {
        (mantissa << 4) + 8
    } else {
        ((mantissa << 3) + 0x84) << exponent
    };

    if sign != 0 { -sample } else { sample }
}

// ============================================================================
// PART 9: IMAGE BUFFER AND PIXEL OPERATIONS
// ============================================================================
// An image is a 2D grid of pixels. Each pixel has 1-4 channels:
//   - Grayscale: 1 channel (luminance)
//   - RGB: 3 channels (red, green, blue)
//   - RGBA: 4 channels (red, green, blue, alpha/transparency)
//
// We store pixels as u8 (0-255) for memory efficiency, but convert to
// f64 (0.0-1.0) for processing that requires precision.

#[derive(Debug, Clone)]
pub struct Image {
    pub pixels: Vec<u8>,  // row-major, channel-interleaved
    pub width: u32,
    pub height: u32,
    pub channels: u8,     // 1=gray, 3=RGB, 4=RGBA
}

impl Image {
    pub fn new(width: u32, height: u32, channels: u8) -> Self {
        let size = width as usize * height as usize * channels as usize;
        Image { pixels: vec![0; size], width, height, channels }
    }

    pub fn from_pixels(pixels: Vec<u8>, width: u32, height: u32, channels: u8) -> Self {
        Image { pixels, width, height, channels }
    }

    /// Get the pixel value at (x, y) for a specific channel.
    pub fn get(&self, x: u32, y: u32, ch: u8) -> u8 {
        if x >= self.width || y >= self.height || ch >= self.channels { return 0; }
        let idx = (y as usize * self.width as usize + x as usize) * self.channels as usize + ch as usize;
        self.pixels[idx]
    }

    /// Set the pixel value at (x, y) for a specific channel.
    pub fn set(&mut self, x: u32, y: u32, ch: u8, value: u8) {
        if x >= self.width || y >= self.height || ch >= self.channels { return; }
        let idx = (y as usize * self.width as usize + x as usize) * self.channels as usize + ch as usize;
        self.pixels[idx] = value;
    }

    /// Get all channels of a pixel as a vector.
    pub fn get_pixel(&self, x: u32, y: u32) -> Vec<u8> {
        (0..self.channels).map(|ch| self.get(x, y, ch)).collect()
    }

    /// Set all channels of a pixel.
    pub fn set_pixel(&mut self, x: u32, y: u32, pixel: &[u8]) {
        for (ch, &val) in pixel.iter().enumerate() {
            self.set(x, y, ch as u8, val);
        }
    }

    /// Fill the entire image with a color.
    pub fn fill(&mut self, color: &[u8]) {
        for y in 0..self.height {
            for x in 0..self.width {
                self.set_pixel(x, y, color);
            }
        }
    }
}

// ============================================================================
// PART 10: IMAGE CONVOLUTION
// ============================================================================
// Convolution is the fundamental operation of image processing. A small
// matrix (kernel) slides over the image, and at each position, the output
// pixel is the weighted sum of the input pixels covered by the kernel.
// Different kernels produce different effects.

/// Apply a convolution kernel to an image. Returns a new image.
pub fn convolve(image: &Image, kernel: &[f64], kernel_size: usize) -> Image {
    let mut output = Image::new(image.width, image.height, image.channels);
    let half = kernel_size as i32 / 2;

    for y in 0..image.height {
        for x in 0..image.width {
            for ch in 0..image.channels {
                let mut sum = 0.0;
                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let ix = x as i32 + kx as i32 - half;
                        let iy = y as i32 + ky as i32 - half;
                        // Clamp to image bounds (replicate border pixels)
                        let ix = ix.clamp(0, image.width as i32 - 1) as u32;
                        let iy = iy.clamp(0, image.height as i32 - 1) as u32;
                        let pixel = image.get(ix, iy, ch) as f64;
                        sum += pixel * kernel[ky * kernel_size + kx];
                    }
                }
                output.set(x, y, ch, sum.clamp(0.0, 255.0) as u8);
            }
        }
    }
    output
}

/// Predefined convolution kernels.
pub fn kernel_blur_3x3() -> Vec<f64> {
    vec![1.0/9.0; 9]
}

pub fn kernel_gaussian_3x3() -> Vec<f64> {
    vec![
        1.0/16.0, 2.0/16.0, 1.0/16.0,
        2.0/16.0, 4.0/16.0, 2.0/16.0,
        1.0/16.0, 2.0/16.0, 1.0/16.0,
    ]
}

pub fn kernel_sharpen() -> Vec<f64> {
    vec![
         0.0, -1.0,  0.0,
        -1.0,  5.0, -1.0,
         0.0, -1.0,  0.0,
    ]
}

pub fn kernel_edge_detect() -> Vec<f64> {
    vec![
        -1.0, -1.0, -1.0,
        -1.0,  8.0, -1.0,
        -1.0, -1.0, -1.0,
    ]
}

pub fn kernel_emboss() -> Vec<f64> {
    vec![
        -2.0, -1.0, 0.0,
        -1.0,  1.0, 1.0,
         0.0,  1.0, 2.0,
    ]
}

pub fn kernel_sobel_x() -> Vec<f64> {
    vec![
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0,
    ]
}

pub fn kernel_sobel_y() -> Vec<f64> {
    vec![
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0,
    ]
}

// ============================================================================
// PART 11: COLOR SPACE CONVERSION
// ============================================================================
// Different color spaces represent colors in different ways:
//   - RGB: additive mixing of red, green, blue (displays)
//   - HSV: hue (color), saturation (purity), value (brightness) — intuitive
//   - YUV: luminance (Y) + chrominance (UV) — used in video compression
//     because the eye is more sensitive to brightness than color, so UV
//     channels can be compressed more aggressively

/// Convert RGB (0-255 each) to HSV (H: 0-360°, S: 0-1, V: 0-1).
pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let r = r as f64 / 255.0;
    let g = g as f64 / 255.0;
    let b = b as f64 / 255.0;

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;
    let s = if max == 0.0 { 0.0 } else { delta / max };

    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };
    (h, s, v)
}

/// Convert HSV to RGB.
pub fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 60.0 { (c, x, 0.0) }
        else if h < 120.0 { (x, c, 0.0) }
        else if h < 180.0 { (0.0, c, x) }
        else if h < 240.0 { (0.0, x, c) }
        else if h < 300.0 { (x, 0.0, c) }
        else { (c, 0.0, x) };

    (((r + m) * 255.0) as u8, ((g + m) * 255.0) as u8, ((b + m) * 255.0) as u8)
}

/// Convert RGB to YUV (BT.601 standard, used in SDTV).
pub fn rgb_to_yuv(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let r = r as f64 / 255.0;
    let g = g as f64 / 255.0;
    let b = b as f64 / 255.0;
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let u = -0.14713 * r - 0.28886 * g + 0.436 * b;
    let v = 0.615 * r - 0.51499 * g - 0.10001 * b;
    (y, u, v)
}

/// Convert YUV to RGB.
pub fn yuv_to_rgb(y: f64, u: f64, v: f64) -> (u8, u8, u8) {
    let r = (y + 1.13983 * v).clamp(0.0, 1.0);
    let g = (y - 0.39465 * u - 0.58060 * v).clamp(0.0, 1.0);
    let b = (y + 2.03211 * u).clamp(0.0, 1.0);
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Convert an RGB image to grayscale using luminance weights.
pub fn to_grayscale(image: &Image) -> Image {
    if image.channels < 3 { return image.clone(); }
    let mut gray = Image::new(image.width, image.height, 1);
    for y in 0..image.height {
        for x in 0..image.width {
            let r = image.get(x, y, 0) as f64;
            let g = image.get(x, y, 1) as f64;
            let b = image.get(x, y, 2) as f64;
            // ITU-R BT.601 luminance weights
            let lum = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
            gray.set(x, y, 0, lum);
        }
    }
    gray
}

// ============================================================================
// PART 12: IMAGE TRANSFORMATIONS
// ============================================================================

/// Flip an image horizontally (mirror).
pub fn flip_horizontal(image: &Image) -> Image {
    let mut flipped = Image::new(image.width, image.height, image.channels);
    for y in 0..image.height {
        for x in 0..image.width {
            let src_x = image.width - 1 - x;
            for ch in 0..image.channels {
                flipped.set(x, y, ch, image.get(src_x, y, ch));
            }
        }
    }
    flipped
}

/// Flip an image vertically.
pub fn flip_vertical(image: &Image) -> Image {
    let mut flipped = Image::new(image.width, image.height, image.channels);
    for y in 0..image.height {
        let src_y = image.height - 1 - y;
        for x in 0..image.width {
            for ch in 0..image.channels {
                flipped.set(x, y, ch, image.get(x, src_y, ch));
            }
        }
    }
    flipped
}

/// Crop a rectangular region from an image.
pub fn crop(image: &Image, x: u32, y: u32, w: u32, h: u32) -> Image {
    let w = w.min(image.width.saturating_sub(x));
    let h = h.min(image.height.saturating_sub(y));
    let mut cropped = Image::new(w, h, image.channels);
    for dy in 0..h {
        for dx in 0..w {
            for ch in 0..image.channels {
                cropped.set(dx, dy, ch, image.get(x + dx, y + dy, ch));
            }
        }
    }
    cropped
}

/// Resize an image using nearest-neighbor interpolation (fast, blocky).
pub fn resize_nearest(image: &Image, new_width: u32, new_height: u32) -> Image {
    let mut resized = Image::new(new_width, new_height, image.channels);
    let x_ratio = image.width as f64 / new_width as f64;
    let y_ratio = image.height as f64 / new_height as f64;

    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = (x as f64 * x_ratio) as u32;
            let src_y = (y as f64 * y_ratio) as u32;
            for ch in 0..image.channels {
                resized.set(x, y, ch, image.get(src_x, src_y, ch));
            }
        }
    }
    resized
}

/// Resize an image using bilinear interpolation (smoother, slower).
pub fn resize_bilinear(image: &Image, new_width: u32, new_height: u32) -> Image {
    let mut resized = Image::new(new_width, new_height, image.channels);
    let x_ratio = (image.width as f64 - 1.0) / new_width.max(1) as f64;
    let y_ratio = (image.height as f64 - 1.0) / new_height.max(1) as f64;

    for y in 0..new_height {
        for x in 0..new_width {
            let gx = x as f64 * x_ratio;
            let gy = y as f64 * y_ratio;
            let gxi = gx.floor() as u32;
            let gyi = gy.floor() as u32;
            let fx = gx - gxi as f64;
            let fy = gy - gyi as f64;

            let x0 = gxi.min(image.width - 1);
            let x1 = (gxi + 1).min(image.width - 1);
            let y0 = gyi.min(image.height - 1);
            let y1 = (gyi + 1).min(image.height - 1);

            for ch in 0..image.channels {
                let tl = image.get(x0, y0, ch) as f64;
                let tr = image.get(x1, y0, ch) as f64;
                let bl = image.get(x0, y1, ch) as f64;
                let br = image.get(x1, y1, ch) as f64;

                let top = tl * (1.0 - fx) + tr * fx;
                let bot = bl * (1.0 - fx) + br * fx;
                let val = top * (1.0 - fy) + bot * fy;
                resized.set(x, y, ch, val.clamp(0.0, 255.0) as u8);
            }
        }
    }
    resized
}

/// Rotate an image 90 degrees clockwise.
pub fn rotate_90_cw(image: &Image) -> Image {
    let mut rotated = Image::new(image.height, image.width, image.channels);
    for y in 0..image.height {
        for x in 0..image.width {
            for ch in 0..image.channels {
                rotated.set(image.height - 1 - y, x, ch, image.get(x, y, ch));
            }
        }
    }
    rotated
}

// ============================================================================
// PART 13: HISTOGRAM
// ============================================================================
// A histogram counts the frequency of each intensity level (0-255) in
// an image. Histogram equalization redistributes intensities to use the
// full range, improving contrast in washed-out or dark images.

/// Compute the histogram of a single-channel image (or one channel of a multi-channel image).
pub fn compute_histogram(image: &Image, channel: u8) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for y in 0..image.height {
        for x in 0..image.width {
            let val = image.get(x, y, channel) as usize;
            hist[val] += 1;
        }
    }
    hist
}

/// Apply histogram equalization to improve contrast.
pub fn histogram_equalize(image: &Image) -> Image {
    let mut output = image.clone();
    let total_pixels = (image.width * image.height) as f64;

    for ch in 0..image.channels {
        let hist = compute_histogram(image, ch);
        // Compute cumulative distribution function
        let mut cdf = [0u32; 256];
        cdf[0] = hist[0];
        for i in 1..256 {
            cdf[i] = cdf[i - 1] + hist[i];
        }
        // Find minimum non-zero CDF value
        let cdf_min = cdf.iter().find(|&&c| c > 0).copied().unwrap_or(0);

        // Map each pixel through the equalization function
        let mut map = [0u8; 256];
        for i in 0..256 {
            if total_pixels > cdf_min as f64 {
                map[i] = ((cdf[i] as f64 - cdf_min as f64) / (total_pixels - cdf_min as f64) * 255.0)
                    .clamp(0.0, 255.0) as u8;
            }
        }

        for y in 0..image.height {
            for x in 0..image.width {
                let val = image.get(x, y, ch) as usize;
                output.set(x, y, ch, map[val]);
            }
        }
    }
    output
}

// ============================================================================
// PART 14: VIDEO FRAME BUFFER AND OPERATIONS
// ============================================================================
// Video is a sequence of images (frames) displayed at a fixed rate.
// Video processing operates on sequences of frames, detecting changes
// between frames (motion detection) and creating transitions.

/// A video: a sequence of frames at a fixed frame rate.
#[derive(Debug, Clone)]
pub struct Video {
    pub frames: Vec<Image>,
    pub fps: f64,
    pub width: u32,
    pub height: u32,
}

impl Video {
    pub fn new(width: u32, height: u32, fps: f64) -> Self {
        Video { frames: Vec::new(), fps, width, height }
    }

    pub fn add_frame(&mut self, frame: Image) {
        self.frames.push(frame);
    }

    pub fn duration(&self) -> f64 {
        self.frames.len() as f64 / self.fps
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

/// Detect motion between two frames by computing the absolute difference.
/// Returns a grayscale image where bright pixels indicate motion.
pub fn detect_motion(frame_a: &Image, frame_b: &Image, threshold: u8) -> Image {
    let w = frame_a.width.min(frame_b.width);
    let h = frame_a.height.min(frame_b.height);
    let mut motion = Image::new(w, h, 1);

    for y in 0..h {
        for x in 0..w {
            let mut diff = 0u32;
            let ch_count = frame_a.channels.min(frame_b.channels).min(3);
            for ch in 0..ch_count {
                let a = frame_a.get(x, y, ch) as i32;
                let b = frame_b.get(x, y, ch) as i32;
                diff += (a - b).unsigned_abs();
            }
            let avg_diff = (diff / ch_count as u32) as u8;
            let value = if avg_diff > threshold { 255 } else { 0 };
            motion.set(x, y, 0, value);
        }
    }
    motion
}

/// Crossfade transition between two frames.
/// t = 0.0 → entirely frame_a, t = 1.0 → entirely frame_b.
pub fn crossfade(frame_a: &Image, frame_b: &Image, t: f64) -> Image {
    let w = frame_a.width.min(frame_b.width);
    let h = frame_a.height.min(frame_b.height);
    let ch = frame_a.channels.min(frame_b.channels);
    let mut result = Image::new(w, h, ch);
    let t = t.clamp(0.0, 1.0);

    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let a = frame_a.get(x, y, c) as f64;
                let b = frame_b.get(x, y, c) as f64;
                let blended = a * (1.0 - t) + b * t;
                result.set(x, y, c, blended.clamp(0.0, 255.0) as u8);
            }
        }
    }
    result
}

/// Horizontal wipe transition.
/// t = 0.0 → entirely frame_a, t = 1.0 → entirely frame_b.
pub fn wipe_horizontal(frame_a: &Image, frame_b: &Image, t: f64) -> Image {
    let w = frame_a.width.min(frame_b.width);
    let h = frame_a.height.min(frame_b.height);
    let ch = frame_a.channels.min(frame_b.channels);
    let mut result = Image::new(w, h, ch);
    let split = (w as f64 * t.clamp(0.0, 1.0)) as u32;

    for y in 0..h {
        for x in 0..w {
            let src = if x < split { frame_b } else { frame_a };
            for c in 0..ch {
                result.set(x, y, c, src.get(x, y, c));
            }
        }
    }
    result
}

// ============================================================================
// PART 15: SPATIAL AUDIO
// ============================================================================
// Spatial audio creates the illusion that sounds come from specific
// positions in space. The two main techniques:
//   - Stereo panning: adjusts the balance between left and right channels
//   - Distance attenuation: reduces volume with distance (inverse square law)

/// Apply stereo panning to a mono audio buffer.
/// pan: -1.0 = full left, 0.0 = center, 1.0 = full right
/// Uses constant-power panning (sin/cos law) for consistent perceived loudness.
pub fn stereo_pan(mono: &AudioBuffer, pan: f64) -> AudioBuffer {
    let pan = pan.clamp(-1.0, 1.0);
    // Constant-power panning: left = cos(θ), right = sin(θ)
    // where θ = (pan + 1) * π/4 maps [-1, 1] to [0, π/2]
    let theta = (pan + 1.0) * PI / 4.0;
    let left_gain = theta.cos();
    let right_gain = theta.sin();

    let mut stereo = AudioBuffer::new(mono.sample_rate, 2);
    for frame in 0..mono.num_frames() {
        let sample = mono.get(frame, 0);
        stereo.push_frame(&[sample * left_gain, sample * right_gain]);
    }
    stereo
}

/// Apply distance-based attenuation.
/// Uses the inverse distance model: gain = 1 / (1 + distance * rolloff)
pub fn distance_attenuation(buffer: &mut AudioBuffer, distance: f64, rolloff: f64) {
    let gain = 1.0 / (1.0 + distance * rolloff);
    buffer.apply_gain(gain);
}

// ============================================================================
// PART 16: TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Oscillator Tests ---

    #[test]
    fn test_sine_oscillator_frequency() {
        // Generate 1 second of 440 Hz sine at 44100 Hz sample rate
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, 1.0, 44100);
        let buffer = osc.generate(1.0);

        assert_eq!(buffer.num_frames(), 44100);
        assert_eq!(buffer.sample_rate, 44100);

        // Check that the signal crosses zero approximately 2*440 = 880 times
        // (each cycle has 2 zero crossings)
        let mut crossings = 0;
        for i in 1..buffer.num_frames() {
            let prev = buffer.get(i - 1, 0);
            let curr = buffer.get(i, 0);
            if (prev >= 0.0 && curr < 0.0) || (prev < 0.0 && curr >= 0.0) {
                crossings += 1;
            }
        }
        // Allow 1% tolerance
        assert!((crossings as f64 - 880.0).abs() < 10.0,
                "Expected ~880 zero crossings, got {}", crossings);
    }

    #[test]
    fn test_square_wave_range() {
        let mut osc = Oscillator::new(Waveform::Square, 100.0, 1.0, 44100);
        let buffer = osc.generate(0.1);
        for i in 0..buffer.num_frames() {
            let s = buffer.get(i, 0);
            assert!(s == 1.0 || s == -1.0, "Square wave sample should be ±1, got {}", s);
        }
    }

    #[test]
    fn test_waveform_amplitude() {
        for wf in [Waveform::Sine, Waveform::Square, Waveform::Sawtooth, Waveform::Triangle] {
            let mut osc = Oscillator::new(wf, 440.0, 0.5, 44100);
            let buffer = osc.generate(0.1);
            for i in 0..buffer.num_frames() {
                let s = buffer.get(i, 0).abs();
                assert!(s <= 0.51, "{:?} amplitude exceeded: {}", wf, s);
            }
        }
    }

    #[test]
    fn test_silence_is_silent() {
        let buf = AudioBuffer::silence(44100, 2, 0.5);
        assert_eq!(buf.num_frames(), 22050);
        assert!(buf.rms().iter().all(|&r| r == 0.0));
    }

    // --- ADSR Envelope Tests ---

    #[test]
    fn test_adsr_envelope_shape() {
        let env = AdsrEnvelope::new(0.1, 0.1, 0.5, 0.2, 1000);
        let curve = env.generate(0.5); // 0.5s gate time

        // At t=0, should be 0 (start of attack)
        assert!(curve[0] < 0.01);

        // At t=0.1s (end of attack), should be near 1.0
        let attack_end = 100; // sample 100 = 0.1s at 1000 Hz
        assert!((curve[attack_end] - 1.0).abs() < 0.02,
                "Attack end: {}", curve[attack_end]);

        // At t=0.2s (end of decay), should be near sustain level (0.5)
        let decay_end = 200;
        assert!((curve[decay_end] - 0.5).abs() < 0.05,
                "Decay end: {}", curve[decay_end]);

        // During sustain (t=0.3s), should be near 0.5
        let sustain_mid = 300;
        assert!((curve[sustain_mid] - 0.5).abs() < 0.02,
                "Sustain: {}", curve[sustain_mid]);

        // At end of release (t=0.7s), should be near 0
        let release_end = curve.len() - 1;
        assert!(curve[release_end] < 0.02,
                "Release end: {}", curve[release_end]);
    }

    // --- Biquad Filter Tests ---

    #[test]
    fn test_lowpass_attenuates_high_frequencies() {
        // Generate a high-frequency signal (10 kHz) and filter with a 1 kHz low-pass
        let mut osc = Oscillator::new(Waveform::Sine, 10000.0, 1.0, 44100);
        let mut buffer = osc.generate(0.1);
        let peak_before = buffer.peak()[0];

        let mut filter = BiquadFilter::design(FilterType::LowPass, 1000.0, 0.707, 0.0, 44100);
        filter.process(&mut buffer);

        let peak_after = buffer.peak()[0];
        // The 10 kHz signal should be significantly attenuated by a 1 kHz low-pass
        assert!(peak_after < peak_before * 0.2,
                "Low-pass filter didn't attenuate: before={}, after={}", peak_before, peak_after);
    }

    #[test]
    fn test_highpass_attenuates_low_frequencies() {
        let mut osc = Oscillator::new(Waveform::Sine, 100.0, 1.0, 44100);
        let mut buffer = osc.generate(0.1);
        let peak_before = buffer.peak()[0];

        let mut filter = BiquadFilter::design(FilterType::HighPass, 5000.0, 0.707, 0.0, 44100);
        filter.process(&mut buffer);

        let peak_after = buffer.peak()[0];
        assert!(peak_after < peak_before * 0.2,
                "High-pass filter didn't attenuate: before={}, after={}", peak_before, peak_after);
    }

    // --- FFT Tests ---

    #[test]
    fn test_fft_pure_sine() {
        let n = 256;
        let freq = 10.0;
        let sample_rate = 256.0;

        let input: Vec<Complex> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                Complex::new((2.0 * PI * freq * t).sin(), 0.0)
            })
            .collect();

        let spectrum = fft(&input);
        let magnitudes = magnitude_spectrum(&spectrum);

        // The peak should be at bin 10 (frequency 10 Hz at 256 Hz sample rate)
        let mut max_bin = 0;
        let mut max_mag = 0.0;
        for i in 1..n / 2 {
            if magnitudes[i] > max_mag {
                max_mag = magnitudes[i];
                max_bin = i;
            }
        }
        assert_eq!(max_bin, 10, "Peak should be at bin 10, got bin {}", max_bin);
    }

    #[test]
    fn test_fft_inverse_roundtrip() {
        let input: Vec<Complex> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];

        let spectrum = fft(&input);
        let recovered = ifft(&spectrum);

        for i in 0..input.len() {
            assert!((recovered[i].re - input[i].re).abs() < 1e-10,
                    "Mismatch at {}: {} vs {}", i, recovered[i].re, input[i].re);
            assert!(recovered[i].im.abs() < 1e-10,
                    "Imaginary part should be ~0: {}", recovered[i].im);
        }
    }

    #[test]
    fn test_fft_dc_component() {
        // Constant signal: FFT should have all energy in bin 0 (DC)
        let n = 8;
        let input: Vec<Complex> = vec![Complex::new(3.0, 0.0); n];
        let spectrum = fft(&input);

        assert!((spectrum[0].re - 24.0).abs() < 1e-10, "DC = {}", spectrum[0].re); // 3.0 * 8
        for i in 1..n {
            assert!(spectrum[i].magnitude() < 1e-10,
                    "Non-DC bin {} should be ~0: {}", i, spectrum[i].magnitude());
        }
    }

    // --- MIDI Tests ---

    #[test]
    fn test_midi_note_on_parse() {
        let msg = MidiMessage::parse(&[0x90, 60, 100]).unwrap();
        assert_eq!(msg, MidiMessage::NoteOn { channel: 0, note: 60, velocity: 100 });
    }

    #[test]
    fn test_midi_note_on_velocity_zero_is_note_off() {
        let msg = MidiMessage::parse(&[0x91, 64, 0]).unwrap();
        assert_eq!(msg, MidiMessage::NoteOff { channel: 1, note: 64, velocity: 0 });
    }

    #[test]
    fn test_midi_pitch_bend() {
        // Center (0): LSB=0, MSB=64
        let msg = MidiMessage::parse(&[0xE0, 0x00, 0x40]).unwrap();
        assert_eq!(msg, MidiMessage::PitchBend { channel: 0, value: 0 });

        // Max up: LSB=127, MSB=127
        let msg = MidiMessage::parse(&[0xE0, 0x7F, 0x7F]).unwrap();
        assert_eq!(msg, MidiMessage::PitchBend { channel: 0, value: 8191 });
    }

    #[test]
    fn test_midi_roundtrip() {
        let messages = vec![
            MidiMessage::NoteOn { channel: 0, note: 60, velocity: 100 },
            MidiMessage::NoteOff { channel: 3, note: 72, velocity: 64 },
            MidiMessage::ControlChange { channel: 0, controller: 1, value: 127 },
            MidiMessage::ProgramChange { channel: 9, program: 42 },
        ];

        for msg in &messages {
            let bytes = msg.to_bytes();
            let parsed = MidiMessage::parse(&bytes).unwrap();
            assert_eq!(*msg, parsed, "Roundtrip failed for {:?}", msg);
        }
    }

    #[test]
    fn test_midi_note_to_freq() {
        // A4 = 440 Hz = MIDI note 69
        assert!((MidiMessage::note_to_freq(69) - 440.0).abs() < 0.01);
        // Middle C = MIDI note 60
        assert!((MidiMessage::note_to_freq(60) - 261.63).abs() < 0.1);
        // A3 = 220 Hz = MIDI note 57
        assert!((MidiMessage::note_to_freq(57) - 220.0).abs() < 0.01);
    }

    #[test]
    fn test_midi_note_names() {
        assert_eq!(MidiMessage::note_to_name(60), "C4");
        assert_eq!(MidiMessage::note_to_name(69), "A4");
        assert_eq!(MidiMessage::note_to_name(0), "C-1");
        assert_eq!(MidiMessage::note_to_name(127), "G9");
    }

    // --- Codec Tests ---

    #[test]
    fn test_pcm16_roundtrip() {
        let samples = [0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.75];
        for &s in &samples {
            let encoded = encode_pcm16(s);
            let decoded = decode_pcm16(encoded);
            assert!((decoded - s).abs() < 0.001, "PCM16 roundtrip: {} → {} → {}", s, encoded, decoded);
        }
    }

    #[test]
    fn test_ulaw_roundtrip_preserves_sign() {
        for &sample in &[0i16, 1000, -1000, 10000, -10000, 32000, -32000] {
            let encoded = encode_ulaw(sample);
            let decoded = decode_ulaw(encoded);
            // μ-law is lossy but should preserve sign and approximate magnitude
            if sample == 0 {
                assert!(decoded.abs() < 100, "Zero: decoded to {}", decoded);
            } else {
                assert_eq!(sample.signum(), decoded.signum(),
                           "Sign mismatch: {} → {}", sample, decoded);
            }
        }
    }

    // --- Image Tests ---

    #[test]
    fn test_image_pixel_operations() {
        let mut img = Image::new(4, 4, 3);
        img.set_pixel(1, 2, &[255, 128, 0]);
        assert_eq!(img.get(1, 2, 0), 255);
        assert_eq!(img.get(1, 2, 1), 128);
        assert_eq!(img.get(1, 2, 2), 0);
    }

    #[test]
    fn test_grayscale_conversion() {
        let mut img = Image::new(2, 2, 3);
        // Pure red pixel
        img.set_pixel(0, 0, &[255, 0, 0]);
        // Pure green pixel
        img.set_pixel(1, 0, &[0, 255, 0]);

        let gray = to_grayscale(&img);
        assert_eq!(gray.channels, 1);
        // Red luminance: 0.299 * 255 ≈ 76
        assert!((gray.get(0, 0, 0) as i32 - 76).abs() <= 1);
        // Green luminance: 0.587 * 255 ≈ 150
        assert!((gray.get(1, 0, 0) as i32 - 150).abs() <= 1);
    }

    #[test]
    fn test_color_space_rgb_hsv_roundtrip() {
        let test_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                           (128, 128, 128), (255, 255, 0), (0, 255, 255)];
        for (r, g, b) in &test_colors {
            let (h, s, v) = rgb_to_hsv(*r, *g, *b);
            let (r2, g2, b2) = hsv_to_rgb(h, s, v);
            assert!((*r as i32 - r2 as i32).abs() <= 1, "R mismatch: {} vs {}", r, r2);
            assert!((*g as i32 - g2 as i32).abs() <= 1, "G mismatch: {} vs {}", g, g2);
            assert!((*b as i32 - b2 as i32).abs() <= 1, "B mismatch: {} vs {}", b, b2);
        }
    }

    #[test]
    fn test_color_space_rgb_yuv_roundtrip() {
        for (r, g, b) in &[(255u8, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)] {
            let (y, u, v) = rgb_to_yuv(*r, *g, *b);
            let (r2, g2, b2) = yuv_to_rgb(y, u, v);
            assert!((*r as i32 - r2 as i32).abs() <= 2, "R mismatch: {} vs {}", r, r2);
            assert!((*g as i32 - g2 as i32).abs() <= 2, "G mismatch: {} vs {}", g, g2);
            assert!((*b as i32 - b2 as i32).abs() <= 2, "B mismatch: {} vs {}", b, b2);
        }
    }

    #[test]
    fn test_image_flip_horizontal() {
        let mut img = Image::new(3, 1, 1);
        img.set(0, 0, 0, 10);
        img.set(1, 0, 0, 20);
        img.set(2, 0, 0, 30);

        let flipped = flip_horizontal(&img);
        assert_eq!(flipped.get(0, 0, 0), 30);
        assert_eq!(flipped.get(1, 0, 0), 20);
        assert_eq!(flipped.get(2, 0, 0), 10);
    }

    #[test]
    fn test_image_crop() {
        let mut img = Image::new(4, 4, 1);
        for y in 0..4 {
            for x in 0..4 {
                img.set(x, y, 0, (y * 4 + x) as u8);
            }
        }
        let cropped = crop(&img, 1, 1, 2, 2);
        assert_eq!(cropped.width, 2);
        assert_eq!(cropped.height, 2);
        assert_eq!(cropped.get(0, 0, 0), 5);  // original (1,1) = 1*4+1 = 5
        assert_eq!(cropped.get(1, 0, 0), 6);  // original (2,1) = 1*4+2 = 6
        assert_eq!(cropped.get(0, 1, 0), 9);  // original (1,2) = 2*4+1 = 9
        assert_eq!(cropped.get(1, 1, 0), 10); // original (2,2) = 2*4+2 = 10
    }

    #[test]
    fn test_image_resize_nearest() {
        let mut img = Image::new(2, 2, 1);
        img.set(0, 0, 0, 10);
        img.set(1, 0, 0, 20);
        img.set(0, 1, 0, 30);
        img.set(1, 1, 0, 40);

        let resized = resize_nearest(&img, 4, 4);
        assert_eq!(resized.width, 4);
        assert_eq!(resized.height, 4);
        // Top-left 2x2 block should all be 10
        assert_eq!(resized.get(0, 0, 0), 10);
        assert_eq!(resized.get(1, 0, 0), 10);
        assert_eq!(resized.get(0, 1, 0), 10);
    }

    #[test]
    fn test_image_rotate_90() {
        let mut img = Image::new(3, 2, 1);
        // Fill with unique values to verify rotation
        img.set(0, 0, 0, 1);
        img.set(1, 0, 0, 2);
        img.set(2, 0, 0, 3);
        img.set(0, 1, 0, 4);
        img.set(1, 1, 0, 5);
        img.set(2, 1, 0, 6);

        let rotated = rotate_90_cw(&img);
        assert_eq!(rotated.width, 2);   // height becomes width
        assert_eq!(rotated.height, 3);  // width becomes height
        // After 90° CW rotation: top-left corner should be bottom-left of original
        assert_eq!(rotated.get(0, 0, 0), 4);
        assert_eq!(rotated.get(1, 0, 0), 1);
    }

    #[test]
    fn test_convolution_identity() {
        // Identity kernel: center = 1, rest = 0
        let kernel = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let mut img = Image::new(4, 4, 1);
        img.set(2, 2, 0, 200);
        img.set(1, 1, 0, 100);

        let result = convolve(&img, &kernel, 3);
        assert_eq!(result.get(2, 2, 0), 200);
        assert_eq!(result.get(1, 1, 0), 100);
    }

    #[test]
    fn test_histogram() {
        let mut img = Image::new(4, 4, 1);
        // Set all pixels to 128
        for y in 0..4 { for x in 0..4 { img.set(x, y, 0, 128); } }
        // Change one pixel to 0
        img.set(0, 0, 0, 0);

        let hist = compute_histogram(&img, 0);
        assert_eq!(hist[128], 15);  // 16 - 1 = 15 pixels at 128
        assert_eq!(hist[0], 1);     // 1 pixel at 0
        assert_eq!(hist[255], 0);   // no pixels at 255
    }

    // --- Video Tests ---

    #[test]
    fn test_motion_detection() {
        let mut frame_a = Image::new(4, 4, 1);
        let mut frame_b = Image::new(4, 4, 1);

        // frame_a: all zeros
        // frame_b: some pixels changed
        frame_b.set(1, 1, 0, 200);
        frame_b.set(2, 2, 0, 150);

        let motion = detect_motion(&frame_a, &frame_b, 50);
        assert_eq!(motion.get(1, 1, 0), 255); // motion detected
        assert_eq!(motion.get(2, 2, 0), 255); // motion detected
        assert_eq!(motion.get(0, 0, 0), 0);   // no motion
    }

    #[test]
    fn test_crossfade() {
        let mut frame_a = Image::new(2, 2, 1);
        let mut frame_b = Image::new(2, 2, 1);
        frame_a.fill(&[0]);
        frame_b.fill(&[200]);

        let mid = crossfade(&frame_a, &frame_b, 0.5);
        assert_eq!(mid.get(0, 0, 0), 100); // halfway between 0 and 200

        let full_a = crossfade(&frame_a, &frame_b, 0.0);
        assert_eq!(full_a.get(0, 0, 0), 0);

        let full_b = crossfade(&frame_a, &frame_b, 1.0);
        assert_eq!(full_b.get(0, 0, 0), 200);
    }

    // --- Spatial Audio Tests ---

    #[test]
    fn test_stereo_pan_center() {
        let mut mono = AudioBuffer::new(44100, 1);
        mono.push_frame(&[1.0]);

        let stereo = stereo_pan(&mono, 0.0); // center
        assert_eq!(stereo.channels, 2);
        let left = stereo.get(0, 0);
        let right = stereo.get(0, 1);
        // At center, both channels should be equal and non-zero
        assert!((left - right).abs() < 0.01,
                "Center pan should be equal: L={}, R={}", left, right);
        assert!(left > 0.5, "Center should have significant level: {}", left);
    }

    #[test]
    fn test_stereo_pan_hard_left() {
        let mut mono = AudioBuffer::new(44100, 1);
        mono.push_frame(&[1.0]);

        let stereo = stereo_pan(&mono, -1.0); // hard left
        let left = stereo.get(0, 0);
        let right = stereo.get(0, 1);
        assert!(left > 0.9, "Hard left should be loud: {}", left);
        assert!(right < 0.01, "Hard left right should be silent: {}", right);
    }

    #[test]
    fn test_distance_attenuation() {
        let mut buf = AudioBuffer::from_samples(vec![1.0], 44100, 1);
        distance_attenuation(&mut buf, 9.0, 1.0);
        // gain = 1 / (1 + 9*1) = 0.1
        assert!((buf.samples[0] - 0.1).abs() < 0.001);
    }

    // --- Audio Effect Tests ---

    #[test]
    fn test_delay_wet_dry_mix() {
        let mut delay = Delay::new(10.0, 0.0, 1.0, 44100);
        // First sample with full wet mix: should output 0 (nothing delayed yet)
        let out = delay.process_sample(1.0);
        assert!(out.abs() < 0.01, "First output should be ~0 (no delayed signal yet)");
    }

    #[test]
    fn test_distortion_soft_clip() {
        let mut buf = AudioBuffer::from_samples(vec![0.5, -0.5, 2.0, -2.0], 44100, 1);
        distortion(&mut buf, 1.0);
        // tanh soft-clips: all values should be in [-1, 1]
        for s in &buf.samples {
            assert!(s.abs() <= 1.0, "Distortion output out of range: {}", s);
        }
        // High input should be compressed
        assert!(buf.samples[2] < 2.0);
    }

    #[test]
    fn test_ring_modulation() {
        let mut buf = AudioBuffer::from_samples(vec![1.0; 100], 44100, 1);
        ring_modulate(&mut buf, 1000.0);
        // Ring mod multiplies by carrier — output should oscillate
        let has_positive = buf.samples.iter().any(|&s| s > 0.5);
        let has_negative = buf.samples.iter().any(|&s| s < -0.5);
        assert!(has_positive && has_negative, "Ring mod should create oscillation");
    }

    // --- Audio Buffer Tests ---

    #[test]
    fn test_audio_buffer_rms_and_peak() {
        // DC signal at 0.5
        let buf = AudioBuffer::from_samples(vec![0.5; 100], 44100, 1);
        let rms = buf.rms();
        assert!((rms[0] - 0.5).abs() < 0.001);
        assert_eq!(buf.peak(), vec![0.5]);
    }

    #[test]
    fn test_audio_buffer_mix() {
        let mut a = AudioBuffer::from_samples(vec![0.5; 4], 44100, 1);
        let b = AudioBuffer::from_samples(vec![0.3; 4], 44100, 1);
        a.mix(&b, 1.0);
        assert!((a.samples[0] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_audio_buffer_clip() {
        let mut buf = AudioBuffer::from_samples(vec![1.5, -1.5, 0.5], 44100, 1);
        buf.clip();
        assert_eq!(buf.samples, vec![1.0, -1.0, 0.5]);
    }

    // --- Pitch Detection Test ---

    #[test]
    fn test_pitch_detection() {
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, 1.0, 44100);
        let buffer = osc.generate(0.1); // 100ms of 440 Hz

        let detected = detect_pitch(&buffer);
        assert!(detected.is_some());
        let freq = detected.unwrap();
        // Allow ~5 Hz tolerance due to FFT bin resolution
        assert!((freq - 440.0).abs() < 10.0,
                "Detected {} Hz, expected ~440 Hz", freq);
    }
}
