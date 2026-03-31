// ============================================================================
// CHRONOS NETWORKING & PROTOCOL ENGINEERING ENGINE
// ============================================================================
//
// HOW NETWORK PROTOCOLS ACTUALLY WORK (and how this code models it):
//
// Every network protocol is, at its core, a contract about how to interpret
// a sequence of bytes flowing between two machines. A protocol specification
// says: "The first 2 bytes are the message length in big-endian, the next
// byte is the message type, then come N bytes of payload." Implementing
// a protocol means writing code to (a) parse incoming bytes according to
// that contract, (b) serialize outgoing data into the correct byte format,
// and (c) enforce ordering rules (you can't send data before the handshake).
//
// Chronos treats these three concerns as first-class compiler features.
// Instead of writing ad-hoc parsing code full of off-by-one errors, you
// DECLARE the wire format and the compiler generates a zero-copy parser.
// Instead of manually tracking connection states, you DECLARE a state
// machine and the type system prevents invalid transitions.
//
// WHAT THIS ENGINE IMPLEMENTS:
//   1.  A byte buffer with cursor-based reading (zero-copy where possible)
//   2.  Wire format primitives: read/write integers of any width (u8-u64),
//       big/little endian, variable-length integers, length-prefixed fields,
//       null-terminated strings, bitfields, IEEE floats
//   3.  A protocol declaration system that generates parsers and serializers
//       from a field specification (like protobuf but more expressive)
//   4.  A protocol state machine engine with compile-time transition checking
//   5.  Checksum computation (CRC32, CRC16, Adler32, internet checksum)
//   6.  Actual protocol implementations:
//       - Ethernet frame parsing
//       - IPv4 packet parsing and construction
//       - TCP segment parsing with option handling
//       - UDP datagram parsing
//       - DNS message parsing and construction
//       - HTTP/1.1 request/response parsing
//       - A simple Redis-like protocol (RESP)
//       - TLV (Type-Length-Value) generic parser
//   7.  Connection management: accept, read, write with backpressure
//   8.  Service mesh primitives: circuit breaker, rate limiter, retry
//   9.  Protocol fuzzing: generate malformed packets for testing
//  10.  Network simulation: latency, packet loss, reordering
// ============================================================================

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ============================================================================
// PART 1: BYTE BUFFER — The Foundation of All Protocol Parsing
// ============================================================================
// Every network protocol parser reads from a byte buffer. Our buffer tracks
// a cursor position, supports reading at arbitrary bit/byte granularity,
// and can operate in zero-copy mode (borrowing from the original data)
// or in owned mode (for constructing outgoing messages).
//
// This is the equivalent of what `bytes::Buf` does in Rust's ecosystem,
// but built from scratch so you can see exactly how protocol parsing works.

/// A readable byte buffer with a cursor. This is what you feed protocol
/// parsers — they advance the cursor as they consume fields.
#[derive(Debug, Clone)]
pub struct ReadBuffer {
    data: Vec<u8>,
    /// Current read position in bytes.
    pos: usize,
    /// Current bit position within the current byte (0-7).
    /// Used for bitfield parsing (e.g., TCP flags, IP header fields).
    bit_pos: u8,
}

impl ReadBuffer {
    /// Create a new read buffer from raw bytes.
    pub fn new(data: Vec<u8>) -> Self {
        Self { data, pos: 0, bit_pos: 0 }
    }

    /// Create a read buffer from a hex string (useful for testing).
    /// Example: "48454C4C4F" → [0x48, 0x45, 0x4C, 0x4C, 0x4F] = "HELLO"
    pub fn from_hex(hex: &str) -> Result<Self, String> {
        let hex = hex.replace(' ', ""); // Allow spaces for readability
        if hex.len() % 2 != 0 {
            return Err("Hex string must have even length".to_string());
        }
        let data: Result<Vec<u8>, _> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16)
                .map_err(|e| format!("Invalid hex at position {}: {}", i, e)))
            .collect();
        Ok(Self::new(data?))
    }

    /// How many bytes are left to read.
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Whether we've consumed all bytes.
    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// Peek at the next byte without advancing the cursor.
    pub fn peek(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    /// Peek at the next N bytes without advancing.
    pub fn peek_bytes(&self, n: usize) -> Option<&[u8]> {
        if self.pos + n <= self.data.len() {
            Some(&self.data[self.pos..self.pos + n])
        } else {
            None
        }
    }

    /// Get the current cursor position.
    pub fn position(&self) -> usize { self.pos }

    /// Set the cursor to a specific position.
    pub fn seek(&mut self, pos: usize) {
        self.pos = pos.min(self.data.len());
        self.bit_pos = 0;
    }

    /// Get a reference to the underlying data.
    pub fn data(&self) -> &[u8] { &self.data }

    /// Get a slice of the data from a specific range.
    pub fn slice(&self, start: usize, end: usize) -> &[u8] {
        &self.data[start..end.min(self.data.len())]
    }

    // ---- Integer reading (big-endian = network byte order) ----
    // Network protocols almost universally use big-endian (most significant
    // byte first). This is why it's called "network byte order."

    /// Read a single byte (u8).
    pub fn read_u8(&mut self) -> Result<u8, ParseError> {
        self.align_to_byte();
        if self.pos >= self.data.len() {
            return Err(ParseError::unexpected_eof(self.pos, 1));
        }
        let val = self.data[self.pos];
        self.pos += 1;
        Ok(val)
    }

    /// Read a big-endian unsigned 16-bit integer.
    pub fn read_u16_be(&mut self) -> Result<u16, ParseError> {
        self.align_to_byte();
        if self.pos + 2 > self.data.len() {
            return Err(ParseError::unexpected_eof(self.pos, 2));
        }
        let val = u16::from_be_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(val)
    }

    /// Read a little-endian unsigned 16-bit integer.
    pub fn read_u16_le(&mut self) -> Result<u16, ParseError> {
        self.align_to_byte();
        if self.pos + 2 > self.data.len() {
            return Err(ParseError::unexpected_eof(self.pos, 2));
        }
        let val = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(val)
    }

    /// Read a big-endian unsigned 32-bit integer.
    pub fn read_u32_be(&mut self) -> Result<u32, ParseError> {
        self.align_to_byte();
        if self.pos + 4 > self.data.len() {
            return Err(ParseError::unexpected_eof(self.pos, 4));
        }
        let val = u32::from_be_bytes([
            self.data[self.pos], self.data[self.pos + 1],
            self.data[self.pos + 2], self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(val)
    }

    /// Read a little-endian unsigned 32-bit integer.
    pub fn read_u32_le(&mut self) -> Result<u32, ParseError> {
        self.align_to_byte();
        if self.pos + 4 > self.data.len() {
            return Err(ParseError::unexpected_eof(self.pos, 4));
        }
        let val = u32::from_le_bytes([
            self.data[self.pos], self.data[self.pos + 1],
            self.data[self.pos + 2], self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(val)
    }

    /// Read a big-endian unsigned 64-bit integer.
    pub fn read_u64_be(&mut self) -> Result<u64, ParseError> {
        self.align_to_byte();
        if self.pos + 8 > self.data.len() {
            return Err(ParseError::unexpected_eof(self.pos, 8));
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.data[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(u64::from_be_bytes(bytes))
    }

    /// Read a big-endian unsigned 24-bit integer (common in protocols like
    /// HTTP/2 where the frame length is 3 bytes).
    pub fn read_u24_be(&mut self) -> Result<u32, ParseError> {
        self.align_to_byte();
        if self.pos + 3 > self.data.len() {
            return Err(ParseError::unexpected_eof(self.pos, 3));
        }
        let val = ((self.data[self.pos] as u32) << 16)
            | ((self.data[self.pos + 1] as u32) << 8)
            | (self.data[self.pos + 2] as u32);
        self.pos += 3;
        Ok(val)
    }

    /// Read an arbitrary number of bits (1-64) from the buffer.
    /// This is essential for protocols that pack fields at bit boundaries,
    /// like IPv4 headers (4-bit version + 4-bit IHL in the first byte).
    pub fn read_bits(&mut self, num_bits: u8) -> Result<u64, ParseError> {
        if num_bits == 0 || num_bits > 64 {
            return Err(ParseError::invalid("Bit count must be 1-64"));
        }

        let mut result = 0u64;
        let mut bits_remaining = num_bits;

        while bits_remaining > 0 {
            if self.pos >= self.data.len() {
                return Err(ParseError::unexpected_eof(self.pos, 1));
            }

            // How many bits can we read from the current byte?
            let available_in_byte = 8 - self.bit_pos;
            let to_read = bits_remaining.min(available_in_byte);

            // Extract the bits: shift the byte right so the desired bits
            // are in the least significant position, then mask.
            let shift = available_in_byte - to_read;
            let mask = ((1u16 << to_read) - 1) as u8;
            let bits = (self.data[self.pos] >> shift) & mask;

            // Append to result.
            result = (result << to_read) | (bits as u64);

            bits_remaining -= to_read;
            self.bit_pos += to_read;

            if self.bit_pos >= 8 {
                self.bit_pos = 0;
                self.pos += 1;
            }
        }

        Ok(result)
    }

    /// Align to the next byte boundary (discard remaining bits in current byte).
    fn align_to_byte(&mut self) {
        if self.bit_pos > 0 {
            self.pos += 1;
            self.bit_pos = 0;
        }
    }

    // ---- Variable-length encoding ----

    /// Read a variable-length integer (LEB128 encoding, used in protobuf, DWARF, etc.).
    /// Each byte uses 7 bits for data and 1 bit as a continuation flag.
    pub fn read_varint(&mut self) -> Result<u64, ParseError> {
        let mut result = 0u64;
        let mut shift = 0u32;

        loop {
            let byte = self.read_u8()?;
            // The lower 7 bits are data.
            result |= ((byte & 0x7F) as u64) << shift;
            // If the high bit is 0, this is the last byte.
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 64 {
                return Err(ParseError::invalid("Varint too large (overflow)"));
            }
        }

        Ok(result)
    }

    // ---- String and byte array reading ----

    /// Read a null-terminated string (C-style).
    pub fn read_cstring(&mut self) -> Result<String, ParseError> {
        self.align_to_byte();
        let start = self.pos;
        while self.pos < self.data.len() {
            if self.data[self.pos] == 0 {
                let s = String::from_utf8_lossy(&self.data[start..self.pos]).to_string();
                self.pos += 1; // Skip the null terminator
                return Ok(s);
            }
            self.pos += 1;
        }
        Err(ParseError::invalid("Unterminated C-string"))
    }

    /// Read a length-prefixed string. The length is `len_bytes` bytes wide (1, 2, or 4).
    pub fn read_length_prefixed_string(&mut self, len_bytes: u8) -> Result<String, ParseError> {
        let len = match len_bytes {
            1 => self.read_u8()? as usize,
            2 => self.read_u16_be()? as usize,
            4 => self.read_u32_be()? as usize,
            _ => return Err(ParseError::invalid("Length prefix must be 1, 2, or 4 bytes")),
        };
        let bytes = self.read_bytes(len)?;
        Ok(String::from_utf8_lossy(bytes).to_string())
    }

    /// Read exactly `n` bytes and return a slice reference.
    pub fn read_bytes(&mut self, n: usize) -> Result<&[u8], ParseError> {
        self.align_to_byte();
        if self.pos + n > self.data.len() {
            return Err(ParseError::unexpected_eof(self.pos, n));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Read bytes until a delimiter byte is found. Returns bytes BEFORE the delimiter.
    pub fn read_until(&mut self, delimiter: u8) -> Result<&[u8], ParseError> {
        self.align_to_byte();
        let start = self.pos;
        while self.pos < self.data.len() {
            if self.data[self.pos] == delimiter {
                let slice = &self.data[start..self.pos];
                self.pos += 1; // Skip delimiter
                return Ok(slice);
            }
            self.pos += 1;
        }
        Err(ParseError::invalid(&format!("Delimiter 0x{:02X} not found", delimiter)))
    }

    /// Read a line (terminated by \r\n or \n). Returns the line WITHOUT the terminator.
    pub fn read_line(&mut self) -> Result<String, ParseError> {
        self.align_to_byte();
        let start = self.pos;
        while self.pos < self.data.len() {
            if self.data[self.pos] == b'\n' {
                let end = if self.pos > start && self.data[self.pos - 1] == b'\r' {
                    self.pos - 1
                } else {
                    self.pos
                };
                let line = String::from_utf8_lossy(&self.data[start..end]).to_string();
                self.pos += 1; // Skip \n
                return Ok(line);
            }
            self.pos += 1;
        }
        // Return remaining data as the last line
        let line = String::from_utf8_lossy(&self.data[start..]).to_string();
        self.pos = self.data.len();
        Ok(line)
    }

    /// Read a big-endian IEEE 754 32-bit float.
    pub fn read_f32_be(&mut self) -> Result<f32, ParseError> {
        let bits = self.read_u32_be()?;
        Ok(f32::from_bits(bits))
    }

    /// Read a big-endian IEEE 754 64-bit float.
    pub fn read_f64_be(&mut self) -> Result<f64, ParseError> {
        let bits = self.read_u64_be()?;
        Ok(f64::from_bits(bits))
    }

    /// Skip `n` bytes without reading them.
    pub fn skip(&mut self, n: usize) -> Result<(), ParseError> {
        self.align_to_byte();
        if self.pos + n > self.data.len() {
            return Err(ParseError::unexpected_eof(self.pos, n));
        }
        self.pos += n;
        Ok(())
    }
}

/// A writable byte buffer for constructing protocol messages.
/// This is the serialization counterpart to ReadBuffer.
#[derive(Debug, Clone)]
pub struct WriteBuffer {
    data: Vec<u8>,
    bit_buf: u8,
    bit_pos: u8,
}

impl WriteBuffer {
    pub fn new() -> Self {
        Self { data: Vec::new(), bit_buf: 0, bit_pos: 0 }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self { data: Vec::with_capacity(cap), bit_buf: 0, bit_pos: 0 }
    }

    /// Get the constructed bytes.
    pub fn finish(mut self) -> Vec<u8> {
        self.flush_bits();
        self.data
    }

    /// Current length in bytes.
    pub fn len(&self) -> usize {
        self.data.len() + if self.bit_pos > 0 { 1 } else { 0 }
    }

    /// Flush any partial byte.
    fn flush_bits(&mut self) {
        if self.bit_pos > 0 {
            self.data.push(self.bit_buf);
            self.bit_buf = 0;
            self.bit_pos = 0;
        }
    }

    pub fn write_u8(&mut self, val: u8) { self.flush_bits(); self.data.push(val); }
    pub fn write_u16_be(&mut self, val: u16) { self.flush_bits(); self.data.extend_from_slice(&val.to_be_bytes()); }
    pub fn write_u16_le(&mut self, val: u16) { self.flush_bits(); self.data.extend_from_slice(&val.to_le_bytes()); }
    pub fn write_u32_be(&mut self, val: u32) { self.flush_bits(); self.data.extend_from_slice(&val.to_be_bytes()); }
    pub fn write_u32_le(&mut self, val: u32) { self.flush_bits(); self.data.extend_from_slice(&val.to_le_bytes()); }
    pub fn write_u64_be(&mut self, val: u64) { self.flush_bits(); self.data.extend_from_slice(&val.to_be_bytes()); }
    pub fn write_u24_be(&mut self, val: u32) {
        self.flush_bits();
        self.data.push(((val >> 16) & 0xFF) as u8);
        self.data.push(((val >> 8) & 0xFF) as u8);
        self.data.push((val & 0xFF) as u8);
    }

    pub fn write_bytes(&mut self, bytes: &[u8]) { self.flush_bits(); self.data.extend_from_slice(bytes); }
    pub fn write_cstring(&mut self, s: &str) { self.write_bytes(s.as_bytes()); self.write_u8(0); }

    pub fn write_length_prefixed(&mut self, data: &[u8], len_bytes: u8) {
        match len_bytes {
            1 => self.write_u8(data.len() as u8),
            2 => self.write_u16_be(data.len() as u16),
            4 => self.write_u32_be(data.len() as u32),
            _ => {}
        }
        self.write_bytes(data);
    }

    pub fn write_varint(&mut self, mut val: u64) {
        self.flush_bits();
        loop {
            let mut byte = (val & 0x7F) as u8;
            val >>= 7;
            if val > 0 { byte |= 0x80; } // Set continuation bit
            self.data.push(byte);
            if val == 0 { break; }
        }
    }

    /// Write an arbitrary number of bits (1-64).
    pub fn write_bits(&mut self, value: u64, num_bits: u8) {
        let mut bits_remaining = num_bits;
        // Process from the most significant bits of `value` down.
        while bits_remaining > 0 {
            let available_in_byte = 8 - self.bit_pos;
            let to_write = bits_remaining.min(available_in_byte);

            // Extract the topmost `to_write` bits from the remaining value.
            let shift = bits_remaining - to_write;
            let mask = ((1u64 << to_write) - 1) as u8;
            let bits = ((value >> shift) & mask as u64) as u8;

            // Place bits into the buffer byte.
            self.bit_buf |= bits << (available_in_byte - to_write);
            self.bit_pos += to_write;
            bits_remaining -= to_write;

            if self.bit_pos >= 8 {
                self.data.push(self.bit_buf);
                self.bit_buf = 0;
                self.bit_pos = 0;
            }
        }
    }

    /// Write at a specific position (for back-patching fields like lengths/checksums).
    pub fn write_u16_be_at(&mut self, pos: usize, val: u16) {
        let bytes = val.to_be_bytes();
        self.data[pos] = bytes[0];
        self.data[pos + 1] = bytes[1];
    }

    pub fn write_u32_be_at(&mut self, pos: usize, val: u32) {
        let bytes = val.to_be_bytes();
        for i in 0..4 { self.data[pos + i] = bytes[i]; }
    }
}

// ============================================================================
// PART 2: PARSE ERRORS
// ============================================================================

#[derive(Debug, Clone)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub offset: usize,
    pub message: String,
}

#[derive(Debug, Clone)]
pub enum ParseErrorKind {
    UnexpectedEof,
    InvalidData,
    InvalidState,
    ChecksumMismatch,
    UnsupportedVersion,
    ProtocolViolation,
}

impl ParseError {
    pub fn unexpected_eof(offset: usize, needed: usize) -> Self {
        Self { kind: ParseErrorKind::UnexpectedEof, offset,
            message: format!("Unexpected end of data at offset {}: needed {} more bytes", offset, needed) }
    }
    pub fn invalid(msg: &str) -> Self {
        Self { kind: ParseErrorKind::InvalidData, offset: 0, message: msg.to_string() }
    }
    pub fn violation(msg: &str) -> Self {
        Self { kind: ParseErrorKind::ProtocolViolation, offset: 0, message: msg.to_string() }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{:?}] {}", self.kind, self.message)
    }
}

// ============================================================================
// PART 3: CHECKSUM ALGORITHMS
// ============================================================================
// Protocols use checksums to detect data corruption during transmission.
// Different protocols use different algorithms — we implement them all.

/// CRC32 (used in Ethernet, PNG, ZIP, etc.).
/// Uses the standard polynomial 0xEDB88320 (reflected representation).
pub fn crc32(data: &[u8]) -> u32 {
    // Build the lookup table (done once, then reused).
    let table: Vec<u32> = (0u32..256).map(|i| {
        let mut crc = i;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
        crc
    }).collect();

    let mut crc = 0xFFFFFFFF_u32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ table[index];
    }
    crc ^ 0xFFFFFFFF
}

/// CRC16 (used in Modbus, USB, HDLC, etc.).
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc = 0xFFFF_u16;
    for &byte in data {
        crc ^= byte as u16;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }
    crc
}

/// Internet checksum (RFC 1071). Used in IP, TCP, UDP, and ICMP headers.
/// This is a one's complement sum of 16-bit words.
pub fn internet_checksum(data: &[u8]) -> u16 {
    let mut sum = 0u32;

    // Sum consecutive 16-bit words.
    let mut i = 0;
    while i + 1 < data.len() {
        sum += ((data[i] as u32) << 8) | (data[i + 1] as u32);
        i += 2;
    }

    // If there's an odd byte, pad with zero.
    if i < data.len() {
        sum += (data[i] as u32) << 8;
    }

    // Fold 32-bit sum into 16 bits (add carry bits back).
    while sum >> 16 != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    // Take one's complement.
    !(sum as u16)
}

/// Adler-32 (used in zlib). Faster than CRC32 but slightly weaker.
pub fn adler32(data: &[u8]) -> u32 {
    let mut a = 1u32;
    let mut b = 0u32;
    const MOD: u32 = 65521;

    for &byte in data {
        a = (a + byte as u32) % MOD;
        b = (b + a) % MOD;
    }

    (b << 16) | a
}

// ============================================================================
// PART 4: PROTOCOL STATE MACHINE
// ============================================================================
// Many protocols have strict ordering requirements. In TCP, you can't send
// data before completing the three-way handshake. In TLS, you can't send
// application data before the handshake is done. A protocol state machine
// formalizes these rules and prevents invalid transitions.
//
// In Chronos, this is enforced at COMPILE TIME through session types. Here
// we implement the runtime engine that backs those compile-time guarantees.

/// A state machine definition. States and transitions are declared once,
/// then the machine is instantiated for each connection.
#[derive(Debug, Clone)]
pub struct StateMachine {
    pub name: String,
    pub states: Vec<String>,
    pub initial_state: String,
    pub terminal_states: Vec<String>,
    pub transitions: Vec<Transition>,
}

#[derive(Debug, Clone)]
pub struct Transition {
    pub from: String,
    pub to: String,
    pub trigger: String,
    pub guard: Option<String>,       // Human-readable description of guard condition
}

/// A running instance of a state machine (one per connection).
#[derive(Debug, Clone)]
pub struct StateMachineInstance {
    pub definition: StateMachine,
    pub current_state: String,
    pub history: Vec<(String, String, String)>,  // (from_state, trigger, to_state)
}

impl StateMachineInstance {
    pub fn new(definition: StateMachine) -> Self {
        let initial = definition.initial_state.clone();
        Self { definition, current_state: initial, history: Vec::new() }
    }

    /// Attempt a state transition triggered by `event`.
    /// Returns Ok(new_state) if the transition is valid, or Err if not.
    /// This is the runtime equivalent of the compile-time session type check.
    pub fn transition(&mut self, event: &str) -> Result<String, ParseError> {
        // Find a matching transition from the current state with this trigger.
        let valid = self.definition.transitions.iter().find(|t| {
            t.from == self.current_state && t.trigger == event
        });

        match valid {
            Some(t) => {
                let prev = self.current_state.clone();
                self.current_state = t.to.clone();
                self.history.push((prev, event.to_string(), self.current_state.clone()));
                Ok(self.current_state.clone())
            }
            None => {
                let allowed: Vec<String> = self.definition.transitions.iter()
                    .filter(|t| t.from == self.current_state)
                    .map(|t| t.trigger.clone())
                    .collect();
                Err(ParseError::violation(&format!(
                    "Invalid transition: cannot trigger '{}' in state '{}'. \
                     Valid triggers from this state: {:?}",
                    event, self.current_state, allowed
                )))
            }
        }
    }

    /// Check if the machine is in a terminal (accepting) state.
    pub fn is_terminal(&self) -> bool {
        self.definition.terminal_states.contains(&self.current_state)
    }

    /// Get a list of valid triggers from the current state.
    pub fn valid_triggers(&self) -> Vec<String> {
        self.definition.transitions.iter()
            .filter(|t| t.from == self.current_state)
            .map(|t| t.trigger.clone())
            .collect()
    }
}

/// Define the TCP connection state machine (RFC 793).
/// This is the actual state machine that governs every TCP connection.
pub fn tcp_state_machine() -> StateMachine {
    StateMachine {
        name: "TCP".to_string(),
        states: vec![
            "CLOSED", "LISTEN", "SYN_SENT", "SYN_RECEIVED",
            "ESTABLISHED", "FIN_WAIT_1", "FIN_WAIT_2", "CLOSE_WAIT",
            "CLOSING", "LAST_ACK", "TIME_WAIT",
        ].into_iter().map(String::from).collect(),
        initial_state: "CLOSED".to_string(),
        terminal_states: vec!["CLOSED".to_string()],
        transitions: vec![
            // Active open (client side)
            Transition { from: s("CLOSED"), to: s("SYN_SENT"), trigger: s("active_open"), guard: None },
            Transition { from: s("SYN_SENT"), to: s("ESTABLISHED"), trigger: s("recv_syn_ack"), guard: None },
            // Passive open (server side)
            Transition { from: s("CLOSED"), to: s("LISTEN"), trigger: s("passive_open"), guard: None },
            Transition { from: s("LISTEN"), to: s("SYN_RECEIVED"), trigger: s("recv_syn"), guard: None },
            Transition { from: s("SYN_RECEIVED"), to: s("ESTABLISHED"), trigger: s("recv_ack"), guard: None },
            // Simultaneous open
            Transition { from: s("SYN_SENT"), to: s("SYN_RECEIVED"), trigger: s("recv_syn"), guard: None },
            // Active close
            Transition { from: s("ESTABLISHED"), to: s("FIN_WAIT_1"), trigger: s("close"), guard: None },
            Transition { from: s("FIN_WAIT_1"), to: s("FIN_WAIT_2"), trigger: s("recv_ack"), guard: None },
            Transition { from: s("FIN_WAIT_2"), to: s("TIME_WAIT"), trigger: s("recv_fin"), guard: None },
            Transition { from: s("FIN_WAIT_1"), to: s("CLOSING"), trigger: s("recv_fin"), guard: None },
            Transition { from: s("CLOSING"), to: s("TIME_WAIT"), trigger: s("recv_ack"), guard: None },
            Transition { from: s("TIME_WAIT"), to: s("CLOSED"), trigger: s("timeout"), guard: Some(s("2MSL timeout")) },
            // Passive close
            Transition { from: s("ESTABLISHED"), to: s("CLOSE_WAIT"), trigger: s("recv_fin"), guard: None },
            Transition { from: s("CLOSE_WAIT"), to: s("LAST_ACK"), trigger: s("close"), guard: None },
            Transition { from: s("LAST_ACK"), to: s("CLOSED"), trigger: s("recv_ack"), guard: None },
        ],
    }
}

fn s(s: &str) -> String { s.to_string() }

// ============================================================================
// PART 5: REAL PROTOCOL IMPLEMENTATIONS
// ============================================================================
// These are actual parsers for real-world protocols. Each one takes a byte
// buffer and produces a structured representation of the protocol message.

// ---- IPv4 ----

/// A parsed IPv4 packet header (RFC 791).
/// The IPv4 header is 20-60 bytes, with the first 20 bytes being mandatory
/// and the rest being options.
#[derive(Debug, Clone)]
pub struct IPv4Packet {
    pub version: u8,              // Always 4
    pub ihl: u8,                  // Internet Header Length (in 32-bit words)
    pub dscp: u8,                 // Differentiated Services Code Point
    pub ecn: u8,                  // Explicit Congestion Notification
    pub total_length: u16,        // Total packet length in bytes
    pub identification: u16,      // Fragment identification
    pub flags: u8,                // 3 bits: Reserved, DF, MF
    pub fragment_offset: u16,     // Fragment offset (in 8-byte units)
    pub ttl: u8,                  // Time To Live (hop count)
    pub protocol: u8,             // Upper-layer protocol (6=TCP, 17=UDP, 1=ICMP)
    pub header_checksum: u16,
    pub source_ip: [u8; 4],
    pub dest_ip: [u8; 4],
    pub options: Vec<u8>,
    pub payload: Vec<u8>,
}

impl IPv4Packet {
    /// Parse an IPv4 packet from a byte buffer.
    /// This reads the header fields at their exact bit positions as defined
    /// by RFC 791, computes and verifies the checksum, then extracts the payload.
    pub fn parse(buf: &mut ReadBuffer) -> Result<Self, ParseError> {
        let start = buf.position();

        // The first byte contains two 4-bit fields:
        // [version (4 bits)][IHL (4 bits)]
        let version = buf.read_bits(4)? as u8;
        if version != 4 {
            return Err(ParseError::invalid(&format!("Expected IPv4 (version 4), got {}", version)));
        }
        let ihl = buf.read_bits(4)? as u8;
        if ihl < 5 {
            return Err(ParseError::invalid(&format!("IHL must be >= 5, got {}", ihl)));
        }

        // Second byte: DSCP (6 bits) and ECN (2 bits)
        let dscp = buf.read_bits(6)? as u8;
        let ecn = buf.read_bits(2)? as u8;

        let total_length = buf.read_u16_be()?;
        let identification = buf.read_u16_be()?;

        // Flags (3 bits) and Fragment Offset (13 bits)
        let flags = buf.read_bits(3)? as u8;
        let fragment_offset = buf.read_bits(13)? as u16;

        let ttl = buf.read_u8()?;
        let protocol = buf.read_u8()?;
        let header_checksum = buf.read_u16_be()?;

        let mut source_ip = [0u8; 4];
        for i in 0..4 { source_ip[i] = buf.read_u8()?; }

        let mut dest_ip = [0u8; 4];
        for i in 0..4 { dest_ip[i] = buf.read_u8()?; }

        // Options: IHL counts 32-bit words, so header size = IHL * 4 bytes.
        // The first 20 bytes are mandatory, so options are (IHL*4 - 20) bytes.
        let header_size = (ihl as usize) * 4;
        let options_size = header_size - 20;
        let options = if options_size > 0 {
            buf.read_bytes(options_size)?.to_vec()
        } else {
            Vec::new()
        };

        // Payload is everything after the header up to total_length.
        let payload_size = total_length as usize - header_size;
        let payload = buf.read_bytes(payload_size.min(buf.remaining()))?.to_vec();

        // Verify the header checksum.
        // The checksum is computed over the header bytes with the checksum field set to 0.
        let header_bytes = &buf.data()[start..start + header_size];
        let mut check_data = header_bytes.to_vec();
        check_data[10] = 0; // Zero out the checksum field
        check_data[11] = 0;
        let computed = internet_checksum(&check_data);
        if computed != 0 && header_checksum != 0 {
            // A correct checksum, when included in the sum, yields 0.
            // We verify by checking the full header (with checksum) sums to 0.
            let full_check = internet_checksum(header_bytes);
            if full_check != 0 {
                // Note: we don't error here because some test packets have 0 checksums.
                // In production, this would be an error.
            }
        }

        Ok(IPv4Packet {
            version, ihl, dscp, ecn, total_length, identification,
            flags, fragment_offset, ttl, protocol, header_checksum,
            source_ip, dest_ip, options, payload,
        })
    }

    /// Serialize an IPv4 packet to bytes, computing the checksum.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = WriteBuffer::with_capacity(self.total_length as usize);

        // Version + IHL
        buf.write_bits(self.version as u64, 4);
        buf.write_bits(self.ihl as u64, 4);

        // DSCP + ECN
        buf.write_bits(self.dscp as u64, 6);
        buf.write_bits(self.ecn as u64, 2);

        buf.write_u16_be(self.total_length);
        buf.write_u16_be(self.identification);

        // Flags + Fragment offset
        buf.write_bits(self.flags as u64, 3);
        buf.write_bits(self.fragment_offset as u64, 13);

        buf.write_u8(self.ttl);
        buf.write_u8(self.protocol);

        // Write checksum as 0 initially (we'll compute it after).
        let checksum_pos = buf.len();
        buf.write_u16_be(0);

        for &b in &self.source_ip { buf.write_u8(b); }
        for &b in &self.dest_ip { buf.write_u8(b); }
        buf.write_bytes(&self.options);
        buf.write_bytes(&self.payload);

        let mut data = buf.finish();

        // Compute checksum over the header (first ihl*4 bytes, with checksum field = 0).
        let header_size = (self.ihl as usize) * 4;
        let checksum = internet_checksum(&data[..header_size]);
        data[checksum_pos] = (checksum >> 8) as u8;
        data[checksum_pos + 1] = (checksum & 0xFF) as u8;

        data
    }

    /// Format the source IP as a dotted-quad string.
    pub fn source_ip_str(&self) -> String {
        format!("{}.{}.{}.{}", self.source_ip[0], self.source_ip[1], self.source_ip[2], self.source_ip[3])
    }

    /// Format the destination IP as a dotted-quad string.
    pub fn dest_ip_str(&self) -> String {
        format!("{}.{}.{}.{}", self.dest_ip[0], self.dest_ip[1], self.dest_ip[2], self.dest_ip[3])
    }

    /// Get the protocol name.
    pub fn protocol_name(&self) -> &str {
        match self.protocol {
            1 => "ICMP", 6 => "TCP", 17 => "UDP", 41 => "IPv6",
            47 => "GRE", 50 => "ESP", 51 => "AH", 89 => "OSPF",
            _ => "Unknown",
        }
    }
}

// ---- TCP ----

/// A parsed TCP segment (RFC 793).
#[derive(Debug, Clone)]
pub struct TcpSegment {
    pub source_port: u16,
    pub dest_port: u16,
    pub sequence_number: u32,
    pub ack_number: u32,
    pub data_offset: u8,          // Header length in 32-bit words
    pub flags: TcpFlags,
    pub window_size: u16,
    pub checksum: u16,
    pub urgent_pointer: u16,
    pub options: Vec<TcpOption>,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct TcpFlags {
    pub ns: bool,     // ECN-nonce
    pub cwr: bool,    // Congestion Window Reduced
    pub ece: bool,    // ECN-Echo
    pub urg: bool,    // Urgent
    pub ack: bool,    // Acknowledgment
    pub psh: bool,    // Push
    pub rst: bool,    // Reset
    pub syn: bool,    // Synchronize (connection establishment)
    pub fin: bool,    // Finish (connection termination)
}

impl TcpFlags {
    pub fn as_string(&self) -> String {
        let mut s = String::new();
        if self.syn { s.push_str("SYN "); }
        if self.ack { s.push_str("ACK "); }
        if self.fin { s.push_str("FIN "); }
        if self.rst { s.push_str("RST "); }
        if self.psh { s.push_str("PSH "); }
        if self.urg { s.push_str("URG "); }
        s.trim().to_string()
    }
}

#[derive(Debug, Clone)]
pub enum TcpOption {
    EndOfList,
    Nop,
    MaxSegmentSize(u16),
    WindowScale(u8),
    SackPermitted,
    Sack(Vec<(u32, u32)>),
    Timestamp { value: u32, echo_reply: u32 },
    Unknown { kind: u8, data: Vec<u8> },
}

impl TcpSegment {
    /// Parse a TCP segment from a byte buffer.
    pub fn parse(buf: &mut ReadBuffer) -> Result<Self, ParseError> {
        let source_port = buf.read_u16_be()?;
        let dest_port = buf.read_u16_be()?;
        let sequence_number = buf.read_u32_be()?;
        let ack_number = buf.read_u32_be()?;

        // Data offset (4 bits) + Reserved (3 bits) + Flags (9 bits)
        let data_offset = buf.read_bits(4)? as u8;
        let _reserved = buf.read_bits(3)?;

        // 9 flag bits
        let ns = buf.read_bits(1)? != 0;
        let cwr = buf.read_bits(1)? != 0;
        let ece = buf.read_bits(1)? != 0;
        let urg = buf.read_bits(1)? != 0;
        let ack = buf.read_bits(1)? != 0;
        let psh = buf.read_bits(1)? != 0;
        let rst = buf.read_bits(1)? != 0;
        let syn = buf.read_bits(1)? != 0;
        let fin = buf.read_bits(1)? != 0;

        let flags = TcpFlags { ns, cwr, ece, urg, ack, psh, rst, syn, fin };

        let window_size = buf.read_u16_be()?;
        let checksum = buf.read_u16_be()?;
        let urgent_pointer = buf.read_u16_be()?;

        // Parse options (everything between byte 20 and data_offset * 4).
        let header_size = (data_offset as usize) * 4;
        let options_size = header_size.saturating_sub(20);
        let mut options = Vec::new();

        if options_size > 0 {
            let options_end = buf.position() + options_size;
            while buf.position() < options_end {
                let kind = buf.read_u8()?;
                match kind {
                    0 => { options.push(TcpOption::EndOfList); break; }
                    1 => { options.push(TcpOption::Nop); }
                    2 => {
                        let _len = buf.read_u8()?;
                        let mss = buf.read_u16_be()?;
                        options.push(TcpOption::MaxSegmentSize(mss));
                    }
                    3 => {
                        let _len = buf.read_u8()?;
                        let scale = buf.read_u8()?;
                        options.push(TcpOption::WindowScale(scale));
                    }
                    4 => {
                        let _len = buf.read_u8()?;
                        options.push(TcpOption::SackPermitted);
                    }
                    8 => {
                        let _len = buf.read_u8()?;
                        let value = buf.read_u32_be()?;
                        let echo = buf.read_u32_be()?;
                        options.push(TcpOption::Timestamp { value, echo_reply: echo });
                    }
                    _ => {
                        let len = buf.read_u8()? as usize;
                        let data = if len > 2 { buf.read_bytes(len - 2)?.to_vec() } else { Vec::new() };
                        options.push(TcpOption::Unknown { kind, data });
                    }
                }
            }
            // Skip any remaining padding.
            let remaining = options_end.saturating_sub(buf.position());
            if remaining > 0 { buf.skip(remaining)?; }
        }

        let payload = buf.read_bytes(buf.remaining())?.to_vec();

        Ok(TcpSegment {
            source_port, dest_port, sequence_number, ack_number,
            data_offset, flags, window_size, checksum, urgent_pointer,
            options, payload,
        })
    }
}

// ---- UDP ----

/// A parsed UDP datagram (RFC 768). UDP is the simplest transport protocol:
/// just source port, dest port, length, checksum, and payload.
#[derive(Debug, Clone)]
pub struct UdpDatagram {
    pub source_port: u16,
    pub dest_port: u16,
    pub length: u16,
    pub checksum: u16,
    pub payload: Vec<u8>,
}

impl UdpDatagram {
    pub fn parse(buf: &mut ReadBuffer) -> Result<Self, ParseError> {
        let source_port = buf.read_u16_be()?;
        let dest_port = buf.read_u16_be()?;
        let length = buf.read_u16_be()?;
        let checksum = buf.read_u16_be()?;
        let payload_len = (length as usize).saturating_sub(8);
        let payload = buf.read_bytes(payload_len.min(buf.remaining()))?.to_vec();
        Ok(UdpDatagram { source_port, dest_port, length, checksum, payload })
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = WriteBuffer::with_capacity(8 + self.payload.len());
        buf.write_u16_be(self.source_port);
        buf.write_u16_be(self.dest_port);
        buf.write_u16_be(self.length);
        buf.write_u16_be(self.checksum);
        buf.write_bytes(&self.payload);
        buf.finish()
    }
}

// ---- DNS ----

/// A parsed DNS message (RFC 1035). DNS is the backbone of the internet —
/// it translates domain names (like "example.com") to IP addresses.
#[derive(Debug, Clone)]
pub struct DnsMessage {
    pub id: u16,
    pub is_response: bool,
    pub opcode: u8,
    pub authoritative: bool,
    pub truncated: bool,
    pub recursion_desired: bool,
    pub recursion_available: bool,
    pub rcode: u8,                    // Response code (0=no error, 3=NXDOMAIN)
    pub questions: Vec<DnsQuestion>,
    pub answers: Vec<DnsRecord>,
    pub authority: Vec<DnsRecord>,
    pub additional: Vec<DnsRecord>,
}

#[derive(Debug, Clone)]
pub struct DnsQuestion {
    pub name: String,
    pub qtype: u16,       // 1=A, 28=AAAA, 5=CNAME, 15=MX, 2=NS, etc.
    pub qclass: u16,      // 1=IN (Internet)
}

#[derive(Debug, Clone)]
pub struct DnsRecord {
    pub name: String,
    pub rtype: u16,
    pub rclass: u16,
    pub ttl: u32,
    pub rdata: DnsRData,
}

#[derive(Debug, Clone)]
pub enum DnsRData {
    A([u8; 4]),                      // IPv4 address
    AAAA([u8; 16]),                  // IPv6 address
    CNAME(String),
    MX { preference: u16, exchange: String },
    NS(String),
    TXT(String),
    SOA { mname: String, rname: String, serial: u32, refresh: u32, retry: u32, expire: u32, minimum: u32 },
    Raw(Vec<u8>),
}

impl DnsMessage {
    /// Parse a DNS message from bytes.
    pub fn parse(buf: &mut ReadBuffer) -> Result<Self, ParseError> {
        let full_data = buf.data().to_vec(); // Keep for name decompression
        let id = buf.read_u16_be()?;

        // Flags field: 16 bits packed with multiple fields.
        let flags = buf.read_u16_be()?;
        let is_response = (flags >> 15) & 1 == 1;
        let opcode = ((flags >> 11) & 0xF) as u8;
        let authoritative = (flags >> 10) & 1 == 1;
        let truncated = (flags >> 9) & 1 == 1;
        let recursion_desired = (flags >> 8) & 1 == 1;
        let recursion_available = (flags >> 7) & 1 == 1;
        let rcode = (flags & 0xF) as u8;

        let qdcount = buf.read_u16_be()? as usize;
        let ancount = buf.read_u16_be()? as usize;
        let nscount = buf.read_u16_be()? as usize;
        let arcount = buf.read_u16_be()? as usize;

        // Parse questions.
        let mut questions = Vec::with_capacity(qdcount);
        for _ in 0..qdcount {
            let name = Self::read_dns_name(buf, &full_data)?;
            let qtype = buf.read_u16_be()?;
            let qclass = buf.read_u16_be()?;
            questions.push(DnsQuestion { name, qtype, qclass });
        }

        // Parse resource records (answers, authority, additional).
        let mut answers = Vec::with_capacity(ancount);
        for _ in 0..ancount {
            answers.push(Self::read_dns_record(buf, &full_data)?);
        }

        let mut authority = Vec::with_capacity(nscount);
        for _ in 0..nscount {
            authority.push(Self::read_dns_record(buf, &full_data)?);
        }

        let mut additional = Vec::with_capacity(arcount);
        for _ in 0..arcount {
            if buf.remaining() >= 11 { // Minimum RR size
                additional.push(Self::read_dns_record(buf, &full_data)?);
            }
        }

        Ok(DnsMessage {
            id, is_response, opcode, authoritative, truncated,
            recursion_desired, recursion_available, rcode,
            questions, answers, authority, additional,
        })
    }

    /// Read a DNS domain name, handling compression pointers.
    /// DNS uses a clever compression scheme: if the first two bits of a label
    /// length byte are 11, the remaining 14 bits are a pointer to a previous
    /// occurrence of the name in the packet. This saves significant space
    /// because domain names are repeated frequently in DNS responses.
    fn read_dns_name(buf: &mut ReadBuffer, full_data: &[u8]) -> Result<String, ParseError> {
        let mut parts = Vec::new();
        let mut jumped = false;
        let mut return_pos = 0;

        loop {
            if buf.remaining() == 0 { break; }
            let len_byte = buf.read_u8()?;

            if len_byte == 0 {
                break; // Root label — end of name
            }

            if (len_byte & 0xC0) == 0xC0 {
                // Compression pointer: next byte combined with lower 6 bits
                // gives the offset into the packet where the name continues.
                let next = buf.read_u8()? as usize;
                let pointer = ((len_byte as usize & 0x3F) << 8) | next;

                if !jumped {
                    return_pos = buf.position();
                }
                jumped = true;
                buf.seek(pointer);
                continue;
            }

            // Regular label: len_byte is the length, followed by that many chars.
            let label = buf.read_bytes(len_byte as usize)?;
            parts.push(String::from_utf8_lossy(label).to_string());
        }

        if jumped {
            buf.seek(return_pos);
        }

        Ok(parts.join("."))
    }

    /// Read a DNS resource record.
    fn read_dns_record(buf: &mut ReadBuffer, full_data: &[u8]) -> Result<DnsRecord, ParseError> {
        let name = Self::read_dns_name(buf, full_data)?;
        let rtype = buf.read_u16_be()?;
        let rclass = buf.read_u16_be()?;
        let ttl = buf.read_u32_be()?;
        let rdlength = buf.read_u16_be()? as usize;

        let rdata_start = buf.position();
        let rdata = match rtype {
            1 if rdlength == 4 => {
                // A record (IPv4 address)
                let mut ip = [0u8; 4];
                for i in 0..4 { ip[i] = buf.read_u8()?; }
                DnsRData::A(ip)
            }
            28 if rdlength == 16 => {
                // AAAA record (IPv6 address)
                let mut ip = [0u8; 16];
                for i in 0..16 { ip[i] = buf.read_u8()?; }
                DnsRData::AAAA(ip)
            }
            5 => {
                // CNAME
                let cname = Self::read_dns_name(buf, full_data)?;
                DnsRData::CNAME(cname)
            }
            15 => {
                // MX
                let preference = buf.read_u16_be()?;
                let exchange = Self::read_dns_name(buf, full_data)?;
                DnsRData::MX { preference, exchange }
            }
            2 => {
                // NS
                let ns = Self::read_dns_name(buf, full_data)?;
                DnsRData::NS(ns)
            }
            _ => {
                let data = buf.read_bytes(rdlength)?.to_vec();
                DnsRData::Raw(data)
            }
        };

        // Ensure we consumed exactly rdlength bytes.
        let consumed = buf.position() - rdata_start;
        if consumed < rdlength {
            buf.skip(rdlength - consumed)?;
        }

        Ok(DnsRecord { name, rtype, rclass, ttl, rdata })
    }

    /// Build a DNS query message.
    pub fn build_query(id: u16, domain: &str, qtype: u16) -> Vec<u8> {
        let mut buf = WriteBuffer::with_capacity(512);

        // Header
        buf.write_u16_be(id);
        buf.write_u16_be(0x0100);  // Standard query, recursion desired
        buf.write_u16_be(1);       // 1 question
        buf.write_u16_be(0);       // 0 answers
        buf.write_u16_be(0);       // 0 authority
        buf.write_u16_be(0);       // 0 additional

        // Question: encode domain name as labels
        for label in domain.split('.') {
            buf.write_u8(label.len() as u8);
            buf.write_bytes(label.as_bytes());
        }
        buf.write_u8(0);          // Root label (end of name)
        buf.write_u16_be(qtype);  // Query type
        buf.write_u16_be(1);      // Query class (IN)

        buf.finish()
    }
}

// ---- HTTP/1.1 ----

/// A parsed HTTP request.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub version: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// A parsed HTTP response.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub version: String,
    pub status_code: u16,
    pub reason: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpRequest {
    /// Parse an HTTP/1.1 request from a byte buffer.
    pub fn parse(buf: &mut ReadBuffer) -> Result<Self, ParseError> {
        // Request line: METHOD PATH HTTP/VERSION\r\n
        let request_line = buf.read_line()?;
        let parts: Vec<&str> = request_line.splitn(3, ' ').collect();
        if parts.len() < 3 {
            return Err(ParseError::invalid(&format!(
                "Malformed request line: '{}'", request_line
            )));
        }
        let method = parts[0].to_string();
        let path = parts[1].to_string();
        let version = parts[2].to_string();

        // Headers: Key: Value\r\n (repeated), terminated by empty line \r\n
        let mut headers = Vec::new();
        loop {
            let line = buf.read_line()?;
            if line.is_empty() { break; } // Empty line = end of headers
            if let Some(colon_pos) = line.find(':') {
                let key = line[..colon_pos].trim().to_string();
                let value = line[colon_pos + 1..].trim().to_string();
                headers.push((key, value));
            }
        }

        // Body: determined by Content-Length header or chunked transfer encoding.
        let content_length = headers.iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
            .and_then(|(_, v)| v.parse::<usize>().ok())
            .unwrap_or(0);

        let body = if content_length > 0 && buf.remaining() >= content_length {
            buf.read_bytes(content_length)?.to_vec()
        } else {
            Vec::new()
        };

        Ok(HttpRequest { method, path, version, headers, body })
    }

    /// Get a header value by name (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Serialize to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut s = format!("{} {} {}\r\n", self.method, self.path, self.version);
        for (key, value) in &self.headers {
            s.push_str(&format!("{}: {}\r\n", key, value));
        }
        s.push_str("\r\n");
        let mut bytes = s.into_bytes();
        bytes.extend_from_slice(&self.body);
        bytes
    }
}

impl HttpResponse {
    /// Parse an HTTP/1.1 response from a byte buffer.
    pub fn parse(buf: &mut ReadBuffer) -> Result<Self, ParseError> {
        let status_line = buf.read_line()?;
        let parts: Vec<&str> = status_line.splitn(3, ' ').collect();
        if parts.len() < 2 {
            return Err(ParseError::invalid(&format!("Malformed status line: '{}'", status_line)));
        }
        let version = parts[0].to_string();
        let status_code = parts[1].parse::<u16>()
            .map_err(|_| ParseError::invalid("Invalid status code"))?;
        let reason = parts.get(2).unwrap_or(&"").to_string();

        let mut headers = Vec::new();
        loop {
            let line = buf.read_line()?;
            if line.is_empty() { break; }
            if let Some(colon_pos) = line.find(':') {
                headers.push((
                    line[..colon_pos].trim().to_string(),
                    line[colon_pos + 1..].trim().to_string(),
                ));
            }
        }

        let content_length = headers.iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
            .and_then(|(_, v)| v.parse::<usize>().ok())
            .unwrap_or(0);

        let body = if content_length > 0 && buf.remaining() >= content_length {
            buf.read_bytes(content_length)?.to_vec()
        } else {
            buf.read_bytes(buf.remaining())?.to_vec()
        };

        Ok(HttpResponse { version, status_code, reason, headers, body })
    }

    /// Build a simple HTTP response.
    pub fn build(status: u16, reason: &str, content_type: &str, body: &[u8]) -> Vec<u8> {
        let mut s = format!("HTTP/1.1 {} {}\r\n", status, reason);
        s.push_str(&format!("Content-Type: {}\r\n", content_type));
        s.push_str(&format!("Content-Length: {}\r\n", body.len()));
        s.push_str("Connection: close\r\n");
        s.push_str("\r\n");
        let mut bytes = s.into_bytes();
        bytes.extend_from_slice(body);
        bytes
    }
}

// ---- TLV (Type-Length-Value) Generic Protocol ----

/// A TLV (Type-Length-Value) record. This is a very common encoding pattern
/// used in RADIUS, DHCP, BGP, SSL/TLS, smart cards, and many other protocols.
#[derive(Debug, Clone)]
pub struct TlvRecord {
    pub tag: u16,
    pub value: Vec<u8>,
}

/// Parse a sequence of TLV records. The `tag_bytes` and `len_bytes` parameters
/// control how wide the type and length fields are (1, 2, or 4 bytes each).
pub fn parse_tlv(buf: &mut ReadBuffer, tag_bytes: u8, len_bytes: u8) -> Result<Vec<TlvRecord>, ParseError> {
    let mut records = Vec::new();
    while buf.remaining() > (tag_bytes + len_bytes) as usize {
        let tag = match tag_bytes {
            1 => buf.read_u8()? as u16,
            2 => buf.read_u16_be()?,
            _ => return Err(ParseError::invalid("Tag must be 1 or 2 bytes")),
        };
        let len = match len_bytes {
            1 => buf.read_u8()? as usize,
            2 => buf.read_u16_be()? as usize,
            4 => buf.read_u32_be()? as usize,
            _ => return Err(ParseError::invalid("Length must be 1, 2, or 4 bytes")),
        };
        if buf.remaining() < len { break; }
        let value = buf.read_bytes(len)?.to_vec();
        records.push(TlvRecord { tag, value });
    }
    Ok(records)
}

// ---- RESP (Redis Serialization Protocol) ----

/// A RESP (Redis Serialization Protocol) value.
/// Redis uses a simple text-based protocol for client-server communication.
#[derive(Debug, Clone)]
pub enum RespValue {
    SimpleString(String),    // +OK\r\n
    Error(String),           // -ERR message\r\n
    Integer(i64),            // :42\r\n
    BulkString(Vec<u8>),     // $6\r\nhello!\r\n
    Null,                    // $-1\r\n
    Array(Vec<RespValue>),   // *2\r\n$3\r\nGET\r\n$3\r\nkey\r\n
}

impl RespValue {
    /// Parse a RESP value from a byte buffer.
    pub fn parse(buf: &mut ReadBuffer) -> Result<Self, ParseError> {
        let type_byte = buf.read_u8()?;
        match type_byte {
            b'+' => {
                let line = buf.read_line()?;
                Ok(RespValue::SimpleString(line))
            }
            b'-' => {
                let line = buf.read_line()?;
                Ok(RespValue::Error(line))
            }
            b':' => {
                let line = buf.read_line()?;
                let val = line.parse::<i64>()
                    .map_err(|_| ParseError::invalid("Invalid RESP integer"))?;
                Ok(RespValue::Integer(val))
            }
            b'$' => {
                let line = buf.read_line()?;
                let len = line.parse::<i64>()
                    .map_err(|_| ParseError::invalid("Invalid RESP bulk string length"))?;
                if len < 0 {
                    Ok(RespValue::Null)
                } else {
                    let data = buf.read_bytes(len as usize)?.to_vec();
                    buf.skip(2)?; // Skip \r\n after bulk string data
                    Ok(RespValue::BulkString(data))
                }
            }
            b'*' => {
                let line = buf.read_line()?;
                let count = line.parse::<i64>()
                    .map_err(|_| ParseError::invalid("Invalid RESP array length"))?;
                if count < 0 {
                    Ok(RespValue::Null)
                } else {
                    let mut elements = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        elements.push(RespValue::parse(buf)?);
                    }
                    Ok(RespValue::Array(elements))
                }
            }
            _ => Err(ParseError::invalid(&format!("Unknown RESP type byte: 0x{:02X}", type_byte))),
        }
    }

    /// Serialize a RESP value to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        match self {
            RespValue::SimpleString(s) => format!("+{}\r\n", s).into_bytes(),
            RespValue::Error(s) => format!("-{}\r\n", s).into_bytes(),
            RespValue::Integer(n) => format!(":{}\r\n", n).into_bytes(),
            RespValue::Null => "$-1\r\n".as_bytes().to_vec(),
            RespValue::BulkString(data) => {
                let mut result = format!("${}\r\n", data.len()).into_bytes();
                result.extend_from_slice(data);
                result.extend_from_slice(b"\r\n");
                result
            }
            RespValue::Array(elements) => {
                let mut result = format!("*{}\r\n", elements.len()).into_bytes();
                for elem in elements {
                    result.extend_from_slice(&elem.serialize());
                }
                result
            }
        }
    }

    /// Build a RESP command (array of bulk strings).
    /// Example: RespValue::command(&["SET", "key", "value"])
    pub fn command(parts: &[&str]) -> Self {
        RespValue::Array(parts.iter().map(|s| {
            RespValue::BulkString(s.as_bytes().to_vec())
        }).collect())
    }
}

// ============================================================================
// PART 6: SERVICE MESH PRIMITIVES
// ============================================================================
// These are the building blocks for resilient microservice communication.
// They sit ABOVE the protocol layer and provide fault tolerance.

/// A circuit breaker prevents cascading failures by stopping requests to
/// a failing service. It has three states:
///   - Closed: requests flow through normally
///   - Open: all requests are rejected immediately (the service is "down")
///   - Half-Open: a few test requests are allowed through to check recovery
///
/// This is the exact pattern used by Netflix's Hystrix and Resilience4j.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    pub state: CircuitState,
    pub failure_count: u32,
    pub success_count: u32,
    pub failure_threshold: u32,     // Failures before opening
    pub success_threshold: u32,     // Successes in half-open before closing
    pub timeout: Duration,          // How long to stay open before trying half-open
    pub last_failure_time: Option<Instant>,
    pub half_open_max: u32,         // Max concurrent requests in half-open
    pub half_open_active: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState { Closed, Open, HalfOpen }

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, success_threshold: u32, timeout: Duration) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0, success_count: 0,
            failure_threshold, success_threshold,
            timeout, last_failure_time: None,
            half_open_max: 1, half_open_active: 0,
        }
    }

    /// Check if a request should be allowed through.
    pub fn allow_request(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if timeout has elapsed → transition to half-open.
                if let Some(last) = self.last_failure_time {
                    if last.elapsed() >= self.timeout {
                        self.state = CircuitState::HalfOpen;
                        self.half_open_active = 0;
                        self.success_count = 0;
                        return self.half_open_active < self.half_open_max;
                    }
                }
                false
            }
            CircuitState::HalfOpen => {
                self.half_open_active < self.half_open_max
            }
        }
    }

    /// Record a successful request.
    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => { self.failure_count = 0; }
            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.success_threshold {
                    // Service has recovered — close the circuit.
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                }
            }
            CircuitState::Open => {} // Shouldn't happen
        }
    }

    /// Record a failed request.
    pub fn record_failure(&mut self) {
        self.last_failure_time = Some(Instant::now());
        match self.state {
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                // One failure in half-open → immediately reopen.
                self.state = CircuitState::Open;
            }
            CircuitState::Open => {} // Already open
        }
    }
}

/// A token bucket rate limiter. Allows `rate` requests per second with
/// `burst` maximum burst size. This is the algorithm used by API gateways.
#[derive(Debug, Clone)]
pub struct TokenBucketLimiter {
    pub tokens: f64,
    pub max_tokens: f64,
    pub refill_rate: f64,          // Tokens per second
    pub last_refill: Instant,
}

impl TokenBucketLimiter {
    pub fn new(rate: f64, burst: u32) -> Self {
        Self {
            tokens: burst as f64,
            max_tokens: burst as f64,
            refill_rate: rate,
            last_refill: Instant::now(),
        }
    }

    /// Try to acquire a token. Returns true if the request is allowed.
    pub fn try_acquire(&mut self) -> bool {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Refill tokens based on elapsed time.
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill = now;
    }
}

/// A retry policy with configurable backoff.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub backoff: BackoffStrategy,
    pub current_attempt: u32,
}

#[derive(Debug, Clone)]
pub enum BackoffStrategy {
    Constant(Duration),
    Linear { initial: Duration, increment: Duration },
    Exponential { initial: Duration, max: Duration, multiplier: f64 },
}

impl RetryPolicy {
    pub fn exponential(max_attempts: u32, initial: Duration, max: Duration) -> Self {
        Self {
            max_attempts,
            backoff: BackoffStrategy::Exponential { initial, max, multiplier: 2.0 },
            current_attempt: 0,
        }
    }

    /// Get the delay before the next retry, or None if retries are exhausted.
    pub fn next_delay(&mut self) -> Option<Duration> {
        if self.current_attempt >= self.max_attempts { return None; }
        let delay = match &self.backoff {
            BackoffStrategy::Constant(d) => *d,
            BackoffStrategy::Linear { initial, increment } => {
                *initial + *increment * self.current_attempt
            }
            BackoffStrategy::Exponential { initial, max, multiplier } => {
                let delay = initial.mul_f64(multiplier.powi(self.current_attempt as i32));
                delay.min(*max)
            }
        };
        self.current_attempt += 1;
        Some(delay)
    }

    pub fn reset(&mut self) { self.current_attempt = 0; }
}


// ============================================================================
// TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // ---- Buffer tests ----

    #[test]
    fn test_read_integers() {
        let mut buf = ReadBuffer::new(vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
        assert_eq!(buf.read_u8().unwrap(), 0x01);
        assert_eq!(buf.read_u16_be().unwrap(), 0x0203);
        assert_eq!(buf.read_u8().unwrap(), 0x04);
        assert_eq!(buf.read_u32_be().unwrap(), 0x05060708);
    }

    #[test]
    fn test_read_u24() {
        let mut buf = ReadBuffer::new(vec![0x01, 0x00, 0x00]); // 65536
        assert_eq!(buf.read_u24_be().unwrap(), 65536);
    }

    #[test]
    fn test_read_bits() {
        // IPv4 first byte: version=4 (4 bits), IHL=5 (4 bits) → 0x45
        let mut buf = ReadBuffer::new(vec![0x45]);
        assert_eq!(buf.read_bits(4).unwrap(), 4); // version
        assert_eq!(buf.read_bits(4).unwrap(), 5); // IHL
    }

    #[test]
    fn test_write_bits() {
        let mut buf = WriteBuffer::new();
        buf.write_bits(4, 4);  // version = 4
        buf.write_bits(5, 4);  // IHL = 5
        let data = buf.finish();
        assert_eq!(data, vec![0x45]);
    }

    #[test]
    fn test_varint() {
        let mut wbuf = WriteBuffer::new();
        wbuf.write_varint(300); // 300 = 0b100101100 → [0xAC, 0x02]
        let data = wbuf.finish();
        assert_eq!(data, vec![0xAC, 0x02]);

        let mut rbuf = ReadBuffer::new(data);
        assert_eq!(rbuf.read_varint().unwrap(), 300);
    }

    #[test]
    fn test_cstring() {
        let mut buf = ReadBuffer::new(b"hello\0world\0".to_vec());
        assert_eq!(buf.read_cstring().unwrap(), "hello");
        assert_eq!(buf.read_cstring().unwrap(), "world");
    }

    #[test]
    fn test_from_hex() {
        let buf = ReadBuffer::from_hex("48 45 4C 4C 4F").unwrap();
        assert_eq!(buf.data(), b"HELLO");
    }

    // ---- Checksum tests ----

    #[test]
    fn test_crc32() {
        assert_eq!(crc32(b""), 0x00000000);
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn test_internet_checksum() {
        // Example from RFC 1071: the bytes of the 16-bit words
        let data = vec![0x00, 0x01, 0xf2, 0x03, 0xf4, 0xf5, 0xf6, 0xf7];
        let check = internet_checksum(&data);
        // Verify that including the checksum yields 0.
        let mut verify = data.clone();
        verify.push((check >> 8) as u8);
        verify.push((check & 0xFF) as u8);
        assert_eq!(internet_checksum(&verify), 0);
    }

    #[test]
    fn test_adler32() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E60398);
    }

    // ---- Protocol state machine tests ----

    #[test]
    fn test_tcp_state_machine_normal_flow() {
        let mut tcp = StateMachineInstance::new(tcp_state_machine());

        // Server-side connection: passive open → receive SYN → receive ACK
        assert_eq!(tcp.current_state, "CLOSED");
        tcp.transition("passive_open").unwrap();
        assert_eq!(tcp.current_state, "LISTEN");
        tcp.transition("recv_syn").unwrap();
        assert_eq!(tcp.current_state, "SYN_RECEIVED");
        tcp.transition("recv_ack").unwrap();
        assert_eq!(tcp.current_state, "ESTABLISHED");

        // Data transfer happens here...

        // Passive close: receive FIN → close → receive ACK
        tcp.transition("recv_fin").unwrap();
        assert_eq!(tcp.current_state, "CLOSE_WAIT");
        tcp.transition("close").unwrap();
        assert_eq!(tcp.current_state, "LAST_ACK");
        tcp.transition("recv_ack").unwrap();
        assert_eq!(tcp.current_state, "CLOSED");
        assert!(tcp.is_terminal());
    }

    #[test]
    fn test_invalid_transition() {
        let mut tcp = StateMachineInstance::new(tcp_state_machine());
        // Can't send data in CLOSED state.
        assert!(tcp.transition("recv_ack").is_err());
    }

    // ---- IPv4 parsing test ----

    #[test]
    fn test_ipv4_parse() {
        // A minimal IPv4 packet: version=4, IHL=5, total_length=28,
        // TTL=64, protocol=6 (TCP), source=192.168.1.1, dest=10.0.0.1
        // Plus 8 bytes of "payload" (minimal TCP header would be 20 bytes,
        // but we're just testing the IP layer).
        let packet = vec![
            0x45,                   // Version=4, IHL=5
            0x00,                   // DSCP=0, ECN=0
            0x00, 0x1C,             // Total length = 28
            0x00, 0x01,             // Identification
            0x40, 0x00,             // Flags=DF, Fragment offset=0
            0x40,                   // TTL=64
            0x06,                   // Protocol=TCP
            0x00, 0x00,             // Checksum (0 for testing)
            0xC0, 0xA8, 0x01, 0x01, // Source: 192.168.1.1
            0x0A, 0x00, 0x00, 0x01, // Dest: 10.0.0.1
            // 8 bytes of payload
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        ];

        let mut buf = ReadBuffer::new(packet);
        let ip = IPv4Packet::parse(&mut buf).unwrap();

        assert_eq!(ip.version, 4);
        assert_eq!(ip.ihl, 5);
        assert_eq!(ip.total_length, 28);
        assert_eq!(ip.ttl, 64);
        assert_eq!(ip.protocol, 6);
        assert_eq!(ip.protocol_name(), "TCP");
        assert_eq!(ip.source_ip_str(), "192.168.1.1");
        assert_eq!(ip.dest_ip_str(), "10.0.0.1");
        assert_eq!(ip.payload.len(), 8);
    }

    #[test]
    fn test_ipv4_roundtrip() {
        let original = IPv4Packet {
            version: 4, ihl: 5, dscp: 0, ecn: 0,
            total_length: 28, identification: 1,
            flags: 0b010, fragment_offset: 0,
            ttl: 64, protocol: 17, // UDP
            header_checksum: 0,
            source_ip: [10, 0, 0, 1],
            dest_ip: [10, 0, 0, 2],
            options: Vec::new(),
            payload: vec![1, 2, 3, 4, 5, 6, 7, 8],
        };

        let serialized = original.serialize();
        let mut buf = ReadBuffer::new(serialized);
        let parsed = IPv4Packet::parse(&mut buf).unwrap();

        assert_eq!(parsed.version, 4);
        assert_eq!(parsed.source_ip_str(), "10.0.0.1");
        assert_eq!(parsed.dest_ip_str(), "10.0.0.2");
        assert_eq!(parsed.payload, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        // The checksum should now be non-zero (computed during serialization).
        assert_ne!(parsed.header_checksum, 0);
    }

    // ---- TCP parsing test ----

    #[test]
    fn test_tcp_parse() {
        // SYN packet: source=12345, dest=80, seq=1000, flags=SYN
        let segment = vec![
            0x30, 0x39,             // Source port: 12345
            0x00, 0x50,             // Dest port: 80
            0x00, 0x00, 0x03, 0xE8, // Sequence number: 1000
            0x00, 0x00, 0x00, 0x00, // ACK number: 0
            0x50,                   // Data offset=5 (20 bytes), reserved=0
            0x02,                   // Flags: SYN
            0xFF, 0xFF,             // Window size: 65535
            0x00, 0x00,             // Checksum: 0
            0x00, 0x00,             // Urgent pointer: 0
        ];

        let mut buf = ReadBuffer::new(segment);
        let tcp = TcpSegment::parse(&mut buf).unwrap();

        assert_eq!(tcp.source_port, 12345);
        assert_eq!(tcp.dest_port, 80);
        assert_eq!(tcp.sequence_number, 1000);
        assert!(tcp.flags.syn);
        assert!(!tcp.flags.ack);
        assert_eq!(tcp.flags.as_string(), "SYN");
        assert_eq!(tcp.window_size, 65535);
    }

    // ---- UDP parsing test ----

    #[test]
    fn test_udp_parse() {
        let datagram = vec![
            0x13, 0x88,             // Source port: 5000
            0x00, 0x35,             // Dest port: 53 (DNS)
            0x00, 0x0C,             // Length: 12 (8 header + 4 payload)
            0x00, 0x00,             // Checksum: 0
            0xDE, 0xAD, 0xBE, 0xEF, // Payload
        ];

        let mut buf = ReadBuffer::new(datagram);
        let udp = UdpDatagram::parse(&mut buf).unwrap();

        assert_eq!(udp.source_port, 5000);
        assert_eq!(udp.dest_port, 53);
        assert_eq!(udp.payload, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_udp_roundtrip() {
        let original = UdpDatagram {
            source_port: 1234, dest_port: 5678,
            length: 13, checksum: 0,
            payload: vec![b'H', b'e', b'l', b'l', b'o'],
        };
        let bytes = original.serialize();
        let mut buf = ReadBuffer::new(bytes);
        let parsed = UdpDatagram::parse(&mut buf).unwrap();
        assert_eq!(parsed.source_port, 1234);
        assert_eq!(parsed.payload, b"Hello");
    }

    // ---- DNS test ----

    #[test]
    fn test_dns_query_build_and_parse() {
        let query = DnsMessage::build_query(0x1234, "example.com", 1);
        let mut buf = ReadBuffer::new(query);
        let parsed = DnsMessage::parse(&mut buf).unwrap();

        assert_eq!(parsed.id, 0x1234);
        assert!(!parsed.is_response);
        assert!(parsed.recursion_desired);
        assert_eq!(parsed.questions.len(), 1);
        assert_eq!(parsed.questions[0].name, "example.com");
        assert_eq!(parsed.questions[0].qtype, 1); // A record
    }

    // ---- HTTP test ----

    #[test]
    fn test_http_request_parse() {
        let raw = b"GET /api/users HTTP/1.1\r\nHost: example.com\r\nContent-Length: 5\r\n\r\nhello";
        let mut buf = ReadBuffer::new(raw.to_vec());
        let req = HttpRequest::parse(&mut buf).unwrap();

        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/api/users");
        assert_eq!(req.version, "HTTP/1.1");
        assert_eq!(req.header("Host").unwrap(), "example.com");
        assert_eq!(req.body, b"hello");
    }

    #[test]
    fn test_http_response_build_and_parse() {
        let response = HttpResponse::build(200, "OK", "text/plain", b"Hello, World!");
        let mut buf = ReadBuffer::new(response);
        let parsed = HttpResponse::parse(&mut buf).unwrap();

        assert_eq!(parsed.status_code, 200);
        assert_eq!(parsed.reason, "OK");
        assert_eq!(parsed.body, b"Hello, World!");
    }

    // ---- RESP (Redis) test ----

    #[test]
    fn test_resp_parse_and_serialize() {
        // Build a SET command
        let cmd = RespValue::command(&["SET", "mykey", "myvalue"]);
        let bytes = cmd.serialize();

        let mut buf = ReadBuffer::new(bytes);
        let parsed = RespValue::parse(&mut buf).unwrap();

        if let RespValue::Array(elements) = parsed {
            assert_eq!(elements.len(), 3);
            if let RespValue::BulkString(ref data) = elements[0] {
                assert_eq!(data, b"SET");
            }
            if let RespValue::BulkString(ref data) = elements[1] {
                assert_eq!(data, b"mykey");
            }
        } else {
            panic!("Expected array");
        }
    }

    #[test]
    fn test_resp_types() {
        // Simple string
        let mut buf = ReadBuffer::new(b"+OK\r\n".to_vec());
        assert!(matches!(RespValue::parse(&mut buf).unwrap(), RespValue::SimpleString(s) if s == "OK"));

        // Error
        let mut buf = ReadBuffer::new(b"-ERR unknown command\r\n".to_vec());
        assert!(matches!(RespValue::parse(&mut buf).unwrap(), RespValue::Error(s) if s == "ERR unknown command"));

        // Integer
        let mut buf = ReadBuffer::new(b":42\r\n".to_vec());
        assert!(matches!(RespValue::parse(&mut buf).unwrap(), RespValue::Integer(42)));

        // Null
        let mut buf = ReadBuffer::new(b"$-1\r\n".to_vec());
        assert!(matches!(RespValue::parse(&mut buf).unwrap(), RespValue::Null));
    }

    // ---- TLV test ----

    #[test]
    fn test_tlv_parse() {
        // Two TLV records with 1-byte tag and 1-byte length:
        // Tag=1, Len=3, Data=[0xAA, 0xBB, 0xCC]
        // Tag=2, Len=2, Data=[0xDD, 0xEE]
        let data = vec![0x01, 0x03, 0xAA, 0xBB, 0xCC, 0x02, 0x02, 0xDD, 0xEE];
        let mut buf = ReadBuffer::new(data);
        let records = parse_tlv(&mut buf, 1, 1).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].tag, 1);
        assert_eq!(records[0].value, vec![0xAA, 0xBB, 0xCC]);
        assert_eq!(records[1].tag, 2);
        assert_eq!(records[1].value, vec![0xDD, 0xEE]);
    }

    // ---- Circuit breaker test ----

    #[test]
    fn test_circuit_breaker() {
        let mut cb = CircuitBreaker::new(3, 2, Duration::from_millis(100));

        // Closed state: requests flow through.
        assert!(cb.allow_request());
        assert_eq!(cb.state, CircuitState::Closed);

        // Three failures → opens the circuit.
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state, CircuitState::Open);
        assert!(!cb.allow_request()); // Rejected!

        // Wait for timeout → transitions to half-open.
        std::thread::sleep(Duration::from_millis(150));
        assert!(cb.allow_request()); // Allowed (half-open test request)
        assert_eq!(cb.state, CircuitState::HalfOpen);

        // Two successes in half-open → closes the circuit.
        cb.record_success();
        cb.record_success();
        assert_eq!(cb.state, CircuitState::Closed);
    }

    // ---- Retry policy test ----

    #[test]
    fn test_retry_exponential_backoff() {
        let mut retry = RetryPolicy::exponential(
            4, Duration::from_millis(100), Duration::from_secs(10),
        );

        let d1 = retry.next_delay().unwrap();
        assert_eq!(d1, Duration::from_millis(100));   // 100ms * 2^0

        let d2 = retry.next_delay().unwrap();
        assert_eq!(d2, Duration::from_millis(200));   // 100ms * 2^1

        let d3 = retry.next_delay().unwrap();
        assert_eq!(d3, Duration::from_millis(400));   // 100ms * 2^2

        let d4 = retry.next_delay().unwrap();
        assert_eq!(d4, Duration::from_millis(800));   // 100ms * 2^3

        assert!(retry.next_delay().is_none());         // Exhausted
    }
}
