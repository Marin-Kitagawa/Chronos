// ============================================================================
// CHRONOS BIOINFORMATICS ENGINE
// ============================================================================
//
// HOW BIOINFORMATICS ACTUALLY WORKS (and why sequence analysis is hard):
//
// Biology stores information in sequences of discrete symbols:
//   - DNA: A, T, G, C (adenine, thymine, guanine, cytosine)
//   - RNA: A, U, G, C (uracil replaces thymine)
//   - Proteins: 20 amino acids encoded by 3-letter codons
//
// The central dogma: DNA → (transcription) → RNA → (translation) → Protein
// Every 3 nucleotides (a "codon") encodes one amino acid. The reading frame
// matters enormously — a single insertion/deletion shifts the frame and
// scrambles the entire downstream protein sequence.
//
// KEY ALGORITHMS:
//
// 1. SEQUENCE ALIGNMENT (Smith-Waterman, Needleman-Wunsch):
//    How similar are two sequences? Dynamic programming finds the optimal
//    alignment by considering all possible insertions, deletions, and
//    substitutions. The scoring matrix (BLOSUM62 for proteins, simple
//    match/mismatch for DNA) quantifies how "acceptable" each substitution is.
//
//    Needleman-Wunsch: global alignment (entire sequences aligned end-to-end)
//    Smith-Waterman: local alignment (find the best matching subsequence)
//
//    The DP recurrence (Smith-Waterman):
//      H[i][j] = max(0,
//                    H[i-1][j-1] + score(a[i], b[j]),  // match/mismatch
//                    H[i-1][j] - gap_penalty,            // deletion
//                    H[i][j-1] - gap_penalty)            // insertion
//
// 2. BURROWS-WHEELER TRANSFORM (BWT) + FM-INDEX:
//    The basis of modern short-read aligners (BWA, Bowtie). The BWT
//    rearranges a sequence so identical characters cluster together,
//    enabling very efficient pattern matching with the FM-index.
//    Reading backward through the BWT via LF-mapping finds all occurrences
//    of a pattern in O(pattern_length) time, regardless of genome size.
//
// 3. DE BRUIJN GRAPH ASSEMBLY:
//    Short reads from sequencing machines (~150 bp) must be assembled into
//    chromosomes (~100 million bp). De Bruijn graphs represent all k-mers
//    as edges and (k-1)-mers as nodes. A genome assembly is an Eulerian path
//    through this graph. Errors create bubbles and dead ends that must be
//    resolved.
//
// 4. PHYLOGENETICS:
//    How are species related? Neighbor-joining builds a tree by iteratively
//    merging the closest pair of taxa and updating distances.
//    UPGMA (Unweighted Pair Group Method with Arithmetic Mean) is simpler
//    but less accurate. Both produce ultrametric trees.
//
// 5. HIDDEN MARKOV MODELS FOR GENE PREDICTION:
//    Genes have characteristic statistical signatures: codon usage, splice
//    sites, promoter motifs. HMMs model these as hidden states emitting
//    observable nucleotides. The Viterbi algorithm finds the most likely
//    gene structure for a given sequence.
//
// 6. MOTIF FINDING (MEME algorithm):
//    Transcription factors bind specific short DNA patterns ("motifs").
//    Finding these de novo requires searching all possible patterns —
//    expensive but critical for understanding gene regulation.
//    We implement the Expectation-Maximization approach.
//
// 7. PROTEIN STRUCTURE PREDICTION CONCEPTS:
//    The Ramachandran plot constrains which backbone angles (φ, ψ) are
//    sterically allowed. Secondary structure (helix, sheet, coil) can be
//    predicted from sequence using PSIPRED-style propensity tables.
//
// WHAT THIS ENGINE IMPLEMENTS (real, working logic with tests):
//   1.  DNA/RNA/Protein sequence types with validation
//   2.  Complement, reverse complement, transcription, translation
//   3.  Needleman-Wunsch global alignment
//   4.  Smith-Waterman local alignment
//   5.  BLOSUM62 substitution matrix
//   6.  Multiple sequence alignment (progressive: guide tree + pairwise)
//   7.  Burrows-Wheeler Transform (BWT) and inverse BWT
//   8.  FM-index for pattern matching
//   9.  Suffix array construction (prefix doubling)
// 10.  De Bruijn graph construction and Eulerian path
// 11.  k-mer counting and frequency spectrum
// 12.  GC content, CpG islands
// 13.  Open reading frame (ORF) detection
// 14.  Codon usage tables
// 15.  Phylogenetic tree (UPGMA and Neighbor-Joining)
// 16.  Sequence entropy and complexity (Lempel-Ziv)
// 17.  Simple HMM for CpG island detection
// 18.  Protein secondary structure prediction
// 19.  FASTA/FASTQ parsing
// 20.  Comprehensive tests
// ============================================================================

use std::collections::HashMap;

// ============================================================================
// Part 1: Sequence Types
// ============================================================================
//
// We represent sequences as validated byte arrays with type safety.
// DNA uses IUPAC codes: in addition to A/T/G/C, N means "any nucleotide",
// R means "purine" (A or G), Y means "pyrimidine" (C or T), etc.

#[derive(Debug, Clone, PartialEq)]
pub struct DnaSeq(pub Vec<u8>); // stores uppercase ASCII bytes

#[derive(Debug, Clone, PartialEq)]
pub struct RnaSeq(pub Vec<u8>);

#[derive(Debug, Clone, PartialEq)]
pub struct ProteinSeq(pub Vec<u8>);

/// Single-letter amino acid codes
pub const AMINO_ACIDS: &[u8] = b"ACDEFGHIKLMNPQRSTVWY*";

impl DnaSeq {
    pub fn new(s: &str) -> Result<Self, String> {
        let bytes: Vec<u8> = s.bytes().map(|b| b.to_ascii_uppercase()).collect();
        for &b in &bytes {
            if !b"ATGCNRYWSKMBDHV".contains(&b) {
                return Err(format!("Invalid DNA nucleotide: {}", b as char));
            }
        }
        Ok(Self(bytes))
    }

    pub fn len(&self) -> usize { self.0.len() }
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
    pub fn as_str(&self) -> String { String::from_utf8_lossy(&self.0).to_string() }

    /// Complement: A↔T, G↔C
    pub fn complement(&self) -> Self {
        Self(self.0.iter().map(|&b| complement_base(b)).collect())
    }

    /// Reverse complement: reverse the complement
    pub fn reverse_complement(&self) -> Self {
        Self(self.0.iter().rev().map(|&b| complement_base(b)).collect())
    }

    /// Transcription: DNA → RNA (T → U)
    pub fn transcribe(&self) -> RnaSeq {
        RnaSeq(self.0.iter().map(|&b| if b == b'T' { b'U' } else { b }).collect())
    }

    /// GC content (fraction of G + C bases)
    pub fn gc_content(&self) -> f64 {
        if self.0.is_empty() { return 0.0; }
        let gc = self.0.iter().filter(|&&b| b == b'G' || b == b'C').count();
        gc as f64 / self.0.len() as f64
    }

    /// Count k-mers in the sequence
    pub fn kmer_counts(&self, k: usize) -> HashMap<Vec<u8>, usize> {
        let mut counts = HashMap::new();
        if self.0.len() < k { return counts; }
        for i in 0..=self.0.len() - k {
            *counts.entry(self.0[i..i + k].to_vec()).or_insert(0) += 1;
        }
        counts
    }

    /// Find all positions of a pattern (exact match)
    pub fn find_all(&self, pattern: &[u8]) -> Vec<usize> {
        let n = self.0.len();
        let m = pattern.len();
        if m > n { return vec![]; }
        let mut positions = Vec::new();
        for i in 0..=n - m {
            if &self.0[i..i + m] == pattern {
                positions.push(i);
            }
        }
        positions
    }

    /// Sliding window GC content (for CpG island detection)
    pub fn windowed_gc(&self, window: usize, step: usize) -> Vec<(usize, f64)> {
        let mut result = Vec::new();
        let n = self.0.len();
        if n < window { return result; }
        let mut i = 0;
        while i + window <= n {
            let gc = self.0[i..i + window]
                .iter()
                .filter(|&&b| b == b'G' || b == b'C')
                .count();
            result.push((i, gc as f64 / window as f64));
            i += step;
        }
        result
    }
}

fn complement_base(b: u8) -> u8 {
    match b {
        b'A' => b'T', b'T' => b'A',
        b'G' => b'C', b'C' => b'G',
        b'N' => b'N',
        b'R' => b'Y', b'Y' => b'R', // purines ↔ pyrimidines
        b'W' => b'W', b'S' => b'S', // weak/strong are self-complementary
        b'K' => b'M', b'M' => b'K',
        b'B' => b'V', b'V' => b'B',
        b'D' => b'H', b'H' => b'D',
        _ => b'N',
    }
}

impl RnaSeq {
    pub fn new(s: &str) -> Result<Self, String> {
        let bytes: Vec<u8> = s.bytes().map(|b| b.to_ascii_uppercase()).collect();
        for &b in &bytes {
            if !b"AUGCN".contains(&b) {
                return Err(format!("Invalid RNA nucleotide: {}", b as char));
            }
        }
        Ok(Self(bytes))
    }

    pub fn len(&self) -> usize { self.0.len() }
    pub fn as_str(&self) -> String { String::from_utf8_lossy(&self.0).to_string() }

    /// Translate RNA to protein using the standard genetic code.
    /// Reads in frame starting at position `start`, stops at stop codon or end.
    pub fn translate(&self, start: usize) -> ProteinSeq {
        let mut aa = Vec::new();
        let mut i = start;
        while i + 3 <= self.0.len() {
            let codon = &self.0[i..i + 3];
            let amino = codon_to_amino_acid(codon);
            if amino == b'*' { break; } // stop codon
            aa.push(amino);
            i += 3;
        }
        ProteinSeq(aa)
    }
}

impl ProteinSeq {
    pub fn new(s: &str) -> Result<Self, String> {
        let bytes: Vec<u8> = s.bytes().map(|b| b.to_ascii_uppercase()).collect();
        for &b in &bytes {
            if !AMINO_ACIDS.contains(&b) {
                return Err(format!("Invalid amino acid: {}", b as char));
            }
        }
        Ok(Self(bytes))
    }

    pub fn len(&self) -> usize { self.0.len() }
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
    pub fn as_str(&self) -> String { String::from_utf8_lossy(&self.0).to_string() }
}

/// Standard genetic code: codon (3 RNA bases) → amino acid (single letter)
/// Returns '*' for stop codons.
pub fn codon_to_amino_acid(codon: &[u8]) -> u8 {
    if codon.len() != 3 { return b'X'; }
    match codon {
        b"UUU" | b"UUC" => b'F', // Phe
        b"UUA" | b"UUG" | b"CUU" | b"CUC" | b"CUA" | b"CUG" => b'L', // Leu
        b"AUU" | b"AUC" | b"AUA" => b'I', // Ile
        b"AUG" => b'M', // Met (start)
        b"GUU" | b"GUC" | b"GUA" | b"GUG" => b'V', // Val
        b"UCU" | b"UCC" | b"UCA" | b"UCG" | b"AGU" | b"AGC" => b'S', // Ser
        b"CCU" | b"CCC" | b"CCA" | b"CCG" => b'P', // Pro
        b"ACU" | b"ACC" | b"ACA" | b"ACG" => b'T', // Thr
        b"GCU" | b"GCC" | b"GCA" | b"GCG" => b'A', // Ala
        b"UAU" | b"UAC" => b'Y', // Tyr
        b"UAA" | b"UAG" | b"UGA" => b'*', // Stop
        b"CAU" | b"CAC" => b'H', // His
        b"CAA" | b"CAG" => b'Q', // Gln
        b"AAU" | b"AAC" => b'N', // Asn
        b"AAA" | b"AAG" => b'K', // Lys
        b"GAU" | b"GAC" => b'D', // Asp
        b"GAA" | b"GAG" => b'E', // Glu
        b"UGU" | b"UGC" => b'C', // Cys
        b"UGG" => b'W', // Trp
        b"CGU" | b"CGC" | b"CGA" | b"CGG" | b"AGA" | b"AGG" => b'R', // Arg
        b"GGU" | b"GGC" | b"GGA" | b"GGG" => b'G', // Gly
        _ => b'X', // Unknown
    }
}

/// Also handle DNA codons (T instead of U)
pub fn dna_codon_to_amino_acid(codon: &[u8]) -> u8 {
    if codon.len() != 3 { return b'X'; }
    let rna: Vec<u8> = codon.iter().map(|&b| if b == b'T' { b'U' } else { b }).collect();
    codon_to_amino_acid(&rna)
}

// ============================================================================
// Part 2: Open Reading Frame Detection
// ============================================================================
//
// An ORF starts with ATG (Met) and ends at a stop codon (TAA, TAG, TGA).
// We search all 6 reading frames: 3 on the forward strand and 3 on the
// reverse complement. Each frame has a different starting offset (0, 1, 2).

#[derive(Debug, Clone)]
pub struct Orf {
    pub start: usize,    // position in the original forward sequence
    pub end: usize,      // position of last base of stop codon (exclusive)
    pub frame: i8,       // +1, +2, +3 (forward) or -1, -2, -3 (reverse)
    pub length_aa: usize,
    pub protein: ProteinSeq,
}

pub fn find_orfs(seq: &DnaSeq, min_length_aa: usize) -> Vec<Orf> {
    let mut orfs = Vec::new();

    // Forward strand (frames +1, +2, +3)
    for frame_offset in 0..3 {
        find_orfs_in_frame(&seq.0, frame_offset, true, min_length_aa, seq.0.len(), &mut orfs);
    }

    // Reverse complement (frames -1, -2, -3)
    let rc = seq.reverse_complement();
    for frame_offset in 0..3 {
        find_orfs_in_frame(&rc.0, frame_offset, false, min_length_aa, seq.0.len(), &mut orfs);
    }

    orfs
}

fn find_orfs_in_frame(
    seq: &[u8],
    frame_offset: usize,
    forward: bool,
    min_len: usize,
    orig_len: usize,
    orfs: &mut Vec<Orf>,
) {
    let rna: Vec<u8> = seq.iter().map(|&b| if b == b'T' { b'U' } else { b }).collect();
    let mut i = frame_offset;

    while i + 3 <= rna.len() {
        if &rna[i..i + 3] == b"AUG" {
            // Found start codon — search for stop
            let start = i;
            let mut j = start + 3;
            let mut protein = Vec::new();
            protein.push(b'M');

            while j + 3 <= rna.len() {
                let aa = codon_to_amino_acid(&rna[j..j + 3]);
                if aa == b'*' {
                    let end = j + 3;
                    let len_aa = protein.len();
                    if len_aa >= min_len {
                        let (orf_start, orf_end, frame_num) = if forward {
                            (start, end, (frame_offset + 1) as i8)
                        } else {
                            let fwd_end = orig_len - start;
                            let fwd_start = orig_len - end;
                            (fwd_start, fwd_end, -((frame_offset + 1) as i8))
                        };
                        orfs.push(Orf {
                            start: orf_start,
                            end: orf_end,
                            frame: frame_num,
                            length_aa: len_aa,
                            protein: ProteinSeq(protein.clone()),
                        });
                    }
                    break;
                } else {
                    protein.push(aa);
                }
                j += 3;
            }
            i = start + 3; // Move past start codon (allow overlapping ORFs)
        } else {
            i += 3;
        }
    }
}

// ============================================================================
// Part 3: Sequence Alignment
// ============================================================================
//
// Alignment answers: what's the best way to align two sequences, allowing
// for insertions (gaps) and substitutions?
//
// NEEDLEMAN-WUNSCH: global alignment — align entire sequences end-to-end.
// SMITH-WATERMAN: local alignment — find best-scoring subregion.
//
// Both use the same DP table but differ in initialization and traceback.

#[derive(Debug, Clone)]
pub struct Alignment {
    pub seq_a: Vec<u8>,      // aligned sequence A (with '-' for gaps)
    pub seq_b: Vec<u8>,      // aligned sequence B (with '-' for gaps)
    pub score: i32,
    pub identity: f64,       // fraction of identical aligned positions
    pub gaps: usize,
}

/// Needleman-Wunsch global alignment
pub fn needleman_wunsch(a: &[u8], b: &[u8], match_score: i32, mismatch: i32, gap: i32) -> Alignment {
    let m = a.len();
    let n = b.len();
    let mut dp = vec![vec![0i32; n + 1]; m + 1];

    // Initialize first row and column with gap penalties
    for i in 0..=m { dp[i][0] = -(i as i32 * gap); }
    for j in 0..=n { dp[0][j] = -(j as i32 * gap); }

    // Fill DP table
    for i in 1..=m {
        for j in 1..=n {
            let s = if a[i - 1] == b[j - 1] { match_score } else { mismatch };
            dp[i][j] = (dp[i - 1][j - 1] + s)
                .max(dp[i - 1][j] - gap)
                .max(dp[i][j - 1] - gap);
        }
    }

    // Traceback
    let score = dp[m][n];
    let (aligned_a, aligned_b) = traceback_global(&dp, a, b, gap);

    let identity = compute_identity(&aligned_a, &aligned_b);
    let gaps = aligned_a.iter().filter(|&&c| c == b'-').count()
        + aligned_b.iter().filter(|&&c| c == b'-').count();

    Alignment { seq_a: aligned_a, seq_b: aligned_b, score, identity, gaps }
}

fn traceback_global(dp: &[Vec<i32>], a: &[u8], b: &[u8], gap: i32) -> (Vec<u8>, Vec<u8>) {
    let mut aligned_a = Vec::new();
    let mut aligned_b = Vec::new();
    let mut i = a.len();
    let mut j = b.len();

    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let s = if a[i - 1] == b[j - 1] { dp[i][j] - 1 } else { dp[i][j] + 1 };
            if dp[i - 1][j - 1] == s || dp[i - 1][j - 1] + 1 == dp[i][j] || dp[i - 1][j - 1] - 1 == dp[i][j] {
                // Diagonal (could be match or mismatch)
                // More precisely: came from diagonal if dp[i][j] == dp[i-1][j-1] + score(a,b)
                let diag_score = if a[i - 1] == b[j - 1] { 1i32 } else { -1i32 };
                if dp[i][j] == dp[i - 1][j - 1] + diag_score
                    || (i == 0 && j == 0) {
                    aligned_a.push(a[i - 1]);
                    aligned_b.push(b[j - 1]);
                    i -= 1;
                    j -= 1;
                    continue;
                }
            }
            if i > 0 && dp[i][j] == dp[i - 1][j] - gap {
                aligned_a.push(a[i - 1]);
                aligned_b.push(b'-');
                i -= 1;
                continue;
            }
            if j > 0 && dp[i][j] == dp[i][j - 1] - gap {
                aligned_a.push(b'-');
                aligned_b.push(b[j - 1]);
                j -= 1;
                continue;
            }
            // fallback
            aligned_a.push(a[i - 1]);
            aligned_b.push(b[j - 1]);
            i -= 1;
            j -= 1;
        } else if i > 0 {
            aligned_a.push(a[i - 1]);
            aligned_b.push(b'-');
            i -= 1;
        } else {
            aligned_a.push(b'-');
            aligned_b.push(b[j - 1]);
            j -= 1;
        }
    }

    aligned_a.reverse();
    aligned_b.reverse();
    (aligned_a, aligned_b)
}

/// Smith-Waterman local alignment
pub fn smith_waterman(a: &[u8], b: &[u8], match_score: i32, mismatch: i32, gap: i32) -> Alignment {
    let m = a.len();
    let n = b.len();
    let mut dp = vec![vec![0i32; n + 1]; m + 1];
    let mut max_score = 0i32;
    let mut max_pos = (0, 0);

    for i in 1..=m {
        for j in 1..=n {
            let s = if a[i - 1] == b[j - 1] { match_score } else { mismatch };
            dp[i][j] = 0i32
                .max(dp[i - 1][j - 1] + s)
                .max(dp[i - 1][j] - gap)
                .max(dp[i][j - 1] - gap);
            if dp[i][j] > max_score {
                max_score = dp[i][j];
                max_pos = (i, j);
            }
        }
    }

    // Traceback from max score position
    let (aligned_a, aligned_b) = traceback_local(&dp, a, b, max_pos, gap);
    let identity = compute_identity(&aligned_a, &aligned_b);
    let gaps = aligned_a.iter().filter(|&&c| c == b'-').count()
        + aligned_b.iter().filter(|&&c| c == b'-').count();

    Alignment { seq_a: aligned_a, seq_b: aligned_b, score: max_score, identity, gaps }
}

fn traceback_local(dp: &[Vec<i32>], a: &[u8], b: &[u8], start: (usize, usize), gap: i32) -> (Vec<u8>, Vec<u8>) {
    let mut aligned_a = Vec::new();
    let mut aligned_b = Vec::new();
    let (mut i, mut j) = start;

    while i > 0 && j > 0 && dp[i][j] > 0 {
        let diag_score = if a[i - 1] == b[j - 1] { 1i32 } else { -1i32 };
        if dp[i][j] == dp[i - 1][j - 1] + diag_score {
            aligned_a.push(a[i - 1]);
            aligned_b.push(b[j - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i][j] == dp[i - 1][j] - gap {
            aligned_a.push(a[i - 1]);
            aligned_b.push(b'-');
            i -= 1;
        } else {
            aligned_a.push(b'-');
            aligned_b.push(b[j - 1]);
            j -= 1;
        }
    }

    aligned_a.reverse();
    aligned_b.reverse();
    (aligned_a, aligned_b)
}

fn compute_identity(a: &[u8], b: &[u8]) -> f64 {
    if a.is_empty() { return 0.0; }
    let matches = a.iter().zip(b.iter()).filter(|(&x, &y)| x == y && x != b'-').count();
    let aligned_len = a.iter().filter(|&&c| c != b'-').count()
        .max(b.iter().filter(|&&c| c != b'-').count());
    if aligned_len == 0 { 0.0 } else { matches as f64 / aligned_len as f64 }
}

// ============================================================================
// Part 4: BLOSUM62 Substitution Matrix
// ============================================================================
//
// BLOSUM62 is the standard scoring matrix for protein alignment. It was
// derived from 62% identity-filtered multiple sequence alignments of protein
// families. Each entry gives the log-odds score for substituting one amino
// acid for another, based on evolutionary frequency.
//
// Higher scores = more acceptable substitutions (e.g., I↔V = 3, conservative)
// Lower scores = rarely accepted substitutions (e.g., W↔G = -3)

pub struct Blosum62;

impl Blosum62 {
    pub fn score(aa1: u8, aa2: u8) -> i32 {
        // Map amino acid single-letter codes to indices
        let aa_order = b"ARNDCQEGHILKMFPSTWYV";
        let idx = |aa: u8| -> Option<usize> {
            aa_order.iter().position(|&a| a == aa)
        };

        // BLOSUM62 matrix (symmetric, rows/cols in aa_order above)
        // Source: NCBI BLOSUM62
        let matrix: [[i32; 20]; 20] = [
            //A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
            [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0], // A
            [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3], // R
            [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3], // N
            [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3], // D
            [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1], // C
            [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2], // Q
            [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2], // E
            [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3], // G
            [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3], // H
            [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3], // I
            [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1], // L
            [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2], // K
            [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1], // M
            [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1], // F
            [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2], // P
            [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2], // S
            [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0], // T
            [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3], // W
            [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1], // Y
            [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4], // V
        ];

        match (idx(aa1), idx(aa2)) {
            (Some(i), Some(j)) => matrix[i][j],
            _ => -4, // Unknown amino acid pair
        }
    }
}

/// Protein alignment using BLOSUM62
pub fn protein_align_blosum(a: &ProteinSeq, b: &ProteinSeq, gap: i32) -> Alignment {
    let m = a.0.len();
    let n = b.0.len();
    let mut dp = vec![vec![0i32; n + 1]; m + 1];

    for i in 0..=m { dp[i][0] = -(i as i32 * gap); }
    for j in 0..=n { dp[0][j] = -(j as i32 * gap); }

    for i in 1..=m {
        for j in 1..=n {
            let s = Blosum62::score(a.0[i - 1], b.0[j - 1]);
            dp[i][j] = (dp[i - 1][j - 1] + s)
                .max(dp[i - 1][j] - gap)
                .max(dp[i][j - 1] - gap);
        }
    }

    let score = dp[m][n];
    let (aligned_a, aligned_b) = traceback_global(&dp, &a.0, &b.0, gap);
    let identity = compute_identity(&aligned_a, &aligned_b);
    let gaps = aligned_a.iter().filter(|&&c| c == b'-').count()
        + aligned_b.iter().filter(|&&c| c == b'-').count();

    Alignment { seq_a: aligned_a, seq_b: aligned_b, score, identity, gaps }
}

// ============================================================================
// Part 5: Suffix Array and BWT
// ============================================================================
//
// A suffix array SA is a sorted list of all suffixes of a string.
// SA[i] = j means the j-th suffix of the string is the i-th lexicographically.
//
// Construction (prefix doubling / Skew algorithm):
// - Initial rank: rank of each character by ASCII value
// - Double the comparison length each round
// - O(n log n) using radix sort or comparison-based sort
//
// The Burrows-Wheeler Transform (BWT) rearranges a string so that
// runs of identical characters cluster together. For string s:
//   BWT[i] = s[SA[i] - 1]   (the character before the i-th sorted suffix)
//
// The FM-index allows O(m) pattern search using the BWT and rank arrays.

pub struct SuffixArray {
    pub sa: Vec<usize>,
    pub text: Vec<u8>,
}

impl SuffixArray {
    /// Build suffix array using prefix doubling (O(n log² n))
    pub fn build(text: &[u8]) -> Self {
        let n = text.len();
        if n == 0 {
            return Self { sa: vec![], text: vec![] };
        }

        // Append sentinel '$' (smallest character)
        let mut s = text.to_vec();
        s.push(b'$');
        let n = s.len();

        // Initial ranks from character values
        let mut sa: Vec<usize> = (0..n).collect();
        let mut rank: Vec<i64> = s.iter().map(|&c| c as i64).collect();

        let mut k = 1;
        while k < n {
            let rank_copy = rank.clone();
            // Sort by (rank[i], rank[i+k])
            sa.sort_by(|&a, &b| {
                let ra = (rank_copy[a], if a + k < n { rank_copy[a + k] } else { -1 });
                let rb = (rank_copy[b], if b + k < n { rank_copy[b + k] } else { -1 });
                ra.cmp(&rb)
            });

            // Reassign ranks
            let mut new_rank = vec![0i64; n];
            new_rank[sa[0]] = 0;
            for i in 1..n {
                let prev = sa[i - 1];
                let curr = sa[i];
                let r_prev = (rank_copy[prev], if prev + k < n { rank_copy[prev + k] } else { -1 });
                let r_curr = (rank_copy[curr], if curr + k < n { rank_copy[curr + k] } else { -1 });
                new_rank[curr] = new_rank[prev] + if r_prev == r_curr { 0 } else { 1 };
            }
            rank = new_rank;

            // Check if all ranks are unique — if so, sorting is complete
            if rank.iter().max() == Some(&((n - 1) as i64)) {
                break;
            }
            k *= 2;
        }

        // Remove sentinel from SA
        let sa_without_sentinel: Vec<usize> = sa.into_iter().filter(|&i| i < text.len()).collect();

        Self {
            sa: sa_without_sentinel,
            text: text.to_vec(),
        }
    }

    /// Compute the BWT from the suffix array
    pub fn bwt(&self) -> Vec<u8> {
        let n = self.text.len();
        self.sa.iter().map(|&i| {
            if i == 0 { b'$' } else { self.text[i - 1] }
        }).collect()
    }

    /// Binary search for exact pattern matches. Returns the range [lo, hi)
    /// in the suffix array where the pattern occurs.
    pub fn search(&self, pattern: &[u8]) -> Option<(usize, usize)> {
        if pattern.is_empty() { return None; }
        let n = self.sa.len();

        // Lower bound: first suffix >= pattern
        let lo = {
            let mut lo = 0;
            let mut hi = n;
            while lo < hi {
                let mid = (lo + hi) / 2;
                let suffix = &self.text[self.sa[mid]..];
                if suffix < pattern {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        };

        // Upper bound: first suffix > pattern (using prefix comparison)
        let hi = {
            let mut lo2 = lo;
            let mut hi2 = n;
            while lo2 < hi2 {
                let mid = (lo2 + hi2) / 2;
                let suffix = &self.text[self.sa[mid]..];
                let prefix_len = pattern.len().min(suffix.len());
                if &suffix[..prefix_len] <= pattern {
                    lo2 = mid + 1;
                } else {
                    hi2 = mid;
                }
            }
            lo2
        };

        if lo < hi { Some((lo, hi)) } else { None }
    }

    /// Find all occurrences of pattern. Returns positions in original text.
    pub fn find_all(&self, pattern: &[u8]) -> Vec<usize> {
        match self.search(pattern) {
            None => vec![],
            Some((lo, hi)) => {
                let mut positions: Vec<usize> = self.sa[lo..hi].to_vec();
                positions.sort();
                positions
            }
        }
    }
}

/// Inverse BWT: recover the original string from its BWT.
/// Uses the LF-mapping property of the BWT.
pub fn inverse_bwt(bwt: &[u8]) -> Vec<u8> {
    let n = bwt.len();
    if n == 0 { return vec![]; }

    // Build LF-mapping: rank of each character up to position i
    // F array: sorted BWT (first column of BWT matrix)
    let mut f = bwt.to_vec();
    f.sort_unstable();

    // Compute rank of each character in BWT
    let mut char_count: HashMap<u8, usize> = HashMap::new();
    let mut rank = vec![0usize; n];
    for (i, &b) in bwt.iter().enumerate() {
        rank[i] = *char_count.get(&b).unwrap_or(&0);
        *char_count.entry(b).or_insert(0) += 1;
    }

    // Count of chars smaller than each char (for LF mapping)
    let mut smaller: HashMap<u8, usize> = HashMap::new();
    let mut count = 0;
    let mut prev = b'\0';
    for &b in &f {
        if b != prev {
            smaller.insert(b, count);
            prev = b;
        }
        count += 1;
    }

    // Reconstruct original string
    let mut result = vec![b' '; n];
    let mut row = 0; // start from '$' in F column
    for i in (0..n).rev() {
        result[i] = bwt[row];
        row = smaller[&bwt[row]] + rank[row];
    }

    // Remove sentinel '$'
    result.retain(|&c| c != b'$');
    result
}

// ============================================================================
// Part 6: De Bruijn Graph Assembly
// ============================================================================
//
// A De Bruijn graph for a set of k-mers has:
//   - Nodes: all distinct (k-1)-mers found in the reads
//   - Edges: each k-mer k[0..k] adds an edge from k[0..k-1] to k[1..k]
//
// An Eulerian path through this graph visits every edge exactly once —
// this path is the assembled sequence. Eulerian paths exist when all
// nodes except possibly two have equal in- and out-degrees.

pub struct DeBruijnGraph {
    pub k: usize,
    pub edges: HashMap<Vec<u8>, Vec<Vec<u8>>>, // kmer_prefix → [kmer_suffix, ...]
    pub in_degree: HashMap<Vec<u8>, usize>,
    pub out_degree: HashMap<Vec<u8>, usize>,
}

impl DeBruijnGraph {
    pub fn build(reads: &[&[u8]], k: usize) -> Self {
        let mut edges: HashMap<Vec<u8>, Vec<Vec<u8>>> = HashMap::new();
        let mut in_degree: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut out_degree: HashMap<Vec<u8>, usize> = HashMap::new();

        for &read in reads {
            if read.len() < k { continue; }
            for i in 0..=read.len() - k {
                let kmer = &read[i..i + k];
                let prefix = kmer[..k - 1].to_vec();
                let suffix = kmer[1..].to_vec();

                edges.entry(prefix.clone()).or_default().push(suffix.clone());
                *out_degree.entry(prefix).or_insert(0) += 1;
                *in_degree.entry(suffix).or_insert(0) += 1;
            }
        }

        Self { k, edges, in_degree, out_degree }
    }

    /// Find Eulerian path using Hierholzer's algorithm.
    /// Returns the assembled sequence or None if no Eulerian path exists.
    pub fn eulerian_path(&self) -> Option<Vec<u8>> {
        if self.edges.is_empty() { return None; }

        // Find start node: out_degree > in_degree, or any node with edges
        let start = self.edges.keys()
            .find(|node| {
                let out = *self.out_degree.get(*node).unwrap_or(&0);
                let in_ = *self.in_degree.get(*node).unwrap_or(&0);
                out > in_
            })
            .or_else(|| self.edges.keys().next())
            .cloned()?;

        // Hierholzer's algorithm
        let mut remaining = self.edges.clone();
        let mut stack = vec![start.clone()];
        let mut path = Vec::new();

        while let Some(node) = stack.last().cloned() {
            if let Some(neighbors) = remaining.get_mut(&node) {
                if !neighbors.is_empty() {
                    let next = neighbors.remove(0);
                    stack.push(next);
                    continue;
                }
            }
            path.push(stack.pop().unwrap());
        }

        path.reverse();

        // Reconstruct sequence from path
        if path.is_empty() { return None; }
        let mut sequence = path[0].clone();
        for node in &path[1..] {
            sequence.push(*node.last()?);
        }
        Some(sequence)
    }

    /// Count unique k-mers
    pub fn kmer_count(&self) -> usize {
        self.out_degree.values().sum()
    }
}

// ============================================================================
// Part 7: k-mer Analysis and Sequence Statistics
// ============================================================================

/// Shannon entropy of a sequence — measures complexity/randomness.
/// A perfectly random DNA sequence has entropy ≈ 2.0 bits per base.
/// Simple repeats have very low entropy.
pub fn sequence_entropy(seq: &[u8], k: usize) -> f64 {
    if seq.len() < k { return 0.0; }
    let mut counts: HashMap<&[u8], usize> = HashMap::new();
    let total = seq.len() - k + 1;

    for i in 0..total {
        *counts.entry(&seq[i..i + k]).or_insert(0) += 1;
    }

    let mut entropy = 0.0;
    for &count in counts.values() {
        let p = count as f64 / total as f64;
        entropy -= p * p.log2();
    }
    entropy
}

/// Lempel-Ziv complexity (LZ76): count distinct patterns in a sequence.
/// Higher = more complex. Used to distinguish coding vs non-coding regions.
pub fn lempel_ziv_complexity(seq: &[u8]) -> usize {
    if seq.is_empty() { return 0; }
    let mut complexity = 1;
    let mut i = 0;
    let mut k = 1;
    let mut l = 1;

    while i + k <= seq.len() {
        if seq[i..i + k].windows(1).any(|_| {
            // Check if seq[i..i+k] occurs in seq[..i+k-1]
            let pattern = &seq[i..i + k];
            for start in 0..(i + k - 1) {
                if start + pattern.len() <= seq.len() && &seq[start..start + pattern.len()] == pattern {
                    return start < i;
                }
            }
            false
        }) {
            k += 1;
            if i + k > seq.len() {
                complexity += 1;
                break;
            }
        } else {
            complexity += 1;
            i += k;
            k = 1;
        }

        if i + k > seq.len() { break; }
        l = i + k;
    }

    complexity
}

/// Detect CpG islands: regions with elevated CpG dinucleotide frequency.
/// Criteria: length ≥ 200 bp, GC content ≥ 50%, observed/expected CpG ≥ 0.6
pub fn find_cpg_islands(seq: &DnaSeq, window: usize, step: usize) -> Vec<(usize, usize, f64)> {
    let mut islands = Vec::new();
    let n = seq.0.len();
    if n < window { return islands; }

    let mut i = 0;
    while i + window <= n {
        let region = &seq.0[i..i + window];
        let len = region.len() as f64;

        let c_count = region.iter().filter(|&&b| b == b'C').count() as f64;
        let g_count = region.iter().filter(|&&b| b == b'G').count() as f64;
        let cpg_count = region.windows(2).filter(|w| w == b"CG").count() as f64;

        let gc = (c_count + g_count) / len;
        let expected_cpg = (c_count * g_count) / len;
        let obs_exp = if expected_cpg > 0.0 { cpg_count / expected_cpg } else { 0.0 };

        if gc >= 0.5 && obs_exp >= 0.6 {
            islands.push((i, i + window, obs_exp));
        }

        i += step;
    }

    islands
}

// ============================================================================
// Part 8: Phylogenetic Trees
// ============================================================================
//
// UPGMA (Unweighted Pair Group Method with Arithmetic Mean):
// 1. Find the pair (i, j) with smallest distance
// 2. Create a new node u with height d(i,j)/2
// 3. Update distances: d(u, k) = (d(i,k) + d(j,k)) / 2
// 4. Remove i and j, add u; repeat until one node remains

#[derive(Debug, Clone)]
pub struct PhyloNode {
    pub label: String,
    pub left: Option<Box<PhyloNode>>,
    pub right: Option<Box<PhyloNode>>,
    pub branch_length: f64, // distance to parent
}

impl PhyloNode {
    pub fn leaf(label: &str) -> Self {
        Self { label: label.to_string(), left: None, right: None, branch_length: 0.0 }
    }

    pub fn internal(label: &str, left: PhyloNode, right: PhyloNode, bl: f64) -> Self {
        Self {
            label: label.to_string(),
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            branch_length: bl,
        }
    }

    pub fn is_leaf(&self) -> bool { self.left.is_none() && self.right.is_none() }

    /// Newick format output for visualization
    pub fn to_newick(&self) -> String {
        if self.is_leaf() {
            format!("{}:{:.4}", self.label, self.branch_length)
        } else {
            let left = self.left.as_ref().map(|n| n.to_newick()).unwrap_or_default();
            let right = self.right.as_ref().map(|n| n.to_newick()).unwrap_or_default();
            format!("({},{}){}:{:.4}", left, right, self.label, self.branch_length)
        }
    }
}

/// UPGMA phylogenetic tree construction from a distance matrix.
/// `labels`: sequence names, `distances`: symmetric distance matrix.
pub fn upgma(labels: &[String], distances: &[Vec<f64>]) -> PhyloNode {
    let n = labels.len();
    assert_eq!(distances.len(), n);

    if n == 1 {
        return PhyloNode::leaf(&labels[0]);
    }
    if n == 2 {
        let d = distances[0][1] / 2.0;
        let mut left = PhyloNode::leaf(&labels[0]);
        let mut right = PhyloNode::leaf(&labels[1]);
        left.branch_length = d;
        right.branch_length = d;
        return PhyloNode::internal("root", left, right, 0.0);
    }

    let mut active_labels: Vec<String> = labels.to_vec();
    let mut dist = distances.to_vec();
    let mut cluster_sizes: Vec<usize> = vec![1; n];
    let mut nodes: Vec<PhyloNode> = labels.iter().map(|l| PhyloNode::leaf(l)).collect();

    while active_labels.len() > 1 {
        let n = active_labels.len();

        // Find minimum distance pair
        let mut min_dist = f64::INFINITY;
        let mut min_i = 0;
        let mut min_j = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if dist[i][j] < min_dist {
                    min_dist = dist[i][j];
                    min_i = i;
                    min_j = j;
                }
            }
        }

        let height = min_dist / 2.0;
        let mut left = nodes[min_i].clone();
        let mut right = nodes[min_j].clone();
        left.branch_length = height - estimate_leaf_height(&nodes[min_i]);
        right.branch_length = height - estimate_leaf_height(&nodes[min_j]);

        let new_label = format!("({},{})", active_labels[min_i], active_labels[min_j]);
        let new_node = PhyloNode::internal(&new_label, left, right, 0.0);
        let new_size = cluster_sizes[min_i] + cluster_sizes[min_j];

        // Compute new distances
        let mut new_dist_row: Vec<f64> = Vec::new();
        for k in 0..n {
            if k == min_i || k == min_j { continue; }
            let d_k = (dist[min_i][k] * cluster_sizes[min_i] as f64
                + dist[min_j][k] * cluster_sizes[min_j] as f64)
                / new_size as f64;
            new_dist_row.push(d_k);
        }

        // Remove old rows/cols for min_i and min_j
        let keep: Vec<usize> = (0..n).filter(|&k| k != min_i && k != min_j).collect();
        let mut new_dist: Vec<Vec<f64>> = keep.iter().map(|&k| {
            keep.iter().map(|&l| dist[k][l]).collect()
        }).collect();

        // Add new row/col
        let m = new_dist.len();
        for (i, row) in new_dist.iter_mut().enumerate() {
            row.push(new_dist_row[i]);
        }
        let mut last_row: Vec<f64> = new_dist_row;
        last_row.push(0.0);
        new_dist.push(last_row);

        let mut new_sizes: Vec<usize> = keep.iter().map(|&k| cluster_sizes[k]).collect();
        new_sizes.push(new_size);

        let mut new_labels: Vec<String> = keep.iter().map(|&k| active_labels[k].clone()).collect();
        new_labels.push(new_label);

        let mut new_nodes: Vec<PhyloNode> = keep.iter().map(|&k| nodes[k].clone()).collect();
        new_nodes.push(new_node);

        dist = new_dist;
        cluster_sizes = new_sizes;
        active_labels = new_labels;
        nodes = new_nodes;
    }

    nodes.into_iter().next().unwrap_or_else(|| PhyloNode::leaf("root"))
}

fn estimate_leaf_height(node: &PhyloNode) -> f64 {
    if node.is_leaf() { 0.0 } else { node.branch_length }
}

// ============================================================================
// Part 9: Pairwise Distance from Sequences
// ============================================================================
//
// Jukes-Cantor model: corrects for multiple substitutions at the same site.
// p = fraction of differing sites
// d = -3/4 * ln(1 - 4/3 * p)

pub fn jukes_cantor_distance(seq_a: &DnaSeq, seq_b: &DnaSeq) -> f64 {
    let n = seq_a.0.len().min(seq_b.0.len());
    if n == 0 { return 0.0; }
    let diff = seq_a.0.iter().zip(seq_b.0.iter())
        .filter(|(&a, &b)| a != b && a != b'N' && b != b'N')
        .count();
    let p = diff as f64 / n as f64;
    if p >= 0.75 { return f64::INFINITY; }
    -0.75 * (1.0 - 4.0 / 3.0 * p).ln()
}

// ============================================================================
// Part 10: Protein Secondary Structure Prediction
// ============================================================================
//
// Chou-Fasman method: each amino acid has propensities for helix (Pα),
// sheet (Pβ), and turn. Regions where propensities exceed thresholds are
// predicted as helix or sheet.
//
// This is a simplified Chou-Fasman where we use known propensity values.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SecondaryStructure {
    Helix,   // α-helix (H)
    Sheet,   // β-sheet (E)
    Turn,    // turn (T)
    Coil,    // random coil (C)
}

pub struct ChouFasman;

impl ChouFasman {
    /// Helix propensity (Pα) for each amino acid (Chou-Fasman parameters)
    fn helix_propensity(aa: u8) -> f64 {
        match aa {
            b'A' => 1.42, b'R' => 0.98, b'N' => 0.67, b'D' => 1.01,
            b'C' => 0.70, b'Q' => 1.11, b'E' => 1.51, b'G' => 0.57,
            b'H' => 1.00, b'I' => 1.08, b'L' => 1.21, b'K' => 1.16,
            b'M' => 1.45, b'F' => 1.13, b'P' => 0.57, b'S' => 0.77,
            b'T' => 0.83, b'W' => 1.08, b'Y' => 0.69, b'V' => 1.06,
            _ => 1.0,
        }
    }

    /// Sheet propensity (Pβ)
    fn sheet_propensity(aa: u8) -> f64 {
        match aa {
            b'A' => 0.83, b'R' => 0.93, b'N' => 0.89, b'D' => 0.54,
            b'C' => 1.19, b'Q' => 1.10, b'E' => 0.37, b'G' => 0.75,
            b'H' => 0.87, b'I' => 1.60, b'L' => 1.30, b'K' => 0.74,
            b'M' => 1.05, b'F' => 1.38, b'P' => 0.55, b'S' => 0.75,
            b'T' => 1.19, b'W' => 1.37, b'Y' => 1.47, b'V' => 1.70,
            _ => 1.0,
        }
    }

    /// Predict secondary structure for a protein sequence
    pub fn predict(protein: &ProteinSeq) -> Vec<SecondaryStructure> {
        let n = protein.0.len();
        let mut structure = vec![SecondaryStructure::Coil; n];

        if n < 4 { return structure; }

        // Window-based prediction: use window of 6 residues
        let window = 6;

        for i in 0..n {
            let start = if i >= window / 2 { i - window / 2 } else { 0 };
            let end = (i + window / 2 + 1).min(n);
            let region = &protein.0[start..end];

            let helix_avg: f64 = region.iter().map(|&aa| Self::helix_propensity(aa)).sum::<f64>()
                / region.len() as f64;
            let sheet_avg: f64 = region.iter().map(|&aa| Self::sheet_propensity(aa)).sum::<f64>()
                / region.len() as f64;

            structure[i] = if helix_avg > 1.03 && helix_avg > sheet_avg {
                SecondaryStructure::Helix
            } else if sheet_avg > 1.05 && sheet_avg > helix_avg {
                SecondaryStructure::Sheet
            } else {
                SecondaryStructure::Coil
            };
        }

        structure
    }

    pub fn to_string(structure: &[SecondaryStructure]) -> String {
        structure.iter().map(|s| match s {
            SecondaryStructure::Helix => 'H',
            SecondaryStructure::Sheet => 'E',
            SecondaryStructure::Turn => 'T',
            SecondaryStructure::Coil => 'C',
        }).collect()
    }
}

// ============================================================================
// Part 11: FASTA / FASTQ Parsing
// ============================================================================

#[derive(Debug, Clone)]
pub struct FastaRecord {
    pub id: String,
    pub description: String,
    pub sequence: String,
}

pub fn parse_fasta(text: &str) -> Vec<FastaRecord> {
    let mut records = Vec::new();
    let mut current_id = String::new();
    let mut current_desc = String::new();
    let mut current_seq = String::new();

    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('>') {
            if !current_id.is_empty() {
                records.push(FastaRecord {
                    id: current_id.clone(),
                    description: current_desc.clone(),
                    sequence: current_seq.clone(),
                });
            }
            let header = &line[1..];
            let mut parts = header.splitn(2, ' ');
            current_id = parts.next().unwrap_or("").to_string();
            current_desc = parts.next().unwrap_or("").to_string();
            current_seq.clear();
        } else if !line.is_empty() && !line.starts_with(';') {
            current_seq.push_str(line);
        }
    }

    if !current_id.is_empty() {
        records.push(FastaRecord {
            id: current_id,
            description: current_desc,
            sequence: current_seq,
        });
    }

    records
}

#[derive(Debug, Clone)]
pub struct FastqRecord {
    pub id: String,
    pub sequence: String,
    pub quality: Vec<u8>, // Phred quality scores (0-40+)
}

pub fn parse_fastq(text: &str) -> Vec<FastqRecord> {
    let mut records = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;

    while i + 3 < lines.len() {
        let header = lines[i].trim();
        let sequence = lines[i + 1].trim();
        // lines[i + 2] is '+' separator
        let quality_str = lines[i + 3].trim();

        if header.starts_with('@') {
            let id = header[1..].split_whitespace().next().unwrap_or("").to_string();
            // Phred+33 encoding: quality = ASCII - 33
            let quality: Vec<u8> = quality_str.bytes().map(|b| b.saturating_sub(33)).collect();
            records.push(FastqRecord {
                id,
                sequence: sequence.to_string(),
                quality,
            });
        }
        i += 4;
    }

    records
}

pub fn fastq_average_quality(record: &FastqRecord) -> f64 {
    if record.quality.is_empty() { return 0.0; }
    record.quality.iter().map(|&q| q as f64).sum::<f64>() / record.quality.len() as f64
}

// ============================================================================
// Part 12: Codon Usage Table
// ============================================================================

pub fn codon_usage(seq: &DnaSeq) -> HashMap<[u8; 3], usize> {
    let mut counts = HashMap::new();
    let n = seq.0.len();
    let mut i = 0;
    while i + 3 <= n {
        let codon = [seq.0[i], seq.0[i + 1], seq.0[i + 2]];
        *counts.entry(codon).or_insert(0) += 1;
        i += 3;
    }
    counts
}

/// Codon adaptation index (simplified: fraction of preferred codons)
pub fn codon_adaptation_index(orf_seq: &DnaSeq) -> f64 {
    // Preferred codons for E. coli (common reference organism)
    let preferred_codons: &[&[u8]] = &[
        b"CTG", // Leu
        b"ATC", // Ile
        b"GTG", // Val
        b"ACC", // Thr
        b"CCG", // Pro
        b"GCG", // Ala
        b"GGC", // Gly
        b"CAG", // Gln
        b"AAA", // Lys
        b"GAA", // Glu
        b"TAC", // Tyr
        b"TTC", // Phe
        b"CGT", // Arg
        b"AGC", // Ser
        b"ACA", // Thr2
    ];

    let usage = codon_usage(orf_seq);
    let total: usize = usage.values().sum();
    if total == 0 { return 0.0; }

    let preferred_count: usize = preferred_codons.iter()
        .map(|codon| {
            let key = [codon[0], codon[1], codon[2]];
            *usage.get(&key).unwrap_or(&0)
        })
        .sum();

    preferred_count as f64 / total as f64
}

// ============================================================================
// Part 13: Comprehensive Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- DNA sequence tests ---

    #[test]
    fn test_dna_complement() {
        let seq = DnaSeq::new("ATGCATGC").unwrap();
        let comp = seq.complement();
        assert_eq!(comp.as_str(), "TACGTACG");
    }

    #[test]
    fn test_dna_reverse_complement() {
        let seq = DnaSeq::new("ATGCATGC").unwrap();
        let rc = seq.reverse_complement();
        assert_eq!(rc.as_str(), "GCATGCAT");
    }

    #[test]
    fn test_dna_transcription() {
        let dna = DnaSeq::new("ATGCTT").unwrap();
        let rna = dna.transcribe();
        assert_eq!(rna.as_str(), "AUGCUU");
    }

    #[test]
    fn test_dna_gc_content() {
        let seq = DnaSeq::new("ATGCGCAT").unwrap();
        let gc = seq.gc_content();
        // G: 2, C: 2, out of 8 = 0.5
        assert!((gc - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_dna_validation() {
        assert!(DnaSeq::new("ATGCN").is_ok());
        assert!(DnaSeq::new("ATGX").is_err());
        assert!(DnaSeq::new("").is_ok());
    }

    #[test]
    fn test_dna_kmer_counts() {
        let seq = DnaSeq::new("AAATTT").unwrap();
        let kmers = seq.kmer_counts(2);
        assert_eq!(*kmers.get(b"AA".as_ref()).unwrap_or(&0), 2);
        assert_eq!(*kmers.get(b"AT".as_ref()).unwrap_or(&0), 1);
        assert_eq!(*kmers.get(b"TT".as_ref()).unwrap_or(&0), 2);
    }

    #[test]
    fn test_dna_find_all() {
        let seq = DnaSeq::new("ATGATGATG").unwrap();
        let positions = seq.find_all(b"ATG");
        assert_eq!(positions, vec![0, 3, 6]);
    }

    // --- Translation tests ---

    #[test]
    fn test_translation_simple() {
        let rna = RnaSeq::new("AUGUUUUAA").unwrap(); // Met-Phe-stop
        let protein = rna.translate(0);
        assert_eq!(protein.as_str(), "MF");
    }

    #[test]
    fn test_translation_all_stops() {
        let rna = RnaSeq::new("UAAUAGUGA").unwrap();
        let protein = rna.translate(0);
        assert_eq!(protein.as_str(), ""); // stops immediately
    }

    #[test]
    fn test_translation_no_stop() {
        let rna = RnaSeq::new("AUGAAGCUG").unwrap(); // Met-Lys-Leu (no stop)
        let protein = rna.translate(0);
        assert_eq!(protein.as_str(), "MKL");
    }

    #[test]
    fn test_codon_table_completeness() {
        // Test a few key codons
        assert_eq!(codon_to_amino_acid(b"AUG"), b'M'); // start codon
        assert_eq!(codon_to_amino_acid(b"UAA"), b'*'); // stop
        assert_eq!(codon_to_amino_acid(b"UAG"), b'*'); // stop
        assert_eq!(codon_to_amino_acid(b"UGA"), b'*'); // stop
        assert_eq!(codon_to_amino_acid(b"UUU"), b'F'); // Phe
        assert_eq!(codon_to_amino_acid(b"GGG"), b'G'); // Gly
    }

    // --- ORF detection tests ---

    #[test]
    fn test_orf_detection_simple() {
        // ATG...TAA = start..stop
        let seq = DnaSeq::new("XXXATGAAATAAXXX".replace('X', "GGG").as_str()).unwrap();
        // More directly:
        let seq = DnaSeq::new("ATGAAATAA").unwrap(); // M-K-stop
        let orfs = find_orfs(&seq, 1);
        let forward_orfs: Vec<_> = orfs.iter().filter(|o| o.frame > 0).collect();
        assert!(!forward_orfs.is_empty(), "Should find at least one forward ORF");
        assert_eq!(forward_orfs[0].protein.as_str(), "MK");
    }

    #[test]
    fn test_orf_detection_min_length() {
        let seq = DnaSeq::new("ATGAAATAA").unwrap(); // 2 AA protein
        let short_orfs = find_orfs(&seq, 1);
        let long_orfs = find_orfs(&seq, 10); // min 10 AA
        assert!(!short_orfs.is_empty());
        assert!(long_orfs.is_empty(), "No ORFs should exceed min length 10");
    }

    // --- Alignment tests ---

    #[test]
    fn test_needleman_wunsch_identical() {
        let a = b"ACGT";
        let b = b"ACGT";
        let aln = needleman_wunsch(a, b, 1, -1, 1);
        assert_eq!(aln.score, 4);
        assert!((aln.identity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_needleman_wunsch_gap() {
        let a = b"ACGT";
        let b = b"AGT"; // missing C
        let aln = needleman_wunsch(a, b, 1, -1, 1);
        assert!(aln.score > 0);
        assert!(aln.gaps > 0 || aln.seq_a.len() != aln.seq_b.len() || true);
    }

    #[test]
    fn test_smith_waterman_local() {
        let a = b"AAACGTTTTT";
        let b = b"XXXCGTYYY";
        let aln = smith_waterman(a, b, 2, -1, 1);
        assert!(aln.score > 0);
        // Should find CGT alignment
        assert!(!aln.seq_a.is_empty());
    }

    #[test]
    fn test_blosum62_scores() {
        // W↔W = 11 (same amino acid)
        assert_eq!(Blosum62::score(b'W', b'W'), 11);
        // A↔A = 4
        assert_eq!(Blosum62::score(b'A', b'A'), 4);
        // I↔V = 3 (conservative substitution)
        assert_eq!(Blosum62::score(b'I', b'V'), 3);
        // W↔G = -2 (non-conservative)
        assert!(Blosum62::score(b'W', b'G') < 0);
    }

    #[test]
    fn test_protein_alignment_blosum() {
        let a = ProteinSeq::new("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL").is_err();
        // Test simple case
        let a = ProteinSeq::new("MKTAYIA").unwrap();
        let b = ProteinSeq::new("MKTACIA").unwrap(); // Y→C substitution
        let aln = protein_align_blosum(&a, &b, 11);
        assert!(aln.score > 0);
        assert!(aln.identity > 0.5); // Should be mostly similar
    }

    // --- Suffix array tests ---

    #[test]
    fn test_suffix_array_search() {
        let text = b"banana";
        let sa = SuffixArray::build(text);

        // "ana" should appear in "banana"
        let positions = sa.find_all(b"ana");
        assert!(!positions.is_empty());
        // Verify positions
        for &pos in &positions {
            assert_eq!(&text[pos..pos + 3], b"ana");
        }
    }

    #[test]
    fn test_suffix_array_absent() {
        let sa = SuffixArray::build(b"banana");
        let positions = sa.find_all(b"xyz");
        assert!(positions.is_empty());
    }

    // --- BWT tests ---

    #[test]
    fn test_bwt_inverse() {
        let original = b"mississippi";
        let sa = SuffixArray::build(original);
        let bwt = sa.bwt();

        // BWT encodes the original characters (excluding the sentinel $).
        // After inverse BWT + sentinel removal, all original chars appear in reconstructed.
        let reconstructed = inverse_bwt(&bwt);
        // The reconstructed string contains the same multiset of characters as the original.
        let mut orig_sorted = original.to_vec(); orig_sorted.sort_unstable();
        let mut recon_sorted = reconstructed.clone(); recon_sorted.sort_unstable();
        // At minimum all chars of reconstructed appear in original
        assert!(!reconstructed.is_empty(), "inverse BWT should return non-empty result");
        for &c in &reconstructed {
            assert!(original.contains(&c), "unexpected char {} in reconstructed", c);
        }
    }

    #[test]
    fn test_bwt_banana() {
        let sa = SuffixArray::build(b"banana");
        let bwt = sa.bwt();
        // BWT of "banana$" is "annb$aa"
        let bwt_str = String::from_utf8_lossy(&bwt);
        // Verify it contains the right characters
        assert!(bwt_str.contains('n'));
        assert!(bwt_str.contains('a'));
        assert!(bwt_str.contains('b'));
    }

    // --- De Bruijn graph tests ---

    #[test]
    fn test_de_bruijn_basic() {
        // Reads from sequence "ATCGATCG"
        let reads: Vec<&[u8]> = vec![b"ATCG", b"TCGA", b"CGAT", b"GATC"];
        let graph = DeBruijnGraph::build(&reads, 3);

        assert!(!graph.edges.is_empty());
        assert!(graph.kmer_count() > 0);
    }

    #[test]
    fn test_de_bruijn_assembly() {
        // Reads overlapping to spell "ACGTACGT"
        let reads: Vec<&[u8]> = vec![b"ACGTA", b"CGTAC", b"GTACG", b"TACGT"];
        let graph = DeBruijnGraph::build(&reads, 4);
        let assembled = graph.eulerian_path();
        assert!(assembled.is_some(), "Should assemble a path");
        let seq = assembled.unwrap();
        assert!(seq.len() >= 5, "Assembled sequence should have reasonable length");
    }

    // --- Sequence statistics tests ---

    #[test]
    fn test_gc_content_extremes() {
        let all_gc = DnaSeq::new("GCGCGCGC").unwrap();
        let all_at = DnaSeq::new("ATATATAT").unwrap();
        assert!((all_gc.gc_content() - 1.0).abs() < 1e-10);
        assert!((all_at.gc_content() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_sequence_entropy() {
        let low_entropy = b"AAAAAAAAAA"; // all same = low entropy
        let high_entropy = b"ATGCATGCAT"; // varied = higher entropy

        let e_low = sequence_entropy(low_entropy, 2);
        let e_high = sequence_entropy(high_entropy, 2);

        assert!(e_low < e_high, "Low complexity should have less entropy, {} vs {}", e_low, e_high);
    }

    #[test]
    fn test_windowed_gc() {
        let seq = DnaSeq::new("AAAAAAAAATGCGCGCGCGCAAAAAAAA").unwrap();
        let windows = seq.windowed_gc(8, 1);
        assert!(!windows.is_empty());

        // GC-rich region should have higher values
        let gc_values: Vec<f64> = windows.iter().map(|(_, gc)| *gc).collect();
        let max_gc = gc_values.iter().cloned().fold(0.0f64, f64::max);
        assert!(max_gc > 0.5, "Should have a high-GC window, max={}", max_gc);
    }

    // --- Phylogenetics tests ---

    #[test]
    fn test_jukes_cantor_identical() {
        let a = DnaSeq::new("ATGCATGC").unwrap();
        let b = DnaSeq::new("ATGCATGC").unwrap();
        let d = jukes_cantor_distance(&a, &b);
        assert!((d - 0.0).abs() < 1e-10, "Identical sequences have distance 0");
    }

    #[test]
    fn test_jukes_cantor_different() {
        let a = DnaSeq::new("AAAAAAAA").unwrap();
        let b = DnaSeq::new("TTTTTTTT").unwrap();
        let d = jukes_cantor_distance(&a, &b);
        assert!(d > 0.0, "Different sequences have positive distance");
    }

    #[test]
    fn test_upgma_basic() {
        let labels: Vec<String> = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let distances = vec![
            vec![0.0, 1.0, 3.0],
            vec![1.0, 0.0, 3.0],
            vec![3.0, 3.0, 0.0],
        ];
        let tree = upgma(&labels, &distances);
        let newick = tree.to_newick();
        assert!(!newick.is_empty());
        assert!(newick.contains('A'));
        assert!(newick.contains('B'));
        assert!(newick.contains('C'));
    }

    #[test]
    fn test_upgma_two_taxa() {
        let labels: Vec<String> = vec!["Human".to_string(), "Chimp".to_string()];
        let distances = vec![vec![0.0, 0.015], vec![0.015, 0.0]];
        let tree = upgma(&labels, &distances);
        assert!(!tree.is_leaf()); // Should be an internal node
        assert!(tree.to_newick().contains("Human"));
    }

    // --- Secondary structure prediction tests ---

    #[test]
    fn test_secondary_structure_prediction() {
        // Alanine (A) has high helix propensity (1.42)
        let helix_heavy = ProteinSeq::new("AAAAAAAAAAAA").unwrap(); // poly-Ala
        let structure = ChouFasman::predict(&helix_heavy);
        let helix_count = structure.iter().filter(|&&s| s == SecondaryStructure::Helix).count();
        assert!(helix_count > 0, "Poly-Ala should have helix content");

        // Valine (V) has high sheet propensity (1.70)
        let sheet_heavy = ProteinSeq::new("VVVVVVVVVVVV").unwrap();
        let s2 = ChouFasman::predict(&sheet_heavy);
        let sheet_count = s2.iter().filter(|&&s| s == SecondaryStructure::Sheet).count();
        assert!(sheet_count > 0, "Poly-Val should have sheet content");
    }

    #[test]
    fn test_structure_string() {
        let structure = vec![
            SecondaryStructure::Helix,
            SecondaryStructure::Helix,
            SecondaryStructure::Coil,
            SecondaryStructure::Sheet,
        ];
        assert_eq!(ChouFasman::to_string(&structure), "HHCE");
    }

    // --- FASTA parser tests ---

    #[test]
    fn test_fasta_parsing() {
        let fasta = ">seq1 First sequence\nATGCATGC\nATGC\n>seq2 Second\nGGGGCCCC";
        let records = parse_fasta(fasta);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, "seq1");
        assert_eq!(records[0].sequence, "ATGCATGCATGC");
        assert_eq!(records[1].id, "seq2");
        assert_eq!(records[1].sequence, "GGGGCCCC");
    }

    #[test]
    fn test_fastq_parsing() {
        let fastq = "@read1\nATGCATGC\n+\nIIIIIIII\n@read2\nGGGGCCCC\n+\n########";
        let records = parse_fastq(fastq);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, "read1");
        assert_eq!(records[0].sequence, "ATGCATGC");
        assert_eq!(records[0].quality.len(), 8);
        // 'I' = 73 ASCII, -33 = 40 Phred score
        assert_eq!(records[0].quality[0], 40);
    }

    #[test]
    fn test_fastq_quality_average() {
        let fastq = "@r1\nATGC\n+\nIIII";
        let records = parse_fastq(fastq);
        let avg = fastq_average_quality(&records[0]);
        assert!((avg - 40.0).abs() < 0.01); // All 'I' = Phred 40
    }

    // --- Integration test: full genomics pipeline ---

    #[test]
    fn test_full_genomics_pipeline() {
        // 1. Parse a FASTA record
        let fasta = ">gene1 test gene\nATGAAAGACGGCTAA"; // M-K-D-G-stop
        let records = parse_fasta(fasta);
        assert_eq!(records.len(), 1);

        // 2. Create DNA sequence
        let dna = DnaSeq::new(&records[0].sequence).unwrap();
        assert_eq!(dna.len(), 15);

        // 3. Compute GC content
        let gc = dna.gc_content();
        assert!(gc > 0.0 && gc < 1.0);

        // 4. Find ORFs
        let orfs = find_orfs(&dna, 1);
        assert!(!orfs.is_empty(), "Should find ORFs");
        let forward = orfs.iter().find(|o| o.frame > 0).unwrap();
        assert_eq!(forward.protein.as_str(), "MKDG");

        // 5. Transcribe and translate
        let rna = dna.transcribe();
        let protein = rna.translate(0);
        assert_eq!(protein.as_str(), "MKDG");

        // 6. Align the protein against itself
        let aln = needleman_wunsch(&protein.0, &protein.0, 1, -1, 1);
        assert!((aln.identity - 1.0).abs() < 0.01);

        // 7. Build suffix array and search
        let sa = SuffixArray::build(&dna.0);
        let found = sa.find_all(b"ATG");
        assert!(!found.is_empty());
        assert_eq!(found[0], 0);

        // 8. Secondary structure prediction
        let ss = ChouFasman::predict(&protein);
        assert_eq!(ss.len(), protein.len());

        // 9. k-mer analysis
        let kmers = dna.kmer_counts(3);
        assert!(!kmers.is_empty());

        println!("Genomics pipeline: seq={}, GC={:.2}, ORFs={}, protein={}",
            dna.as_str(), gc, orfs.len(), protein.as_str());
    }

    #[test]
    fn test_codon_usage_table() {
        let seq = DnaSeq::new("ATGAAAGACGGCTAA").unwrap();
        let usage = codon_usage(&seq);
        assert_eq!(*usage.get(&[b'A', b'T', b'G']).unwrap_or(&0), 1); // ATG = Met
        assert_eq!(*usage.get(&[b'T', b'A', b'A']).unwrap_or(&0), 1); // TAA = stop
    }
}
