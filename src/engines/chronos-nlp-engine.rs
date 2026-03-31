// ============================================================================
// CHRONOS NATURAL LANGUAGE PROCESSING ENGINE
// ============================================================================
//
// HOW NLP ACTUALLY WORKS (from first principles):
//
// Natural Language Processing bridges the gap between human language and
// computation. Language is messy, ambiguous, context-dependent, and
// evolving — making it one of the hardest problems in computer science.
//
// The fundamental challenge: language is symbolic and compositional.
// "The cat sat on the mat" has meaning because of the arrangement of
// symbols (words) according to rules (grammar). But the same words in
// different order ("The mat sat on the cat") mean something different.
// And the same meaning can be expressed many ways ("A feline rested
// atop a rug").
//
// The key theoretical foundations:
//
// 1. TOKENIZATION: Breaking text into meaningful units. Seems trivial
//    ("split on spaces") but isn't: contractions (don't → do + n't),
//    hyphenated words, URLs, CJK characters (no spaces), subword
//    tokenization (BPE, WordPiece) for handling unknown words.
//
// 2. FORMAL LANGUAGE THEORY (Chomsky hierarchy):
//    - Regular languages: can be recognized by finite automata (regex)
//    - Context-free: recognized by pushdown automata (most programming
//      languages, approximate natural language syntax)
//    - Context-sensitive: closer to actual natural language
//    - The syntax of natural language is roughly context-free, but
//      semantics require much more
//
// 3. STATISTICAL NLP: Instead of hand-writing rules, learn patterns
//    from data. N-gram models capture local context. TF-IDF measures
//    word importance. Naive Bayes and logistic regression for classification.
//
// 4. DISTRIBUTIONAL SEMANTICS: "You shall know a word by the company
//    it keeps" (Firth, 1957). Words that appear in similar contexts
//    have similar meanings. Word vectors (Word2Vec, GloVe) encode this.
//
// 5. SEQUENCE MODELS: Language is sequential. Hidden Markov Models for
//    POS tagging. CRFs for named entity recognition. RNNs/LSTMs/
//    Transformers for modern deep learning approaches.
//
// WHAT THIS ENGINE IMPLEMENTS (real, working logic with tests):
//   1.  Tokenizer (whitespace, punctuation-aware, sentence splitting)
//   2.  N-gram extraction and language model (bigram, trigram)
//   3.  TF-IDF (Term Frequency — Inverse Document Frequency)
//   4.  Text normalization (lowercasing, stemming, stop word removal)
//   5.  Porter Stemmer (the classic English stemming algorithm)
//   6.  Edit distance (Levenshtein) for spelling correction
//   7.  Naive Bayes text classifier
//   8.  Sentiment analysis (lexicon-based)
//   9.  POS tagger (rule-based + suffix heuristics)
//  10.  Cosine similarity for document comparison
//  11.  BM25 ranking (the algorithm behind most search engines)
//  12.  Regular expression engine (NFA-based, Thompson's construction)
//  13.  Byte Pair Encoding (BPE) tokenizer
//  14.  Text summarization (extractive, using sentence scoring)
//  15.  Phonetic encoding (Soundex for fuzzy name matching)
//  16.  Comprehensive tests
// ============================================================================

use std::collections::{HashMap, HashSet, BTreeMap};

// ============================================================================
// Part 1: Tokenizer
// ============================================================================
//
// Tokenization is the first step in any NLP pipeline. We split text into
// tokens (words, punctuation, numbers) and also split into sentences.
//
// Key decisions:
// - Punctuation is a separate token (so "Hello!" becomes ["Hello", "!"])
// - Contractions are kept together ("don't" stays as one token)
// - Numbers are their own tokens
// - Sentence splitting uses ., !, ? followed by whitespace or end of text

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Word(String),
    Number(String),
    Punctuation(char),
    Whitespace,
}

pub fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        if c.is_whitespace() {
            // Skip whitespace, but record it if needed
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            if !tokens.is_empty() {
                tokens.push(Token::Whitespace);
            }
            continue;
        }

        if c.is_alphabetic() || c == '\'' {
            // Word token: letters + embedded apostrophes (don't, it's)
            let start = i;
            while i < chars.len()
                && (chars[i].is_alphabetic()
                    || chars[i] == '\''
                    || chars[i] == '-' && i + 1 < chars.len() && chars[i + 1].is_alphabetic())
            {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            tokens.push(Token::Word(word));
            continue;
        }

        if c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            let num: String = chars[start..i].iter().collect();
            tokens.push(Token::Number(num));
            continue;
        }

        // Punctuation or other character
        tokens.push(Token::Punctuation(c));
        i += 1;
    }

    // Remove trailing whitespace token
    if matches!(tokens.last(), Some(Token::Whitespace)) {
        tokens.pop();
    }

    tokens
}

/// Extract just the words from tokens, lowercased
pub fn extract_words(tokens: &[Token]) -> Vec<String> {
    tokens
        .iter()
        .filter_map(|t| match t {
            Token::Word(w) => Some(w.to_lowercase()),
            _ => None,
        })
        .collect()
}

/// Split text into sentences using basic heuristics
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    // Common abbreviations that don't end sentences
    let abbreviations: HashSet<&str> = ["mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "inc", "ltd"]
        .iter()
        .copied()
        .collect();

    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        current.push(chars[i]);

        if chars[i] == '.' || chars[i] == '!' || chars[i] == '?' {
            // Check if this is really a sentence boundary
            let is_end = if i + 1 >= chars.len() {
                true // End of text
            } else if chars[i + 1].is_whitespace()
                && (i + 2 >= chars.len() || chars[i + 2].is_uppercase())
            {
                // Check if the word before the period is an abbreviation
                let trimmed = current.trim_end_matches(|c: char| c == '.' || c == '!' || c == '?');
                let last_word = trimmed.split_whitespace().last().unwrap_or("").to_lowercase();
                !abbreviations.contains(last_word.as_str())
            } else if chars[i + 1].is_whitespace() {
                true
            } else {
                false
            };

            if is_end {
                let sentence = current.trim().to_string();
                if !sentence.is_empty() {
                    sentences.push(sentence);
                }
                current = String::new();
                // Skip whitespace after sentence
                i += 1;
                while i < chars.len() && chars[i].is_whitespace() {
                    i += 1;
                }
                continue;
            }
        }

        i += 1;
    }

    let remaining = current.trim().to_string();
    if !remaining.is_empty() {
        sentences.push(remaining);
    }

    sentences
}

// ============================================================================
// Part 2: Text Normalization and Stop Words
// ============================================================================
//
// Raw text contains variation that doesn't affect meaning: case differences,
// common words that appear everywhere ("the", "is", "at"), and different
// word forms ("running" vs "run"). Normalization reduces this noise.

/// Common English stop words that carry little semantic content
pub fn stop_words() -> HashSet<String> {
    let words = [
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "am", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "shall", "should", "may",
        "might", "must", "can", "could", "i", "me", "my", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "it", "its",
        "they", "them", "their", "this", "that", "these", "those",
        "what", "which", "who", "whom", "not", "no", "so", "if",
        "then", "than", "too", "very", "just", "about", "up", "out",
        "into", "over", "after", "before", "between", "under", "again",
        "there", "here", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such",
        "only", "own", "same",
    ];
    words.iter().map(|w| w.to_string()).collect()
}

/// Remove stop words from a word list
pub fn remove_stop_words(words: &[String]) -> Vec<String> {
    let stops = stop_words();
    words
        .iter()
        .filter(|w| !stops.contains(w.as_str()))
        .cloned()
        .collect()
}

// ============================================================================
// Part 3: Porter Stemmer
// ============================================================================
//
// The Porter Stemmer (1980) is the most widely used English stemming
// algorithm. It strips suffixes in a series of steps to reduce words
// to their stems: "running" → "run", "happily" → "happili" (not
// perfect, but consistent and fast).
//
// The algorithm works by measuring the "consonant-vowel" pattern of a
// word. A "measure" m is the number of VC (vowel-consonant) sequences.
// Rules are applied conditionally based on m.
//
// We implement a simplified but functional version of the algorithm.

pub fn porter_stem(word: &str) -> String {
    if word.len() <= 2 {
        return word.to_string();
    }

    let mut w = word.to_lowercase();

    // Step 1a: plurals
    if w.ends_with("sses") {
        w = w[..w.len() - 2].to_string();
    } else if w.ends_with("ies") {
        w = w[..w.len() - 2].to_string();
    } else if !w.ends_with("ss") && w.ends_with('s') {
        w = w[..w.len() - 1].to_string();
    }

    // Step 1b: -eed, -ed, -ing
    if w.ends_with("eed") {
        if measure(&w[..w.len() - 3]) > 0 {
            w = w[..w.len() - 1].to_string();
        }
    } else if w.ends_with("ed") {
        let stem = &w[..w.len() - 2];
        if contains_vowel(stem) {
            w = stem.to_string();
            w = step1b_fixup(w);
        }
    } else if w.ends_with("ing") {
        let stem = &w[..w.len() - 3];
        if contains_vowel(stem) {
            w = stem.to_string();
            w = step1b_fixup(w);
        }
    }

    // Step 1c: y → i
    if w.ends_with('y') && contains_vowel(&w[..w.len() - 1]) {
        let len = w.len();
        w.replace_range(len - 1..len, "i");
    }

    // Step 2: map double suffixes
    let step2_rules: Vec<(&str, &str)> = vec![
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("abli", "able"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
    ];

    for (suffix, replacement) in &step2_rules {
        if w.ends_with(suffix) {
            let stem = &w[..w.len() - suffix.len()];
            if measure(stem) > 0 {
                w = format!("{}{}", stem, replacement);
            }
            break;
        }
    }

    // Step 3: more suffix mapping
    let step3_rules: Vec<(&str, &str)> = vec![
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    ];

    for (suffix, replacement) in &step3_rules {
        if w.ends_with(suffix) {
            let stem = &w[..w.len() - suffix.len()];
            if measure(stem) > 0 {
                w = format!("{}{}", stem, replacement);
            }
            break;
        }
    }

    // Step 4: remove suffixes
    let step4_suffixes = [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
        "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
    ];

    for suffix in &step4_suffixes {
        if w.ends_with(suffix) {
            let stem = &w[..w.len() - suffix.len()];
            if measure(stem) > 1 {
                if *suffix == "ion" {
                    // Special case: must end in s or t
                    if stem.ends_with('s') || stem.ends_with('t') {
                        w = stem.to_string();
                    }
                } else {
                    w = stem.to_string();
                }
            }
            break;
        }
    }

    // Step 5: clean up
    if w.ends_with('e') {
        let stem = &w[..w.len() - 1];
        if measure(stem) > 1 || (measure(stem) == 1 && !ends_cvc(stem)) {
            w = stem.to_string();
        }
    }

    if w.ends_with("ll") && measure(&w) > 1 {
        w = w[..w.len() - 1].to_string();
    }

    w
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}

fn is_vowel_in_context(word: &str, i: usize) -> bool {
    let c = word.as_bytes()[i] as char;
    if is_vowel(c) {
        return true;
    }
    if c == 'y' && i > 0 {
        return !is_vowel(word.as_bytes()[i - 1] as char);
    }
    false
}

/// Measure: count the number of VC sequences in the word
fn measure(word: &str) -> usize {
    if word.is_empty() {
        return 0;
    }
    let mut m = 0;
    let mut i = 0;
    let len = word.len();

    // Skip initial consonants
    while i < len && !is_vowel_in_context(word, i) {
        i += 1;
    }

    loop {
        if i >= len {
            break;
        }
        // Skip vowels
        while i < len && is_vowel_in_context(word, i) {
            i += 1;
        }
        if i >= len {
            break;
        }
        // Skip consonants
        while i < len && !is_vowel_in_context(word, i) {
            i += 1;
        }
        m += 1;
    }

    m
}

fn contains_vowel(word: &str) -> bool {
    for i in 0..word.len() {
        if is_vowel_in_context(word, i) {
            return true;
        }
    }
    false
}

fn ends_cvc(word: &str) -> bool {
    let len = word.len();
    if len < 3 {
        return false;
    }
    let c1 = !is_vowel_in_context(word, len - 3);
    let v = is_vowel_in_context(word, len - 2);
    let c2 = !is_vowel_in_context(word, len - 1);
    let last = word.as_bytes()[len - 1] as char;
    c1 && v && c2 && !matches!(last, 'w' | 'x' | 'y')
}

fn step1b_fixup(mut w: String) -> String {
    if w.ends_with("at") || w.ends_with("bl") || w.ends_with("iz") {
        w.push('e');
    } else if w.len() >= 2 {
        let bytes = w.as_bytes();
        let last = bytes[w.len() - 1];
        let prev = bytes[w.len() - 2];
        if last == prev
            && !matches!(last as char, 'l' | 's' | 'z')
        {
            w.pop();
        }
    }
    w
}

// ============================================================================
// Part 4: N-gram Language Model
// ============================================================================
//
// An N-gram model predicts the next word based on the previous N-1 words.
// P(w_n | w_1, ..., w_{n-1}) ≈ P(w_n | w_{n-N+1}, ..., w_{n-1})
//
// The model is built by counting N-gram occurrences in a corpus, then
// normalizing to get probabilities. Smoothing (add-k / Laplace) prevents
// zero probabilities for unseen N-grams.

pub struct NgramModel {
    pub n: usize,
    pub counts: HashMap<Vec<String>, HashMap<String, usize>>,
    pub vocab: HashSet<String>,
    pub smoothing: f64, // Laplace smoothing parameter (add-k)
}

impl NgramModel {
    pub fn new(n: usize, smoothing: f64) -> Self {
        Self {
            n,
            counts: HashMap::new(),
            vocab: HashSet::new(),
            smoothing,
        }
    }

    /// Train the model on a sequence of words
    pub fn train(&mut self, words: &[String]) {
        for w in words {
            self.vocab.insert(w.clone());
        }

        // Add start tokens
        let mut padded: Vec<String> = vec!["<START>".to_string(); self.n - 1];
        padded.extend(words.iter().cloned());
        padded.push("<END>".to_string());
        self.vocab.insert("<START>".to_string());
        self.vocab.insert("<END>".to_string());

        for window in padded.windows(self.n) {
            let context: Vec<String> = window[..self.n - 1].to_vec();
            let word = window[self.n - 1].clone();
            *self
                .counts
                .entry(context)
                .or_insert_with(HashMap::new)
                .entry(word)
                .or_insert(0) += 1;
        }
    }

    /// Get probability of a word given its context
    pub fn probability(&self, context: &[String], word: &str) -> f64 {
        let context_vec = context.to_vec();
        let total: usize = self
            .counts
            .get(&context_vec)
            .map(|m| m.values().sum())
            .unwrap_or(0);
        let count: usize = self
            .counts
            .get(&context_vec)
            .and_then(|m| m.get(word))
            .copied()
            .unwrap_or(0);

        let v = self.vocab.len() as f64;
        (count as f64 + self.smoothing) / (total as f64 + self.smoothing * v)
    }

    /// Compute perplexity on a test sequence (lower = better model fit)
    pub fn perplexity(&self, words: &[String]) -> f64 {
        let mut padded: Vec<String> = vec!["<START>".to_string(); self.n - 1];
        padded.extend(words.iter().cloned());
        padded.push("<END>".to_string());

        let mut log_prob = 0.0;
        let mut count = 0;

        for window in padded.windows(self.n) {
            let context = &window[..self.n - 1];
            let word = &window[self.n - 1];
            let p = self.probability(context, word);
            if p > 0.0 {
                log_prob += p.ln();
                count += 1;
            }
        }

        if count == 0 {
            return f64::INFINITY;
        }
        (-log_prob / count as f64).exp()
    }

    /// Generate text by sampling from the model
    pub fn generate(&self, max_words: usize, seed: &mut u64) -> Vec<String> {
        let mut result = Vec::new();
        let mut context: Vec<String> = vec!["<START>".to_string(); self.n - 1];

        for _ in 0..max_words {
            let next = self.sample_next(&context, seed);
            if next == "<END>" {
                break;
            }
            result.push(next.clone());
            context.push(next);
            if context.len() > self.n - 1 {
                context.remove(0);
            }
        }
        result
    }

    fn sample_next(&self, context: &[String], seed: &mut u64) -> String {
        let context_vec = context.to_vec();
        let distribution = match self.counts.get(&context_vec) {
            Some(d) => d,
            None => return "<END>".to_string(),
        };

        let total: usize = distribution.values().sum();
        if total == 0 {
            return "<END>".to_string();
        }

        // Simple LCG random
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let r = (*seed >> 33) as usize % total;

        let mut cumulative = 0;
        for (word, &count) in distribution {
            cumulative += count;
            if cumulative > r {
                return word.clone();
            }
        }
        "<END>".to_string()
    }
}

/// Extract n-grams from a word sequence
pub fn extract_ngrams(words: &[String], n: usize) -> Vec<Vec<String>> {
    if words.len() < n {
        return vec![];
    }
    words.windows(n).map(|w| w.to_vec()).collect()
}

// ============================================================================
// Part 5: TF-IDF
// ============================================================================
//
// TF-IDF (Term Frequency — Inverse Document Frequency) measures how
// important a word is to a document in a collection.
//
// TF(t, d) = count(t in d) / |d|   (how often does the word appear?)
// IDF(t) = log(N / df(t))          (how rare is the word across documents?)
// TF-IDF(t, d) = TF * IDF          (frequent in doc + rare in corpus = important)
//
// This is the foundation of information retrieval and text mining.

pub struct TfIdf {
    pub documents: Vec<Vec<String>>,
    pub idf: HashMap<String, f64>,
    pub doc_count: usize,
}

impl TfIdf {
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
            idf: HashMap::new(),
            doc_count: 0,
        }
    }

    /// Add documents and compute IDF
    pub fn fit(&mut self, documents: Vec<Vec<String>>) {
        self.doc_count = documents.len();
        let mut df: HashMap<String, usize> = HashMap::new();

        for doc in &documents {
            let unique: HashSet<&String> = doc.iter().collect();
            for word in unique {
                *df.entry(word.clone()).or_insert(0) += 1;
            }
        }

        for (term, count) in &df {
            self.idf.insert(
                term.clone(),
                ((self.doc_count as f64 + 1.0) / (*count as f64 + 1.0)).ln() + 1.0,
            );
        }

        self.documents = documents;
    }

    /// Compute TF for a term in a document
    pub fn tf(term: &str, document: &[String]) -> f64 {
        if document.is_empty() {
            return 0.0;
        }
        let count = document.iter().filter(|w| w.as_str() == term).count();
        count as f64 / document.len() as f64
    }

    /// Get TF-IDF score for a term in a document
    pub fn tfidf(&self, term: &str, document: &[String]) -> f64 {
        let tf = Self::tf(term, document);
        let idf = self.idf.get(term).copied().unwrap_or(0.0);
        tf * idf
    }

    /// Get TF-IDF vector for a document
    pub fn transform(&self, document: &[String]) -> HashMap<String, f64> {
        let mut vector = HashMap::new();
        let unique: HashSet<&String> = document.iter().collect();
        for word in unique {
            let score = self.tfidf(word, document);
            if score > 0.0 {
                vector.insert(word.clone(), score);
            }
        }
        vector
    }
}

// ============================================================================
// Part 6: Edit Distance (Levenshtein)
// ============================================================================
//
// The Levenshtein distance between two strings is the minimum number of
// single-character edits (insertions, deletions, substitutions) needed
// to transform one into the other.
//
// Computed via dynamic programming:
//   dp[i][j] = min(
//     dp[i-1][j] + 1,       // deletion
//     dp[i][j-1] + 1,       // insertion
//     dp[i-1][j-1] + cost   // substitution (cost=0 if same, 1 if different)
//   )
//
// Used for spelling correction, fuzzy matching, DNA sequence alignment.

pub fn edit_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[m][n]
}

/// Find the closest word in a dictionary using edit distance
pub fn spell_correct<'a>(word: &str, dictionary: &'a [String]) -> Option<&'a String> {
    dictionary
        .iter()
        .min_by_key(|candidate| edit_distance(word, candidate))
}

// ============================================================================
// Part 7: Naive Bayes Text Classifier
// ============================================================================
//
// Naive Bayes applies Bayes' theorem with the "naive" assumption that
// features (words) are conditionally independent given the class:
//
//   P(class | document) ∝ P(class) × Π P(word | class)
//
// Despite the obviously wrong independence assumption, it works
// surprisingly well for text classification (spam detection, sentiment,
// topic classification). The reasons: 1) we only need the argmax, not
// exact probabilities, and 2) even incorrect probability estimates
// often preserve the correct ranking.

pub struct NaiveBayesClassifier {
    pub class_word_counts: HashMap<String, HashMap<String, usize>>,
    pub class_doc_counts: HashMap<String, usize>,
    pub class_total_words: HashMap<String, usize>,
    pub vocab: HashSet<String>,
    pub total_docs: usize,
}

impl NaiveBayesClassifier {
    pub fn new() -> Self {
        Self {
            class_word_counts: HashMap::new(),
            class_doc_counts: HashMap::new(),
            class_total_words: HashMap::new(),
            vocab: HashSet::new(),
            total_docs: 0,
        }
    }

    /// Train on labeled documents: (words, class_label)
    pub fn train(&mut self, data: &[(Vec<String>, String)]) {
        for (words, class) in data {
            *self.class_doc_counts.entry(class.clone()).or_insert(0) += 1;
            self.total_docs += 1;

            let word_counts = self
                .class_word_counts
                .entry(class.clone())
                .or_insert_with(HashMap::new);
            let total = self.class_total_words.entry(class.clone()).or_insert(0);

            for word in words {
                self.vocab.insert(word.clone());
                *word_counts.entry(word.clone()).or_insert(0) += 1;
                *total += 1;
            }
        }
    }

    /// Predict the class of a document
    pub fn predict(&self, words: &[String]) -> String {
        let mut best_class = String::new();
        let mut best_score = f64::NEG_INFINITY;
        let v = self.vocab.len() as f64;

        for (class, &doc_count) in &self.class_doc_counts {
            // Log prior: log P(class)
            let log_prior = (doc_count as f64 / self.total_docs as f64).ln();
            let total_words = *self.class_total_words.get(class).unwrap_or(&0) as f64;
            let word_counts = self.class_word_counts.get(class);

            // Log likelihood: Σ log P(word | class) with Laplace smoothing
            let mut log_likelihood = 0.0;
            for word in words {
                let count = word_counts
                    .and_then(|wc| wc.get(word))
                    .copied()
                    .unwrap_or(0) as f64;
                log_likelihood += ((count + 1.0) / (total_words + v)).ln();
            }

            let score = log_prior + log_likelihood;
            if score > best_score {
                best_score = score;
                best_class = class.clone();
            }
        }

        best_class
    }

    /// Get class probabilities (log-space, then normalized)
    pub fn predict_proba(&self, words: &[String]) -> HashMap<String, f64> {
        let mut scores: HashMap<String, f64> = HashMap::new();
        let v = self.vocab.len() as f64;

        for (class, &doc_count) in &self.class_doc_counts {
            let log_prior = (doc_count as f64 / self.total_docs as f64).ln();
            let total_words = *self.class_total_words.get(class).unwrap_or(&0) as f64;
            let word_counts = self.class_word_counts.get(class);

            let mut log_likelihood = 0.0;
            for word in words {
                let count = word_counts
                    .and_then(|wc| wc.get(word))
                    .copied()
                    .unwrap_or(0) as f64;
                log_likelihood += ((count + 1.0) / (total_words + v)).ln();
            }

            scores.insert(class.clone(), log_prior + log_likelihood);
        }

        // Convert from log-space to probabilities using log-sum-exp
        let max_score = scores.values().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = scores.values().map(|s| (s - max_score).exp()).sum();
        let log_sum = max_score + sum.ln();

        scores
            .iter()
            .map(|(k, v)| (k.clone(), (v - log_sum).exp()))
            .collect()
    }
}

// ============================================================================
// Part 8: Sentiment Analysis (Lexicon-Based)
// ============================================================================
//
// Lexicon-based sentiment analysis assigns a score to each word and
// aggregates over the document. Simple but effective for many applications.
//
// We use a built-in sentiment lexicon with positive and negative words
// scored from -1.0 to +1.0, plus handling of negation and intensifiers.

pub struct SentimentAnalyzer {
    pub lexicon: HashMap<String, f64>,
    pub negation_words: HashSet<String>,
    pub intensifiers: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct SentimentResult {
    pub score: f64,      // -1.0 (very negative) to +1.0 (very positive)
    pub positive: f64,
    pub negative: f64,
    pub label: String,   // "positive", "negative", or "neutral"
}

impl SentimentAnalyzer {
    pub fn new() -> Self {
        let mut lexicon = HashMap::new();
        // Positive words
        for (word, score) in [
            ("good", 0.7), ("great", 0.8), ("excellent", 0.9), ("amazing", 0.9),
            ("wonderful", 0.9), ("fantastic", 0.9), ("love", 0.8), ("like", 0.5),
            ("enjoy", 0.6), ("happy", 0.8), ("glad", 0.6), ("pleased", 0.7),
            ("beautiful", 0.7), ("best", 0.9), ("perfect", 1.0), ("awesome", 0.8),
            ("nice", 0.5), ("fine", 0.3), ("brilliant", 0.8), ("superb", 0.9),
            ("outstanding", 0.9), ("delightful", 0.8), ("impressive", 0.7),
            ("remarkable", 0.7), ("pleasant", 0.6), ("recommend", 0.6),
        ] {
            lexicon.insert(word.to_string(), score);
        }
        // Negative words
        for (word, score) in [
            ("bad", -0.7), ("terrible", -0.9), ("horrible", -0.9), ("awful", -0.9),
            ("hate", -0.8), ("dislike", -0.6), ("poor", -0.7), ("worst", -1.0),
            ("ugly", -0.7), ("boring", -0.6), ("disappointing", -0.7),
            ("disappointed", -0.7), ("annoying", -0.6), ("angry", -0.7),
            ("sad", -0.6), ("unhappy", -0.7), ("miserable", -0.8), ("dreadful", -0.8),
            ("pathetic", -0.8), ("lousy", -0.7), ("mediocre", -0.4), ("weak", -0.5),
            ("wrong", -0.5), ("broken", -0.6), ("useless", -0.8), ("waste", -0.6),
        ] {
            lexicon.insert(word.to_string(), score);
        }

        let negation_words: HashSet<String> = ["not", "no", "never", "neither", "nobody",
            "nothing", "nowhere", "nor", "cannot", "can't", "don't", "doesn't",
            "didn't", "won't", "wouldn't", "shouldn't", "isn't", "aren't", "wasn't"]
            .iter().map(|s| s.to_string()).collect();

        let mut intensifiers = HashMap::new();
        for (word, mult) in [
            ("very", 1.5), ("extremely", 2.0), ("really", 1.3), ("absolutely", 2.0),
            ("totally", 1.5), ("incredibly", 1.8), ("highly", 1.3), ("quite", 1.2),
            ("somewhat", 0.7), ("slightly", 0.5), ("barely", 0.3), ("hardly", 0.3),
        ] {
            intensifiers.insert(word.to_string(), mult);
        }

        Self {
            lexicon,
            negation_words,
            intensifiers,
        }
    }

    pub fn analyze(&self, text: &str) -> SentimentResult {
        let tokens = tokenize(text);
        let words = extract_words(&tokens);

        let mut positive_sum = 0.0;
        let mut negative_sum = 0.0;
        let mut word_count = 0;

        let mut negate = false;
        let mut intensifier = 1.0;

        for word in &words {
            if self.negation_words.contains(word.as_str()) {
                negate = true;
                continue;
            }

            if let Some(&mult) = self.intensifiers.get(word.as_str()) {
                intensifier = mult;
                continue;
            }

            if let Some(&score) = self.lexicon.get(word.as_str()) {
                let mut adjusted = score * intensifier;
                if negate {
                    adjusted = -adjusted * 0.75; // Negation flips and slightly dampens
                }

                if adjusted > 0.0 {
                    positive_sum += adjusted;
                } else {
                    negative_sum += adjusted.abs();
                }
                word_count += 1;
            }

            negate = false;
            intensifier = 1.0;
        }

        let total = positive_sum + negative_sum;
        let score = if total > 0.0 {
            (positive_sum - negative_sum) / total
        } else {
            0.0
        };

        let label = if score > 0.1 {
            "positive"
        } else if score < -0.1 {
            "negative"
        } else {
            "neutral"
        }
        .to_string();

        SentimentResult {
            score,
            positive: positive_sum,
            negative: negative_sum,
            label,
        }
    }
}

// ============================================================================
// Part 9: POS Tagger (Rule-Based)
// ============================================================================
//
// Part-of-speech tagging assigns grammatical categories (noun, verb,
// adjective, etc.) to each word. Our rule-based tagger uses:
// 1. A lexicon of known word → POS mappings
// 2. Suffix heuristics for unknown words (-ing → VBG, -ly → RB, etc.)
// 3. Contextual rules (word after "the" is likely a noun)
//
// Uses Penn Treebank tagset (NN, VB, JJ, RB, DT, etc.)

pub struct PosTagger {
    pub lexicon: HashMap<String, String>,
}

impl PosTagger {
    pub fn new() -> Self {
        let mut lexicon = HashMap::new();

        // Determiners
        for w in ["the", "a", "an", "this", "that", "these", "those", "my",
                   "your", "his", "her", "its", "our", "their", "some", "any",
                   "no", "every", "each", "all", "both", "few", "many", "much"] {
            lexicon.insert(w.to_string(), "DT".to_string());
        }

        // Prepositions
        for w in ["in", "on", "at", "to", "for", "with", "by", "from", "of",
                   "about", "into", "through", "during", "before", "after",
                   "above", "below", "between", "under", "over"] {
            lexicon.insert(w.to_string(), "IN".to_string());
        }

        // Conjunctions
        for w in ["and", "or", "but", "nor", "yet", "so", "because", "although",
                   "while", "if", "when", "where", "that", "which", "who"] {
            lexicon.insert(w.to_string(), "CC".to_string());
        }

        // Pronouns
        for w in ["i", "me", "we", "us", "you", "he", "him", "she", "her",
                   "it", "they", "them", "myself", "yourself", "himself"] {
            lexicon.insert(w.to_string(), "PRP".to_string());
        }

        // Common verbs
        for w in ["is", "am", "are", "was", "were", "be", "been", "being",
                   "have", "has", "had", "do", "does", "did", "will", "would",
                   "shall", "should", "may", "might", "must", "can", "could",
                   "go", "get", "make", "know", "think", "take", "see", "come",
                   "want", "say", "give", "find", "tell"] {
            lexicon.insert(w.to_string(), "VB".to_string());
        }

        // Common adjectives
        for w in ["good", "great", "big", "small", "old", "new", "long",
                   "high", "little", "large", "young", "important", "different",
                   "right", "next", "last", "first", "early", "best", "worst"] {
            lexicon.insert(w.to_string(), "JJ".to_string());
        }

        // Common adverbs
        for w in ["not", "very", "often", "always", "never", "also", "just",
                   "still", "already", "even", "now", "then", "here", "there",
                   "quite", "really", "only", "well", "too"] {
            lexicon.insert(w.to_string(), "RB".to_string());
        }

        Self { lexicon }
    }

    /// Tag a sequence of words
    pub fn tag(&self, words: &[String]) -> Vec<(String, String)> {
        let mut tagged = Vec::new();

        for (i, word) in words.iter().enumerate() {
            let lower = word.to_lowercase();

            // First check lexicon
            let tag = if let Some(t) = self.lexicon.get(&lower) {
                t.clone()
            } else {
                // Apply suffix heuristics
                self.guess_tag(&lower, i, words)
            };

            tagged.push((word.clone(), tag));
        }

        // Apply contextual rules (second pass)
        self.apply_contextual_rules(&mut tagged);

        tagged
    }

    fn guess_tag(&self, word: &str, _pos: usize, _words: &[String]) -> String {
        if word.ends_with("ing") {
            "VBG".to_string() // present participle
        } else if word.ends_with("ed") {
            "VBD".to_string() // past tense
        } else if word.ends_with("ly") {
            "RB".to_string() // adverb
        } else if word.ends_with("tion") || word.ends_with("sion") || word.ends_with("ment")
            || word.ends_with("ness") || word.ends_with("ity") || word.ends_with("ance")
            || word.ends_with("ence")
        {
            "NN".to_string() // noun
        } else if word.ends_with("ful") || word.ends_with("ous") || word.ends_with("ive")
            || word.ends_with("able") || word.ends_with("ible") || word.ends_with("al")
            || word.ends_with("ical")
        {
            "JJ".to_string() // adjective
        } else if word.ends_with("er") {
            "NN".to_string() // comparative adj or agent noun
        } else if word.ends_with("est") {
            "JJS".to_string() // superlative
        } else if word.ends_with("s") && !word.ends_with("ss") {
            "NNS".to_string() // plural noun
        } else if word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
            "NNP".to_string() // proper noun
        } else {
            "NN".to_string() // default to noun
        }
    }

    fn apply_contextual_rules(&self, tagged: &mut Vec<(String, String)>) {
        for i in 0..tagged.len() {
            // After determiner, the next content word is likely a noun or adjective
            if i > 0 && tagged[i - 1].1 == "DT" {
                if tagged[i].1 == "VB" {
                    // "the run" — probably a noun
                    tagged[i].1 = "NN".to_string();
                }
            }
        }
    }
}

// ============================================================================
// Part 10: Cosine Similarity
// ============================================================================
//
// Cosine similarity measures the angle between two vectors in a
// high-dimensional space. For document comparison, each dimension
// represents a word, and the value is typically TF-IDF.
//
// cos(A, B) = (A · B) / (|A| × |B|)
//
// Range: 0 (orthogonal, no similarity) to 1 (identical direction).
// This is THE standard similarity measure for text because it's
// invariant to document length.

pub fn cosine_similarity(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (key, val) in a {
        norm_a += val * val;
        if let Some(bval) = b.get(key) {
            dot += val * bval;
        }
    }
    for val in b.values() {
        norm_b += val * val;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

/// Compute similarity between two documents using bag-of-words
pub fn document_similarity(doc_a: &[String], doc_b: &[String]) -> f64 {
    let vec_a = word_frequency_vector(doc_a);
    let vec_b = word_frequency_vector(doc_b);
    cosine_similarity(&vec_a, &vec_b)
}

fn word_frequency_vector(words: &[String]) -> HashMap<String, f64> {
    let mut freq = HashMap::new();
    for w in words {
        *freq.entry(w.clone()).or_insert(0.0) += 1.0;
    }
    freq
}

// ============================================================================
// Part 11: BM25 Ranking
// ============================================================================
//
// BM25 (Best Match 25) is the ranking function used by most search engines.
// It's a probabilistic model that scores documents by relevance to a query.
//
// BM25(D, Q) = Σ IDF(qi) × [f(qi,D) × (k1 + 1)] / [f(qi,D) + k1 × (1 - b + b × |D|/avgdl)]
//
// where:
// - f(qi, D) = frequency of query term qi in document D
// - |D| = document length
// - avgdl = average document length
// - k1, b = tuning parameters (typically k1=1.2, b=0.75)
//
// BM25 improves on TF-IDF by adding document length normalization and
// a saturation function for term frequency (diminishing returns).

pub struct Bm25 {
    pub documents: Vec<Vec<String>>,
    pub doc_lengths: Vec<usize>,
    pub avg_doc_length: f64,
    pub df: HashMap<String, usize>,
    pub k1: f64,
    pub b: f64,
}

impl Bm25 {
    pub fn new(documents: Vec<Vec<String>>, k1: f64, b: f64) -> Self {
        let doc_lengths: Vec<usize> = documents.iter().map(|d| d.len()).collect();
        let avg_doc_length = if doc_lengths.is_empty() {
            0.0
        } else {
            doc_lengths.iter().sum::<usize>() as f64 / doc_lengths.len() as f64
        };

        let mut df: HashMap<String, usize> = HashMap::new();
        for doc in &documents {
            let unique: HashSet<&String> = doc.iter().collect();
            for word in unique {
                *df.entry(word.clone()).or_insert(0) += 1;
            }
        }

        Self {
            documents,
            doc_lengths,
            avg_doc_length,
            df,
            k1,
            b,
        }
    }

    /// IDF component using the standard BM25 formula
    fn idf(&self, term: &str) -> f64 {
        let n = self.documents.len() as f64;
        let df = *self.df.get(term).unwrap_or(&0) as f64;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Score a single document against a query
    pub fn score(&self, doc_idx: usize, query: &[String]) -> f64 {
        let doc = &self.documents[doc_idx];
        let dl = self.doc_lengths[doc_idx] as f64;

        let mut tf_map: HashMap<&str, usize> = HashMap::new();
        for word in doc {
            *tf_map.entry(word.as_str()).or_insert(0) += 1;
        }

        let mut score = 0.0;
        for term in query {
            let idf = self.idf(term);
            let tf = *tf_map.get(term.as_str()).unwrap_or(&0) as f64;
            let numerator = tf * (self.k1 + 1.0);
            let denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avg_doc_length);
            score += idf * numerator / denominator;
        }

        score
    }

    /// Rank all documents by relevance to query
    pub fn rank(&self, query: &[String]) -> Vec<(usize, f64)> {
        let mut scores: Vec<(usize, f64)> = (0..self.documents.len())
            .map(|i| (i, self.score(i, query)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }
}

// ============================================================================
// Part 12: Regular Expression Engine (NFA-based)
// ============================================================================
//
// Thompson's construction converts a regex to an NFA (nondeterministic
// finite automaton), which can then match strings. This is the same
// approach used by grep and most regex engines.
//
// We support: literal chars, . (any), * (zero or more), + (one or more),
// ? (zero or one), | (alternation), and parentheses for grouping.
//
// The NFA is simulated by tracking all possible states simultaneously,
// which gives O(n*m) time complexity where n is the text length and m
// is the number of NFA states (much better than backtracking).

#[derive(Debug, Clone)]
enum NfaEdge {
    Char(char),
    Any,         // matches any character
    Epsilon,     // matches nothing (free transition)
}

#[derive(Debug, Clone)]
struct NfaState {
    edges: Vec<(NfaEdge, usize)>, // (edge_type, target_state)
    is_accept: bool,
}

pub struct Regex {
    states: Vec<NfaState>,
    start: usize,
}

impl Regex {
    /// Compile a regular expression pattern into an NFA
    pub fn compile(pattern: &str) -> Result<Self, String> {
        let tokens = Self::parse_pattern(pattern)?;
        let mut builder = NfaBuilder::new();
        let (start, accept) = builder.build_from_tokens(&tokens, 0)?;
        builder.states[accept].is_accept = true;
        Ok(Self {
            states: builder.states,
            start,
        })
    }

    /// Check if the pattern matches the entire string
    pub fn is_match(&self, text: &str) -> bool {
        let chars: Vec<char> = text.chars().collect();
        let mut current_states = HashSet::new();
        self.add_epsilon_closure(self.start, &mut current_states);

        for &c in &chars {
            let mut next_states = HashSet::new();
            for &state in &current_states {
                for (edge, target) in &self.states[state].edges {
                    match edge {
                        NfaEdge::Char(ec) if *ec == c => {
                            self.add_epsilon_closure(*target, &mut next_states);
                        }
                        NfaEdge::Any => {
                            self.add_epsilon_closure(*target, &mut next_states);
                        }
                        _ => {}
                    }
                }
            }
            current_states = next_states;
        }

        current_states.iter().any(|&s| self.states[s].is_accept)
    }

    /// Find the first match in the text, returning (start, end) indices
    pub fn find(&self, text: &str) -> Option<(usize, usize)> {
        let chars: Vec<char> = text.chars().collect();

        for start_pos in 0..=chars.len() {
            let mut current_states = HashSet::new();
            self.add_epsilon_closure(self.start, &mut current_states);

            if current_states.iter().any(|&s| self.states[s].is_accept) {
                return Some((start_pos, start_pos));
            }

            for end_pos in start_pos..chars.len() {
                let c = chars[end_pos];
                let mut next_states = HashSet::new();
                for &state in &current_states {
                    for (edge, target) in &self.states[state].edges {
                        match edge {
                            NfaEdge::Char(ec) if *ec == c => {
                                self.add_epsilon_closure(*target, &mut next_states);
                            }
                            NfaEdge::Any => {
                                self.add_epsilon_closure(*target, &mut next_states);
                            }
                            _ => {}
                        }
                    }
                }
                current_states = next_states;

                if current_states.iter().any(|&s| self.states[s].is_accept) {
                    return Some((start_pos, end_pos + 1));
                }

                if current_states.is_empty() {
                    break;
                }
            }
        }

        None
    }

    fn add_epsilon_closure(&self, state: usize, set: &mut HashSet<usize>) {
        if !set.insert(state) {
            return; // Already visited
        }
        for (edge, target) in &self.states[state].edges {
            if matches!(edge, NfaEdge::Epsilon) {
                self.add_epsilon_closure(*target, set);
            }
        }
    }

    fn parse_pattern(pattern: &str) -> Result<Vec<RegexToken>, String> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = pattern.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            match chars[i] {
                '.' => tokens.push(RegexToken::Any),
                '*' => tokens.push(RegexToken::Star),
                '+' => tokens.push(RegexToken::Plus),
                '?' => tokens.push(RegexToken::Question),
                '|' => tokens.push(RegexToken::Pipe),
                '(' => tokens.push(RegexToken::LParen),
                ')' => tokens.push(RegexToken::RParen),
                '\\' => {
                    i += 1;
                    if i >= chars.len() {
                        return Err("Trailing backslash".to_string());
                    }
                    tokens.push(RegexToken::Literal(chars[i]));
                }
                c => tokens.push(RegexToken::Literal(c)),
            }
            i += 1;
        }
        Ok(tokens)
    }
}

#[derive(Debug, Clone)]
enum RegexToken {
    Literal(char),
    Any,
    Star,
    Plus,
    Question,
    Pipe,
    LParen,
    RParen,
}

struct NfaBuilder {
    states: Vec<NfaState>,
}

impl NfaBuilder {
    fn new() -> Self {
        Self { states: Vec::new() }
    }

    fn new_state(&mut self) -> usize {
        let id = self.states.len();
        self.states.push(NfaState {
            edges: Vec::new(),
            is_accept: false,
        });
        id
    }

    fn add_edge(&mut self, from: usize, edge: NfaEdge, to: usize) {
        self.states[from].edges.push((edge, to));
    }

    /// Build NFA fragment from tokens, returns (start_state, accept_state)
    fn build_from_tokens(
        &mut self,
        tokens: &[RegexToken],
        pos: usize,
    ) -> Result<(usize, usize), String> {
        self.build_alternation(tokens, &mut { pos })
    }

    fn build_alternation(
        &mut self,
        tokens: &[RegexToken],
        pos: &mut usize,
    ) -> Result<(usize, usize), String> {
        let (mut start, mut accept) = self.build_concatenation(tokens, pos)?;

        while *pos < tokens.len() && matches!(tokens[*pos], RegexToken::Pipe) {
            *pos += 1; // skip |
            let (alt_start, alt_accept) = self.build_concatenation(tokens, pos)?;

            let new_start = self.new_state();
            let new_accept = self.new_state();
            self.add_edge(new_start, NfaEdge::Epsilon, start);
            self.add_edge(new_start, NfaEdge::Epsilon, alt_start);
            self.add_edge(accept, NfaEdge::Epsilon, new_accept);
            self.add_edge(alt_accept, NfaEdge::Epsilon, new_accept);

            start = new_start;
            accept = new_accept;
        }

        Ok((start, accept))
    }

    fn build_concatenation(
        &mut self,
        tokens: &[RegexToken],
        pos: &mut usize,
    ) -> Result<(usize, usize), String> {
        let start = self.new_state();
        let mut current_accept = start;

        while *pos < tokens.len() {
            match &tokens[*pos] {
                RegexToken::Pipe | RegexToken::RParen => break,
                _ => {
                    let (frag_start, frag_accept) = self.build_quantified(tokens, pos)?;
                    self.add_edge(current_accept, NfaEdge::Epsilon, frag_start);
                    current_accept = frag_accept;
                }
            }
        }

        Ok((start, current_accept))
    }

    fn build_quantified(
        &mut self,
        tokens: &[RegexToken],
        pos: &mut usize,
    ) -> Result<(usize, usize), String> {
        let (base_start, base_accept) = self.build_atom(tokens, pos)?;

        if *pos < tokens.len() {
            match tokens[*pos] {
                RegexToken::Star => {
                    *pos += 1;
                    let new_start = self.new_state();
                    let new_accept = self.new_state();
                    self.add_edge(new_start, NfaEdge::Epsilon, base_start);
                    self.add_edge(new_start, NfaEdge::Epsilon, new_accept);
                    self.add_edge(base_accept, NfaEdge::Epsilon, base_start);
                    self.add_edge(base_accept, NfaEdge::Epsilon, new_accept);
                    return Ok((new_start, new_accept));
                }
                RegexToken::Plus => {
                    *pos += 1;
                    let new_accept = self.new_state();
                    self.add_edge(base_accept, NfaEdge::Epsilon, base_start);
                    self.add_edge(base_accept, NfaEdge::Epsilon, new_accept);
                    return Ok((base_start, new_accept));
                }
                RegexToken::Question => {
                    *pos += 1;
                    let new_start = self.new_state();
                    let new_accept = self.new_state();
                    self.add_edge(new_start, NfaEdge::Epsilon, base_start);
                    self.add_edge(new_start, NfaEdge::Epsilon, new_accept);
                    self.add_edge(base_accept, NfaEdge::Epsilon, new_accept);
                    return Ok((new_start, new_accept));
                }
                _ => {}
            }
        }

        Ok((base_start, base_accept))
    }

    fn build_atom(
        &mut self,
        tokens: &[RegexToken],
        pos: &mut usize,
    ) -> Result<(usize, usize), String> {
        if *pos >= tokens.len() {
            let s = self.new_state();
            return Ok((s, s));
        }

        match &tokens[*pos] {
            RegexToken::Literal(c) => {
                let c = *c;
                *pos += 1;
                let start = self.new_state();
                let accept = self.new_state();
                self.add_edge(start, NfaEdge::Char(c), accept);
                Ok((start, accept))
            }
            RegexToken::Any => {
                *pos += 1;
                let start = self.new_state();
                let accept = self.new_state();
                self.add_edge(start, NfaEdge::Any, accept);
                Ok((start, accept))
            }
            RegexToken::LParen => {
                *pos += 1; // skip (
                let result = self.build_alternation(tokens, pos)?;
                if *pos < tokens.len() && matches!(tokens[*pos], RegexToken::RParen) {
                    *pos += 1; // skip )
                }
                Ok(result)
            }
            _ => {
                let s = self.new_state();
                Ok((s, s))
            }
        }
    }
}

// ============================================================================
// Part 13: Byte Pair Encoding (BPE)
// ============================================================================
//
// BPE is a subword tokenization algorithm used by GPT, RoBERTa, etc.
// It starts with character-level tokens and iteratively merges the most
// frequent pair of adjacent tokens. This creates a vocabulary that
// captures common subwords: "un" + "##happi" + "##ness".
//
// Advantages over word-level tokenization:
// - Handles unknown words (any word can be decomposed into subwords)
// - Compact vocabulary (30K tokens vs 100K+ words)
// - Captures morphological structure

pub struct BpeTokenizer {
    pub merges: Vec<(String, String)>,
    pub vocab: HashSet<String>,
}

impl BpeTokenizer {
    /// Train BPE on a corpus with a given number of merges
    pub fn train(corpus: &[String], num_merges: usize) -> Self {
        // Start with character-level tokens
        let mut word_splits: Vec<Vec<String>> = corpus
            .iter()
            .map(|w| w.chars().map(|c| c.to_string()).collect())
            .collect();

        let mut merges = Vec::new();
        let mut vocab: HashSet<String> = HashSet::new();
        for splits in &word_splits {
            for token in splits {
                vocab.insert(token.clone());
            }
        }

        for _ in 0..num_merges {
            // Count all adjacent pairs
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            for splits in &word_splits {
                for window in splits.windows(2) {
                    let pair = (window[0].clone(), window[1].clone());
                    *pair_counts.entry(pair).or_insert(0) += 1;
                }
            }

            // Find most frequent pair
            let best_pair = pair_counts
                .iter()
                .max_by_key(|(_, &count)| count);

            let (best_a, best_b) = match best_pair {
                Some(((a, b), count)) if *count >= 2 => (a.clone(), b.clone()),
                _ => break, // No more pairs to merge
            };

            // Merge that pair in all words
            let merged = format!("{}{}", best_a, best_b);
            vocab.insert(merged.clone());
            merges.push((best_a.clone(), best_b.clone()));

            for splits in &mut word_splits {
                let mut i = 0;
                while i + 1 < splits.len() {
                    if splits[i] == best_a && splits[i + 1] == best_b {
                        splits[i] = merged.clone();
                        splits.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        Self { merges, vocab }
    }

    /// Tokenize a word using learned merges
    pub fn tokenize(&self, word: &str) -> Vec<String> {
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        for (a, b) in &self.merges {
            let merged = format!("{}{}", a, b);
            let mut i = 0;
            while i + 1 < tokens.len() {
                if tokens[i] == *a && tokens[i + 1] == *b {
                    tokens[i] = merged.clone();
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        tokens
    }
}

// ============================================================================
// Part 14: Extractive Text Summarization
// ============================================================================
//
// Extractive summarization selects the most important sentences from the
// original text. We score sentences based on:
// 1. Word frequency (sentences with frequent important words score higher)
// 2. Position (first and last sentences are often more important)
// 3. Length (prefer sentences of medium length — not too short, not too long)
// 4. TF-IDF scores of contained words

pub fn extractive_summary(text: &str, num_sentences: usize) -> Vec<String> {
    let sentences = split_sentences(text);
    if sentences.len() <= num_sentences {
        return sentences;
    }

    let stops = stop_words();

    // Build word frequency map across all sentences
    let mut word_freq: HashMap<String, usize> = HashMap::new();
    for sentence in &sentences {
        let tokens = tokenize(sentence);
        let words = extract_words(&tokens);
        for word in words {
            if !stops.contains(&word) {
                *word_freq.entry(word).or_insert(0) += 1;
            }
        }
    }

    let max_freq = *word_freq.values().max().unwrap_or(&1) as f64;

    // Score each sentence
    let mut scored: Vec<(usize, f64)> = sentences
        .iter()
        .enumerate()
        .map(|(i, sentence)| {
            let tokens = tokenize(sentence);
            let words = extract_words(&tokens);
            let content_words: Vec<_> = words.iter().filter(|w| !stops.contains(*w)).collect();

            // Word importance score
            let word_score: f64 = content_words
                .iter()
                .map(|w| *word_freq.get(*w).unwrap_or(&0) as f64 / max_freq)
                .sum::<f64>()
                / (content_words.len().max(1) as f64);

            // Position score: first and last sentences are important
            let position_score = if i == 0 || i == sentences.len() - 1 {
                1.0
            } else if i < sentences.len() / 4 {
                0.7
            } else {
                0.3
            };

            // Length score: prefer medium-length sentences
            let len = words.len();
            let length_score = if len >= 5 && len <= 30 {
                1.0
            } else if len < 5 {
                0.3
            } else {
                0.5
            };

            let total_score = 0.5 * word_score + 0.3 * position_score + 0.2 * length_score;
            (i, total_score)
        })
        .collect();

    // Select top sentences by score
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut selected_indices: Vec<usize> = scored
        .iter()
        .take(num_sentences)
        .map(|(i, _)| *i)
        .collect();

    // Return in original order
    selected_indices.sort();
    selected_indices
        .iter()
        .map(|&i| sentences[i].clone())
        .collect()
}

// ============================================================================
// Part 15: Phonetic Encoding (Soundex)
// ============================================================================
//
// Soundex encodes words by their pronunciation, so similarly-sounding
// words get the same code. Originally designed for US Census records
// to match names despite spelling variations.
//
// Algorithm:
// 1. Keep the first letter
// 2. Map consonants to digits: B,F,P,V → 1; C,G,J,K,Q,S,X,Z → 2;
//    D,T → 3; L → 4; M,N → 5; R → 6
// 3. Remove vowels, H, W, Y
// 4. Remove consecutive duplicates
// 5. Pad/truncate to 4 characters

pub fn soundex(word: &str) -> String {
    if word.is_empty() {
        return String::new();
    }

    let chars: Vec<char> = word.to_uppercase().chars().filter(|c| c.is_alphabetic()).collect();
    if chars.is_empty() {
        return String::new();
    }

    let first = chars[0];

    let encode = |c: char| -> Option<char> {
        match c {
            'B' | 'F' | 'P' | 'V' => Some('1'),
            'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => Some('2'),
            'D' | 'T' => Some('3'),
            'L' => Some('4'),
            'M' | 'N' => Some('5'),
            'R' => Some('6'),
            _ => None, // A, E, I, O, U, H, W, Y
        }
    };

    let mut result = String::new();
    result.push(first);

    let mut prev_code = encode(first);

    for &c in &chars[1..] {
        let code = encode(c);
        if let Some(digit) = code {
            if code != prev_code {
                result.push(digit);
                if result.len() == 4 {
                    return result;
                }
            }
        }
        prev_code = code;
    }

    // Pad with zeros
    while result.len() < 4 {
        result.push('0');
    }

    result
}

// ============================================================================
// Part 16: Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Tokenizer tests ---

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello, world!");
        assert!(matches!(&tokens[0], Token::Word(w) if w == "Hello"));
        assert!(matches!(&tokens[1], Token::Punctuation(',')));
        assert!(matches!(&tokens[3], Token::Word(w) if w == "world"));
        assert!(matches!(&tokens[4], Token::Punctuation('!')));
    }

    #[test]
    fn test_tokenize_contractions() {
        let tokens = tokenize("don't can't I'm");
        let words = extract_words(&tokens);
        assert!(words.contains(&"don't".to_string()));
        assert!(words.contains(&"can't".to_string()));
        assert!(words.contains(&"i'm".to_string()));
    }

    #[test]
    fn test_tokenize_numbers() {
        let tokens = tokenize("There are 42 cats and 3.14 pies");
        let numbers: Vec<_> = tokens.iter().filter(|t| matches!(t, Token::Number(_))).collect();
        assert_eq!(numbers.len(), 2);
    }

    #[test]
    fn test_sentence_splitting() {
        let text = "Hello world. How are you? I am fine! This is great.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 4);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "How are you?");
    }

    // --- Stemmer tests ---

    #[test]
    fn test_porter_stemmer() {
        assert_eq!(porter_stem("running"), "run");
        assert_eq!(porter_stem("caresses"), "caress");
        assert_eq!(porter_stem("ponies"), "poni");
        assert_eq!(porter_stem("cats"), "cat");
        assert_eq!(porter_stem("agreed"), "agre");
        assert_eq!(porter_stem("plastered"), "plaster");
    }

    #[test]
    fn test_porter_stemmer_step2() {
        assert_eq!(porter_stem("relational"), "relat"); // ational → ate, then further
        assert_eq!(porter_stem("hopeful"), "hope");
    }

    // --- N-gram tests ---

    #[test]
    fn test_extract_ngrams() {
        let words: Vec<String> = vec!["the", "cat", "sat", "on", "the", "mat"]
            .into_iter()
            .map(String::from)
            .collect();
        let bigrams = extract_ngrams(&words, 2);
        assert_eq!(bigrams.len(), 5);
        assert_eq!(bigrams[0], vec!["the", "cat"]);
        assert_eq!(bigrams[4], vec!["the", "mat"]);
    }

    #[test]
    fn test_ngram_model() {
        let mut model = NgramModel::new(2, 0.1);
        let corpus: Vec<String> = vec!["the", "cat", "sat", "on", "the", "mat"]
            .into_iter()
            .map(String::from)
            .collect();
        model.train(&corpus);

        // P("cat" | "the") should be higher than a random word
        let p_cat = model.probability(&["the".to_string()], "cat");
        let p_random = model.probability(&["the".to_string()], "xyz");
        assert!(p_cat > p_random);
    }

    #[test]
    fn test_ngram_perplexity() {
        let mut model = NgramModel::new(2, 0.1);
        let corpus: Vec<String> = "the cat sat on the mat the cat sat on the rug"
            .split_whitespace()
            .map(String::from)
            .collect();
        model.train(&corpus);

        let test: Vec<String> = "the cat sat".split_whitespace().map(String::from).collect();
        let perplexity = model.perplexity(&test);
        assert!(perplexity > 0.0 && perplexity < 100.0, "Perplexity={}", perplexity);
    }

    // --- TF-IDF tests ---

    #[test]
    fn test_tfidf() {
        let docs = vec![
            vec!["the", "cat", "sat"],
            vec!["the", "dog", "ran"],
            vec!["a", "cat", "ran"],
        ]
        .into_iter()
        .map(|d| d.into_iter().map(String::from).collect())
        .collect();

        let mut tfidf = TfIdf::new();
        tfidf.fit(docs);

        // "the" appears in 2 docs, "cat" appears in 2 docs
        // "dog" appears in 1 doc — should have higher IDF
        let idf_dog = tfidf.idf.get("dog").copied().unwrap_or(0.0);
        let idf_the = tfidf.idf.get("the").copied().unwrap_or(0.0);
        assert!(idf_dog > idf_the, "Rare words should have higher IDF");
    }

    // --- Edit distance tests ---

    #[test]
    fn test_edit_distance() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", "abc"), 0);
        assert_eq!(edit_distance("a", "b"), 1);
    }

    #[test]
    fn test_spell_correct() {
        let dictionary: Vec<String> = vec!["hello", "world", "help", "held"]
            .into_iter()
            .map(String::from)
            .collect();
        let result = spell_correct("helo", &dictionary).unwrap();
        assert_eq!(result, "hello");
    }

    // --- Naive Bayes tests ---

    #[test]
    fn test_naive_bayes_classification() {
        let mut classifier = NaiveBayesClassifier::new();

        let training_data = vec![
            (vec!["amazing", "great", "love", "excellent"].into_iter().map(String::from).collect(), "positive".to_string()),
            (vec!["wonderful", "fantastic", "good", "enjoy"].into_iter().map(String::from).collect(), "positive".to_string()),
            (vec!["terrible", "awful", "hate", "horrible"].into_iter().map(String::from).collect(), "negative".to_string()),
            (vec!["bad", "worst", "boring", "disappointing"].into_iter().map(String::from).collect(), "negative".to_string()),
        ];

        classifier.train(&training_data);

        let positive_test: Vec<String> = vec!["great", "love", "amazing"].into_iter().map(String::from).collect();
        assert_eq!(classifier.predict(&positive_test), "positive");

        let negative_test: Vec<String> = vec!["terrible", "horrible", "bad"].into_iter().map(String::from).collect();
        assert_eq!(classifier.predict(&negative_test), "negative");
    }

    #[test]
    fn test_naive_bayes_probabilities() {
        let mut classifier = NaiveBayesClassifier::new();
        let data = vec![
            (vec!["good".to_string()], "pos".to_string()),
            (vec!["bad".to_string()], "neg".to_string()),
        ];
        classifier.train(&data);

        let proba = classifier.predict_proba(&["good".to_string()]);
        assert!(proba["pos"] > proba["neg"]);
    }

    // --- Sentiment tests ---

    #[test]
    fn test_sentiment_positive() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("This movie is great and amazing!");
        assert_eq!(result.label, "positive");
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_sentiment_negative() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("This is terrible and horrible");
        assert_eq!(result.label, "negative");
        assert!(result.score < 0.0);
    }

    #[test]
    fn test_sentiment_negation() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("This is not good");
        // "not" should flip "good" to negative
        assert!(result.negative > 0.0, "Negation should create negative sentiment");
    }

    #[test]
    fn test_sentiment_intensifier() {
        let analyzer = SentimentAnalyzer::new();
        let normal = analyzer.analyze("This is good");
        let intensified = analyzer.analyze("This is very good");
        assert!(
            intensified.positive > normal.positive,
            "Intensifier should increase magnitude"
        );
    }

    // --- POS tagger tests ---

    #[test]
    fn test_pos_tagging() {
        let tagger = PosTagger::new();
        let words: Vec<String> = vec!["the", "cat", "is", "running", "quickly"]
            .into_iter()
            .map(String::from)
            .collect();
        let tagged = tagger.tag(&words);

        assert_eq!(tagged[0].1, "DT");   // the → determiner
        assert_eq!(tagged[2].1, "VB");   // is → verb
        assert_eq!(tagged[3].1, "VBG");  // running → present participle
        assert_eq!(tagged[4].1, "RB");   // quickly → adverb
    }

    #[test]
    fn test_pos_suffix_heuristics() {
        let tagger = PosTagger::new();
        let words: Vec<String> = vec!["beautiful", "education", "slowly"]
            .into_iter()
            .map(String::from)
            .collect();
        let tagged = tagger.tag(&words);

        assert_eq!(tagged[0].1, "JJ");   // -ful → adjective
        assert_eq!(tagged[1].1, "NN");   // -tion → noun
        assert_eq!(tagged[2].1, "RB");   // -ly → adverb
    }

    // --- Cosine similarity tests ---

    #[test]
    fn test_cosine_similarity_identical() {
        let mut a = HashMap::new();
        a.insert("cat".to_string(), 1.0);
        a.insert("dog".to_string(), 2.0);
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6, "Identical vectors should have similarity 1.0");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let mut a = HashMap::new();
        a.insert("cat".to_string(), 1.0);
        let mut b = HashMap::new();
        b.insert("dog".to_string(), 1.0);
        let sim = cosine_similarity(&a, &b);
        assert!((sim).abs() < 1e-6, "Orthogonal vectors should have similarity 0.0");
    }

    #[test]
    fn test_document_similarity() {
        let doc1: Vec<String> = "the cat sat on the mat".split_whitespace().map(String::from).collect();
        let doc2: Vec<String> = "the cat lay on the rug".split_whitespace().map(String::from).collect();
        let doc3: Vec<String> = "quantum physics experiments".split_whitespace().map(String::from).collect();

        let sim_12 = document_similarity(&doc1, &doc2);
        let sim_13 = document_similarity(&doc1, &doc3);
        assert!(sim_12 > sim_13, "Similar docs should have higher similarity");
    }

    // --- BM25 tests ---

    #[test]
    fn test_bm25_ranking() {
        let docs = vec![
            "the cat sat on the mat".split_whitespace().map(String::from).collect(),
            "the dog ran in the park".split_whitespace().map(String::from).collect(),
            "cats and dogs are pets".split_whitespace().map(String::from).collect(),
        ];

        let bm25 = Bm25::new(docs, 1.2, 0.75);
        let query: Vec<String> = vec!["cat".to_string()];
        let ranked = bm25.rank(&query);

        // Document 0 should rank first (contains "cat")
        assert_eq!(ranked[0].0, 0);
    }

    #[test]
    fn test_bm25_multiple_terms() {
        let docs = vec![
            "machine learning algorithms".split_whitespace().map(String::from).collect(),
            "deep learning neural networks".split_whitespace().map(String::from).collect(),
            "cooking recipes for beginners".split_whitespace().map(String::from).collect(),
        ];

        let bm25 = Bm25::new(docs, 1.2, 0.75);
        let query: Vec<String> = vec!["learning".to_string(), "neural".to_string()];
        let ranked = bm25.rank(&query);

        // Document 1 should rank first (matches both terms)
        assert_eq!(ranked[0].0, 1);
    }

    // --- Regex engine tests ---

    #[test]
    fn test_regex_literal() {
        let re = Regex::compile("hello").unwrap();
        assert!(re.is_match("hello"));
        assert!(!re.is_match("helo"));
    }

    #[test]
    fn test_regex_dot() {
        let re = Regex::compile("h.llo").unwrap();
        assert!(re.is_match("hello"));
        assert!(re.is_match("hallo"));
        assert!(!re.is_match("hllo"));
    }

    #[test]
    fn test_regex_star() {
        let re = Regex::compile("ab*c").unwrap();
        assert!(re.is_match("ac"));
        assert!(re.is_match("abc"));
        assert!(re.is_match("abbbbc"));
        assert!(!re.is_match("adc"));
    }

    #[test]
    fn test_regex_plus() {
        let re = Regex::compile("ab+c").unwrap();
        assert!(!re.is_match("ac"));
        assert!(re.is_match("abc"));
        assert!(re.is_match("abbbc"));
    }

    #[test]
    fn test_regex_question() {
        let re = Regex::compile("colou?r").unwrap();
        assert!(re.is_match("color"));
        assert!(re.is_match("colour"));
        assert!(!re.is_match("colouur"));
    }

    #[test]
    fn test_regex_alternation() {
        let re = Regex::compile("cat|dog").unwrap();
        assert!(re.is_match("cat"));
        assert!(re.is_match("dog"));
        assert!(!re.is_match("cow"));
    }

    #[test]
    fn test_regex_find() {
        let re = Regex::compile("world").unwrap();
        let found = re.find("hello world!");
        assert_eq!(found, Some((6, 11)));
    }

    // --- BPE tests ---

    #[test]
    fn test_bpe_training() {
        let corpus: Vec<String> = vec![
            "low", "lower", "newest", "widest", "low", "low", "lower",
        ].into_iter().map(String::from).collect();

        let bpe = BpeTokenizer::train(&corpus, 10);
        assert!(!bpe.merges.is_empty(), "BPE should learn some merges");

        // Tokenize a word — should use learned subwords
        let tokens = bpe.tokenize("low");
        // "low" appears frequently, so its characters should be merged
        assert!(tokens.len() <= 3); // At most 3 chars, likely merged
    }

    #[test]
    fn test_bpe_unknown_word() {
        let corpus: Vec<String> = vec!["hello", "help", "held"]
            .into_iter().map(String::from).collect();
        let bpe = BpeTokenizer::train(&corpus, 5);

        // Should still tokenize unknown word (possibly into characters)
        let tokens = bpe.tokenize("hero");
        assert!(!tokens.is_empty());
    }

    // --- Summarization tests ---

    #[test]
    fn test_extractive_summary() {
        let text = "Machine learning is a subset of artificial intelligence. \
                     It involves training algorithms on data. \
                     The algorithms learn patterns from the data. \
                     This enables predictions on new unseen data. \
                     Machine learning has many applications in industry.";

        let summary = extractive_summary(text, 2);
        assert_eq!(summary.len(), 2);
        // Summary should contain actual sentences from the text
        for s in &summary {
            assert!(text.contains(s.trim_end_matches('.')));
        }
    }

    // --- Soundex tests ---

    #[test]
    fn test_soundex() {
        // Classic Soundex examples
        assert_eq!(soundex("Robert"), "R163");
        assert_eq!(soundex("Rupert"), "R163"); // Same code as Robert
        assert_eq!(soundex("Smith"), "S530");
        assert_eq!(soundex("Smythe"), "S530"); // Same code as Smith
    }

    #[test]
    fn test_soundex_similar_names() {
        // Names that sound similar should have the same code
        assert_eq!(soundex("Johnson"), soundex("Jonson"));
        assert_eq!(soundex("Meyer"), soundex("Meier"));
    }

    #[test]
    fn test_soundex_different_names() {
        assert_ne!(soundex("Smith"), soundex("Jones"));
        assert_ne!(soundex("Robert"), soundex("Smith"));
    }

    // --- Stop words test ---

    #[test]
    fn test_remove_stop_words() {
        let words: Vec<String> = vec!["the", "quick", "brown", "fox", "is", "very", "fast"]
            .into_iter().map(String::from).collect();
        let filtered = remove_stop_words(&words);
        assert!(filtered.contains(&"quick".to_string()));
        assert!(filtered.contains(&"brown".to_string()));
        assert!(filtered.contains(&"fox".to_string()));
        assert!(filtered.contains(&"fast".to_string()));
        assert!(!filtered.contains(&"the".to_string()));
        assert!(!filtered.contains(&"is".to_string()));
    }

    // --- Integration test ---

    #[test]
    fn test_full_nlp_pipeline() {
        // Full pipeline: tokenize → normalize → classify → summarize
        let text = "The movie was absolutely amazing and wonderful. \
                     The acting was superb and the plot was brilliant. \
                     I would highly recommend this excellent film to everyone. \
                     The special effects were also outstanding.";

        // 1. Tokenize and extract words
        let tokens = tokenize(text);
        let words = extract_words(&tokens);
        assert!(words.len() > 10);

        // 2. Remove stop words
        let content_words = remove_stop_words(&words);
        assert!(content_words.len() < words.len());

        // 3. Stem words
        let stemmed: Vec<String> = content_words.iter().map(|w| porter_stem(w)).collect();
        assert!(stemmed.len() == content_words.len());

        // 4. Sentiment analysis
        let analyzer = SentimentAnalyzer::new();
        let sentiment = analyzer.analyze(text);
        assert_eq!(sentiment.label, "positive");
        assert!(sentiment.score > 0.5);

        // 5. POS tagging
        let tagger = PosTagger::new();
        let first_sentence: Vec<String> = "The movie was absolutely amazing"
            .split_whitespace().map(String::from).collect();
        let tagged = tagger.tag(&first_sentence);
        assert_eq!(tagged[0].1, "DT"); // The

        // 6. Summarize
        let summary = extractive_summary(text, 2);
        assert_eq!(summary.len(), 2);
    }
}
