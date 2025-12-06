//! 텍스트 유사도 계산 모듈
//!
//! 다양한 유사도 알고리즘을 제공합니다.

use std::collections::{HashMap, HashSet};
use wasm_bindgen::prelude::*;

/// N-gram 유사도 계산
#[wasm_bindgen]
pub fn ngram_similarity(text1: &str, text2: &str, n: usize) -> f64 {
    let ngrams1 = extract_ngrams(text1, n);
    let ngrams2 = extract_ngrams(text2, n);

    if ngrams1.is_empty() && ngrams2.is_empty() {
        return 1.0;
    }
    if ngrams1.is_empty() || ngrams2.is_empty() {
        return 0.0;
    }

    let intersection = ngrams1.intersection(&ngrams2).count();
    let union = ngrams1.union(&ngrams2).count();

    intersection as f64 / union as f64
}

/// N-gram 추출
fn extract_ngrams(text: &str, n: usize) -> HashSet<String> {
    let chars: Vec<char> = text.to_lowercase().chars().collect();
    let mut ngrams = HashSet::new();

    if chars.len() >= n {
        for i in 0..=(chars.len() - n) {
            let ngram: String = chars[i..i + n].iter().collect();
            ngrams.insert(ngram);
        }
    }

    ngrams
}

/// Levenshtein 편집 거리
#[wasm_bindgen]
pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    let m = s1_chars.len();
    let n = s2_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[m][n]
}

/// Levenshtein 유사도 (정규화된 거리)
#[wasm_bindgen]
pub fn levenshtein_similarity(s1: &str, s2: &str) -> f64 {
    let max_len = s1.len().max(s2.len());
    if max_len == 0 {
        return 1.0;
    }

    let distance = levenshtein_distance(s1, s2);
    1.0 - (distance as f64 / max_len as f64)
}

/// TF-IDF 벡터 생성
pub struct TfIdfVectorizer {
    vocabulary: HashMap<String, usize>,
    idf: Vec<f64>,
}

impl TfIdfVectorizer {
    pub fn new() -> Self {
        Self {
            vocabulary: HashMap::new(),
            idf: Vec::new(),
        }
    }

    /// 문서 컬렉션에서 학습
    pub fn fit(&mut self, documents: &[&str]) {
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let n_docs = documents.len();

        // 어휘 구축 및 문서 빈도 계산
        for doc in documents {
            let unique_words: HashSet<String> = doc
                .to_lowercase()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();

            for word in unique_words {
                *doc_freq.entry(word).or_insert(0) += 1;
            }
        }

        // 어휘 인덱스 생성
        for (idx, word) in doc_freq.keys().enumerate() {
            self.vocabulary.insert(word.clone(), idx);
        }

        // IDF 계산
        self.idf = vec![0.0; self.vocabulary.len()];
        for (word, &df) in &doc_freq {
            if let Some(&idx) = self.vocabulary.get(word) {
                self.idf[idx] = (n_docs as f64 / (df as f64 + 1.0)).ln() + 1.0;
            }
        }
    }

    /// TF-IDF 벡터로 변환
    pub fn transform(&self, document: &str) -> Vec<f64> {
        let words: Vec<String> = document
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let n_words = words.len() as f64;
        let mut vector = vec![0.0; self.vocabulary.len()];

        // TF 계산
        let mut tf: HashMap<String, usize> = HashMap::new();
        for word in &words {
            *tf.entry(word.clone()).or_insert(0) += 1;
        }

        // TF-IDF 계산
        for (word, &count) in &tf {
            if let Some(&idx) = self.vocabulary.get(word) {
                vector[idx] = (count as f64 / n_words) * self.idf[idx];
            }
        }

        vector
    }

    /// 두 문서 간 코사인 유사도
    pub fn cosine_similarity(&self, doc1: &str, doc2: &str) -> f64 {
        let v1 = self.transform(doc1);
        let v2 = self.transform(doc2);

        let dot: f64 = v1.iter().zip(&v2).map(|(a, b)| a * b).sum();
        let mag1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mag2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if mag1 == 0.0 || mag2 == 0.0 {
            return 0.0;
        }

        dot / (mag1 * mag2)
    }
}

/// BM25 점수 계산
pub struct BM25 {
    k1: f64,
    b: f64,
    doc_lengths: Vec<f64>,
    avg_doc_length: f64,
    doc_freq: HashMap<String, usize>,
    n_docs: usize,
}

impl BM25 {
    pub fn new(k1: f64, b: f64) -> Self {
        Self {
            k1,
            b,
            doc_lengths: Vec::new(),
            avg_doc_length: 0.0,
            doc_freq: HashMap::new(),
            n_docs: 0,
        }
    }

    /// 문서 컬렉션에서 학습
    pub fn fit(&mut self, documents: &[&str]) {
        self.n_docs = documents.len();
        self.doc_lengths = documents.iter().map(|d| d.split_whitespace().count() as f64).collect();
        self.avg_doc_length = self.doc_lengths.iter().sum::<f64>() / self.n_docs as f64;

        for doc in documents {
            let unique_words: HashSet<String> = doc
                .to_lowercase()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();

            for word in unique_words {
                *self.doc_freq.entry(word).or_insert(0) += 1;
            }
        }
    }

    /// 쿼리에 대한 문서 점수 계산
    pub fn score(&self, query: &str, document: &str, doc_idx: usize) -> f64 {
        let query_terms: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let doc_terms: Vec<String> = document
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let doc_len = self.doc_lengths.get(doc_idx).unwrap_or(&0.0);

        let mut score = 0.0;

        for term in &query_terms {
            let df = self.doc_freq.get(term).unwrap_or(&0);
            let tf = doc_terms.iter().filter(|t| *t == term).count() as f64;

            // IDF 계산
            let idf = ((self.n_docs as f64 - *df as f64 + 0.5) / (*df as f64 + 0.5) + 1.0).ln();

            // BM25 점수
            let tf_component = (tf * (self.k1 + 1.0))
                / (tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_length));

            score += idf * tf_component;
        }

        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_similarity() {
        let sim = ngram_similarity("hello", "hella", 2);
        assert!(sim > 0.5);
    }

    #[test]
    fn test_levenshtein() {
        let dist = levenshtein_distance("kitten", "sitting");
        assert_eq!(dist, 3);

        let sim = levenshtein_similarity("kitten", "sitting");
        assert!(sim > 0.5);
    }

    #[test]
    fn test_tfidf() {
        let mut vectorizer = TfIdfVectorizer::new();
        let docs = vec![
            "the quick brown fox",
            "the lazy dog sleeps",
            "the fox jumps over",
        ];
        vectorizer.fit(&docs);

        let sim = vectorizer.cosine_similarity("quick fox", "brown fox jumps");
        assert!(sim > 0.0);
    }
}
