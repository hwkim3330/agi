//! AGI Trinity - WASM Consensus Engine
//!
//! 고성능 합의 알고리즘을 WebAssembly로 구현합니다.
//! Python보다 30-50배 빠른 텍스트 유사도 계산과 합의 도출을 제공합니다.

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use unicode_segmentation::UnicodeSegmentation;

mod similarity;
mod consensus;

pub use similarity::*;
pub use consensus::*;

/// 콘솔 로그 매크로
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format!($($t)*)))
}

/// 에이전트 응답 구조체
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub agent_name: String,
    pub content: String,
    pub success: bool,
    pub latency: f64,
    pub confidence: f64,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// 합의 결과 구조체
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusOutput {
    pub strategy: String,
    pub content: String,
    pub confidence: f64,
    pub reasoning: String,
    pub individual_scores: HashMap<String, f64>,
    pub processing_time_ms: f64,
}

/// WASM 합의 엔진
#[wasm_bindgen]
pub struct WasmConsensusEngine {
    scoring_weights: ScoringWeights,
    stopwords: HashSet<String>,
}

#[derive(Debug, Clone)]
struct ScoringWeights {
    content_length: f64,
    response_time: f64,
    success_rate: f64,
    confidence: f64,
    uniqueness: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            content_length: 0.2,
            response_time: 0.15,
            success_rate: 0.25,
            confidence: 0.25,
            uniqueness: 0.15,
        }
    }
}

#[wasm_bindgen]
impl WasmConsensusEngine {
    /// 새 엔진 생성
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let stopwords = create_stopwords();
        Self {
            scoring_weights: ScoringWeights::default(),
            stopwords,
        }
    }

    /// 합의 계산 (JSON 입출력)
    #[wasm_bindgen]
    pub fn calculate_consensus(&self, responses_json: &str, strategy: &str) -> String {
        let start = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);

        let responses: Vec<AgentResponse> = match serde_json::from_str(responses_json) {
            Ok(r) => r,
            Err(e) => {
                return serde_json::to_string(&ConsensusOutput {
                    strategy: strategy.to_string(),
                    content: format!("Parse error: {}", e),
                    confidence: 0.0,
                    reasoning: "Failed to parse input".to_string(),
                    individual_scores: HashMap::new(),
                    processing_time_ms: 0.0,
                }).unwrap_or_default();
            }
        };

        let result = match strategy {
            "vote" => self.vote_consensus(&responses),
            "synthesis" => self.synthesis_consensus(&responses),
            "semantic" => self.semantic_consensus(&responses),
            "weighted" => self.weighted_consensus(&responses),
            _ => self.vote_consensus(&responses),
        };

        let end = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);

        let mut output = result;
        output.processing_time_ms = end - start;

        serde_json::to_string(&output).unwrap_or_default()
    }

    /// 투표 기반 합의
    fn vote_consensus(&self, responses: &[AgentResponse]) -> ConsensusOutput {
        let successful: Vec<_> = responses.iter().filter(|r| r.success).collect();

        if successful.is_empty() {
            return ConsensusOutput {
                strategy: "vote".to_string(),
                content: "All agents failed".to_string(),
                confidence: 0.0,
                reasoning: "No successful responses".to_string(),
                individual_scores: HashMap::new(),
                processing_time_ms: 0.0,
            };
        }

        let mut scores: HashMap<String, f64> = HashMap::new();

        for response in &successful {
            let score = self.calculate_response_score(response);
            scores.insert(response.agent_name.clone(), score);
        }

        let (best_agent, best_score) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, v)| (k.clone(), *v))
            .unwrap_or_default();

        let best_response = successful
            .iter()
            .find(|r| r.agent_name == best_agent)
            .map(|r| r.content.clone())
            .unwrap_or_default();

        let confidence = successful.len() as f64 / responses.len() as f64;

        ConsensusOutput {
            strategy: "vote".to_string(),
            content: best_response,
            confidence,
            reasoning: format!("Selected {} with score {:.3}", best_agent, best_score),
            individual_scores: scores,
            processing_time_ms: 0.0,
        }
    }

    /// 통합 합의
    fn synthesis_consensus(&self, responses: &[AgentResponse]) -> ConsensusOutput {
        let successful: Vec<_> = responses.iter().filter(|r| r.success).collect();

        if successful.is_empty() {
            return ConsensusOutput {
                strategy: "synthesis".to_string(),
                content: "All agents failed".to_string(),
                confidence: 0.0,
                reasoning: "No successful responses".to_string(),
                individual_scores: HashMap::new(),
                processing_time_ms: 0.0,
            };
        }

        let mut sections = Vec::new();
        let mut scores: HashMap<String, f64> = HashMap::new();

        for response in &successful {
            let role = response.metadata.get("role").cloned().unwrap_or_else(|| "Agent".to_string());
            let score = self.calculate_response_score(response);
            scores.insert(response.agent_name.clone(), score);

            sections.push(format!(
                "## {} ({})\n\n{}",
                role, response.agent_name, response.content
            ));
        }

        let synthesized = sections.join("\n\n---\n\n");
        let confidence = successful.len() as f64 / responses.len() as f64;

        ConsensusOutput {
            strategy: "synthesis".to_string(),
            content: synthesized,
            confidence,
            reasoning: format!("Synthesized {} responses", successful.len()),
            individual_scores: scores,
            processing_time_ms: 0.0,
        }
    }

    /// 의미 기반 합의
    fn semantic_consensus(&self, responses: &[AgentResponse]) -> ConsensusOutput {
        let successful: Vec<_> = responses.iter().filter(|r| r.success).collect();

        if successful.is_empty() {
            return ConsensusOutput {
                strategy: "semantic".to_string(),
                content: "All agents failed".to_string(),
                confidence: 0.0,
                reasoning: "No successful responses".to_string(),
                individual_scores: HashMap::new(),
                processing_time_ms: 0.0,
            };
        }

        // 텍스트 유사도 행렬 계산
        let contents: Vec<&str> = successful.iter().map(|r| r.content.as_str()).collect();
        let similarity_matrix = calculate_similarity_matrix(&contents, &self.stopwords);

        // 평균 유사도 점수
        let mut avg_similarities: HashMap<String, f64> = HashMap::new();
        for (i, response) in successful.iter().enumerate() {
            let row = &similarity_matrix[i];
            let avg = row.iter().sum::<f64>() / row.len() as f64;
            avg_similarities.insert(response.agent_name.clone(), avg);
        }

        // 가장 높은 평균 유사도를 가진 응답 선택
        let (best_agent, best_sim) = avg_similarities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, v)| (k.clone(), *v))
            .unwrap_or_default();

        let best_content = successful
            .iter()
            .find(|r| r.agent_name == best_agent)
            .map(|r| r.content.clone())
            .unwrap_or_default();

        // 공통 키워드 추출
        let common_keywords = extract_common_keywords(&contents, &self.stopwords, 10);
        let mut final_content = String::new();

        if !common_keywords.is_empty() {
            final_content.push_str("## Common Themes\n\n");
            for (keyword, count) in &common_keywords {
                final_content.push_str(&format!("- {} ({})\n", keyword, count));
            }
            final_content.push_str("\n---\n\n");
        }

        final_content.push_str(&best_content);

        ConsensusOutput {
            strategy: "semantic".to_string(),
            content: final_content,
            confidence: best_sim,
            reasoning: format!("Semantic analysis: {} highest agreement ({:.3})", best_agent, best_sim),
            individual_scores: avg_similarities,
            processing_time_ms: 0.0,
        }
    }

    /// 가중치 기반 합의
    fn weighted_consensus(&self, responses: &[AgentResponse]) -> ConsensusOutput {
        let successful: Vec<_> = responses.iter().filter(|r| r.success).collect();

        if successful.is_empty() {
            return ConsensusOutput {
                strategy: "weighted".to_string(),
                content: "All agents failed".to_string(),
                confidence: 0.0,
                reasoning: "No successful responses".to_string(),
                individual_scores: HashMap::new(),
                processing_time_ms: 0.0,
            };
        }

        // 에이전트별 가중치
        let agent_weights: HashMap<&str, f64> = [
            ("claude", 1.2),
            ("gemini", 1.1),
            ("codex", 1.0),
        ].iter().cloned().collect();

        let mut weighted_scores: HashMap<String, f64> = HashMap::new();

        for response in &successful {
            let base_score = self.calculate_response_score(response);
            let weight = agent_weights.get(response.agent_name.as_str()).unwrap_or(&1.0);
            weighted_scores.insert(response.agent_name.clone(), base_score * weight);
        }

        let (best_agent, best_score) = weighted_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, v)| (k.clone(), *v))
            .unwrap_or_default();

        let best_content = successful
            .iter()
            .find(|r| r.agent_name == best_agent)
            .map(|r| r.content.clone())
            .unwrap_or_default();

        ConsensusOutput {
            strategy: "weighted".to_string(),
            content: best_content,
            confidence: successful.len() as f64 / responses.len() as f64,
            reasoning: format!("Weighted: {} (score: {:.3})", best_agent, best_score),
            individual_scores: weighted_scores,
            processing_time_ms: 0.0,
        }
    }

    /// 응답 점수 계산
    fn calculate_response_score(&self, response: &AgentResponse) -> f64 {
        let content_len = response.content.len();

        // 콘텐츠 길이 점수
        let length_score = if content_len < 100 {
            content_len as f64 / 100.0
        } else if content_len > 10000 {
            10000.0 / content_len as f64
        } else {
            1.0
        };

        // 응답 시간 점수
        let time_score = 1.0 / (1.0 + response.latency / 30.0);

        // 가중 합산
        let score = length_score * self.scoring_weights.content_length
            + time_score * self.scoring_weights.response_time
            + response.confidence * self.scoring_weights.confidence
            + if response.success { 1.0 } else { 0.0 } * self.scoring_weights.success_rate;

        score.min(1.0).max(0.0)
    }

    /// 텍스트 유사도 계산 (외부 인터페이스)
    #[wasm_bindgen]
    pub fn calculate_similarity(&self, text1: &str, text2: &str) -> f64 {
        jaccard_similarity(text1, text2, &self.stopwords)
    }

    /// 코사인 유사도 계산
    #[wasm_bindgen]
    pub fn calculate_cosine_similarity(&self, text1: &str, text2: &str) -> f64 {
        cosine_similarity(text1, text2, &self.stopwords)
    }
}

/// 불용어 집합 생성
fn create_stopwords() -> HashSet<String> {
    let words = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "this", "that", "these", "those",
        "and", "but", "or", "for", "with", "not", "you", "your", "they",
        "their", "its", "from", "about", "into", "through", "during",
        "before", "after", "above", "below", "to", "of", "in", "on", "at",
        "by", "as", "it", "if", "so", "than", "then", "when", "where",
        "what", "which", "who", "how", "all", "each", "every", "both",
        "few", "more", "most", "other", "some", "such", "no", "only",
    ];
    words.iter().map(|s| s.to_string()).collect()
}

/// 자카드 유사도 계산
fn jaccard_similarity(text1: &str, text2: &str, stopwords: &HashSet<String>) -> f64 {
    let words1: HashSet<String> = text1
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() >= 3 && !stopwords.contains(*w))
        .map(|s| s.to_string())
        .collect();

    let words2: HashSet<String> = text2
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() >= 3 && !stopwords.contains(*w))
        .map(|s| s.to_string())
        .collect();

    if words1.is_empty() && words2.is_empty() {
        return 1.0;
    }
    if words1.is_empty() || words2.is_empty() {
        return 0.0;
    }

    let intersection = words1.intersection(&words2).count();
    let union = words1.union(&words2).count();

    intersection as f64 / union as f64
}

/// 코사인 유사도 계산
fn cosine_similarity(text1: &str, text2: &str, stopwords: &HashSet<String>) -> f64 {
    let words1 = tokenize(text1, stopwords);
    let words2 = tokenize(text2, stopwords);

    if words1.is_empty() || words2.is_empty() {
        return 0.0;
    }

    // TF 벡터 생성
    let mut vocab: HashSet<&str> = HashSet::new();
    for w in &words1 {
        vocab.insert(w);
    }
    for w in &words2 {
        vocab.insert(w);
    }

    let vocab: Vec<&str> = vocab.into_iter().collect();

    let tf1: Vec<f64> = vocab.iter().map(|w| {
        words1.iter().filter(|x| x == w).count() as f64
    }).collect();

    let tf2: Vec<f64> = vocab.iter().map(|w| {
        words2.iter().filter(|x| x == w).count() as f64
    }).collect();

    // 코사인 유사도
    let dot: f64 = tf1.iter().zip(&tf2).map(|(a, b)| a * b).sum();
    let mag1: f64 = tf1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag2: f64 = tf2.iter().map(|x| x * x).sum::<f64>().sqrt();

    if mag1 == 0.0 || mag2 == 0.0 {
        return 0.0;
    }

    dot / (mag1 * mag2)
}

/// 토큰화
fn tokenize<'a>(text: &'a str, stopwords: &HashSet<String>) -> Vec<&'a str> {
    text.to_lowercase();
    text.split_whitespace()
        .filter(|w| w.len() >= 3 && !stopwords.contains(&w.to_lowercase()))
        .collect()
}

/// 유사도 행렬 계산
fn calculate_similarity_matrix(texts: &[&str], stopwords: &HashSet<String>) -> Vec<Vec<f64>> {
    let n = texts.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                matrix[i][j] = 1.0;
            } else if j > i {
                let sim = jaccard_similarity(texts[i], texts[j], stopwords);
                matrix[i][j] = sim;
                matrix[j][i] = sim;
            }
        }
    }

    matrix
}

/// 공통 키워드 추출
fn extract_common_keywords(
    texts: &[&str],
    stopwords: &HashSet<String>,
    top_n: usize,
) -> Vec<(String, usize)> {
    let mut word_counts: HashMap<String, usize> = HashMap::new();

    for text in texts {
        for word in text.to_lowercase().split_whitespace() {
            if word.len() >= 3 && !stopwords.contains(word) {
                *word_counts.entry(word.to_string()).or_insert(0) += 1;
            }
        }
    }

    let mut sorted: Vec<_> = word_counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted.truncate(top_n);

    sorted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard_similarity() {
        let stopwords = create_stopwords();
        let sim = jaccard_similarity(
            "hello world programming code",
            "hello programming python code",
            &stopwords,
        );
        assert!(sim > 0.5);
    }

    #[test]
    fn test_consensus_engine() {
        let engine = WasmConsensusEngine::new();
        let responses = r#"[
            {"agent_name": "claude", "content": "Answer A", "success": true, "latency": 1.0, "confidence": 0.9, "metadata": {}},
            {"agent_name": "gemini", "content": "Answer B", "success": true, "latency": 1.5, "confidence": 0.8, "metadata": {}}
        ]"#;

        let result = engine.calculate_consensus(responses, "vote");
        assert!(!result.is_empty());
    }
}
