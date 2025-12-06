//! 합의 알고리즘 모듈
//!
//! 다양한 합의 메커니즘을 구현합니다.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// 합의 알고리즘 타입
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsensusType {
    Majority,
    Weighted,
    Borda,
    Approval,
}

/// 투표 결과
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteResult {
    pub winner: String,
    pub winner_score: f64,
    pub all_scores: HashMap<String, f64>,
    pub consensus_strength: f64,
}

/// 다수결 투표
#[wasm_bindgen]
pub fn majority_vote(votes_json: &str) -> String {
    let votes: Vec<String> = match serde_json::from_str(votes_json) {
        Ok(v) => v,
        Err(_) => return "{}".to_string(),
    };

    let mut counts: HashMap<String, usize> = HashMap::new();
    for vote in &votes {
        *counts.entry(vote.clone()).or_insert(0) += 1;
    }

    let total = votes.len() as f64;
    let (winner, count) = counts
        .iter()
        .max_by_key(|(_, &c)| c)
        .map(|(k, &v)| (k.clone(), v))
        .unwrap_or_default();

    let scores: HashMap<String, f64> = counts
        .iter()
        .map(|(k, &v)| (k.clone(), v as f64 / total))
        .collect();

    let result = VoteResult {
        winner,
        winner_score: count as f64 / total,
        all_scores: scores,
        consensus_strength: count as f64 / total,
    };

    serde_json::to_string(&result).unwrap_or_default()
}

/// 가중치 투표
#[wasm_bindgen]
pub fn weighted_vote(votes_json: &str, weights_json: &str) -> String {
    let votes: Vec<String> = match serde_json::from_str(votes_json) {
        Ok(v) => v,
        Err(_) => return "{}".to_string(),
    };

    let weights: Vec<f64> = match serde_json::from_str(weights_json) {
        Ok(w) => w,
        Err(_) => vec![1.0; votes.len()],
    };

    let mut weighted_counts: HashMap<String, f64> = HashMap::new();
    let mut total_weight = 0.0;

    for (vote, &weight) in votes.iter().zip(&weights) {
        *weighted_counts.entry(vote.clone()).or_insert(0.0) += weight;
        total_weight += weight;
    }

    let (winner, score) = weighted_counts
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(k, &v)| (k.clone(), v))
        .unwrap_or_default();

    let scores: HashMap<String, f64> = weighted_counts
        .iter()
        .map(|(k, &v)| (k.clone(), v / total_weight))
        .collect();

    let result = VoteResult {
        winner,
        winner_score: score / total_weight,
        all_scores: scores,
        consensus_strength: score / total_weight,
    };

    serde_json::to_string(&result).unwrap_or_default()
}

/// Borda Count 투표
/// 순위 기반 투표로, 각 순위에 점수 부여
#[wasm_bindgen]
pub fn borda_count(rankings_json: &str) -> String {
    // rankings는 각 투표자의 순위 리스트
    // 예: [["A", "B", "C"], ["B", "A", "C"]]
    let rankings: Vec<Vec<String>> = match serde_json::from_str(rankings_json) {
        Ok(r) => r,
        Err(_) => return "{}".to_string(),
    };

    if rankings.is_empty() {
        return "{}".to_string();
    }

    let n_candidates = rankings[0].len();
    let mut scores: HashMap<String, f64> = HashMap::new();

    for ranking in &rankings {
        for (rank, candidate) in ranking.iter().enumerate() {
            // 1위에게 n-1점, 2위에게 n-2점, ...
            let points = (n_candidates - rank - 1) as f64;
            *scores.entry(candidate.clone()).or_insert(0.0) += points;
        }
    }

    let max_score = (n_candidates - 1) as f64 * rankings.len() as f64;
    let (winner, score) = scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(k, &v)| (k.clone(), v))
        .unwrap_or_default();

    let normalized_scores: HashMap<String, f64> = scores
        .iter()
        .map(|(k, &v)| (k.clone(), v / max_score))
        .collect();

    let result = VoteResult {
        winner,
        winner_score: score / max_score,
        all_scores: normalized_scores,
        consensus_strength: score / max_score,
    };

    serde_json::to_string(&result).unwrap_or_default()
}

/// 승인 투표 (Approval Voting)
/// 각 투표자가 여러 후보를 승인할 수 있음
#[wasm_bindgen]
pub fn approval_vote(approvals_json: &str) -> String {
    // approvals는 각 투표자가 승인한 후보 리스트
    // 예: [["A", "B"], ["B", "C"], ["A", "C"]]
    let approvals: Vec<Vec<String>> = match serde_json::from_str(approvals_json) {
        Ok(a) => a,
        Err(_) => return "{}".to_string(),
    };

    let n_voters = approvals.len() as f64;
    let mut counts: HashMap<String, usize> = HashMap::new();

    for voter_approvals in &approvals {
        for candidate in voter_approvals {
            *counts.entry(candidate.clone()).or_insert(0) += 1;
        }
    }

    let (winner, count) = counts
        .iter()
        .max_by_key(|(_, &c)| c)
        .map(|(k, &v)| (k.clone(), v))
        .unwrap_or_default();

    let scores: HashMap<String, f64> = counts
        .iter()
        .map(|(k, &v)| (k.clone(), v as f64 / n_voters))
        .collect();

    let result = VoteResult {
        winner,
        winner_score: count as f64 / n_voters,
        all_scores: scores,
        consensus_strength: count as f64 / n_voters,
    };

    serde_json::to_string(&result).unwrap_or_default()
}

/// 합의 강도 계산
/// 응답들 간의 일치도를 측정
#[wasm_bindgen]
pub fn calculate_consensus_strength(responses_json: &str) -> f64 {
    let responses: Vec<String> = match serde_json::from_str(responses_json) {
        Ok(r) => r,
        Err(_) => return 0.0,
    };

    if responses.len() < 2 {
        return 1.0;
    }

    // 모든 쌍에 대해 유사도 계산
    let mut total_similarity = 0.0;
    let mut pair_count = 0;

    for i in 0..responses.len() {
        for j in (i + 1)..responses.len() {
            let sim = simple_jaccard(&responses[i], &responses[j]);
            total_similarity += sim;
            pair_count += 1;
        }
    }

    if pair_count == 0 {
        return 1.0;
    }

    total_similarity / pair_count as f64
}

/// 간단한 자카드 유사도 (내부용)
fn simple_jaccard(text1: &str, text2: &str) -> f64 {
    use std::collections::HashSet;

    let words1: HashSet<&str> = text1.split_whitespace().collect();
    let words2: HashSet<&str> = text2.split_whitespace().collect();

    if words1.is_empty() && words2.is_empty() {
        return 1.0;
    }

    let intersection = words1.intersection(&words2).count();
    let union = words1.union(&words2).count();

    intersection as f64 / union as f64
}

/// 앙상블 합의
/// 여러 합의 방법을 결합
#[wasm_bindgen]
pub fn ensemble_consensus(
    vote_result_json: &str,
    weighted_result_json: &str,
    semantic_scores_json: &str,
) -> String {
    let vote_result: VoteResult = serde_json::from_str(vote_result_json).unwrap_or(VoteResult {
        winner: String::new(),
        winner_score: 0.0,
        all_scores: HashMap::new(),
        consensus_strength: 0.0,
    });

    let weighted_result: VoteResult = serde_json::from_str(weighted_result_json).unwrap_or(VoteResult {
        winner: String::new(),
        winner_score: 0.0,
        all_scores: HashMap::new(),
        consensus_strength: 0.0,
    });

    let semantic_scores: HashMap<String, f64> =
        serde_json::from_str(semantic_scores_json).unwrap_or_default();

    // 앙상블 점수 계산 (가중 평균)
    let mut ensemble_scores: HashMap<String, f64> = HashMap::new();

    // 모든 후보 수집
    let mut all_candidates: std::collections::HashSet<String> = std::collections::HashSet::new();
    for k in vote_result.all_scores.keys() {
        all_candidates.insert(k.clone());
    }
    for k in weighted_result.all_scores.keys() {
        all_candidates.insert(k.clone());
    }
    for k in semantic_scores.keys() {
        all_candidates.insert(k.clone());
    }

    for candidate in all_candidates {
        let vote_score = vote_result.all_scores.get(&candidate).unwrap_or(&0.0);
        let weighted_score = weighted_result.all_scores.get(&candidate).unwrap_or(&0.0);
        let semantic_score = semantic_scores.get(&candidate).unwrap_or(&0.0);

        // 가중 평균: vote(0.3) + weighted(0.3) + semantic(0.4)
        let ensemble = vote_score * 0.3 + weighted_score * 0.3 + semantic_score * 0.4;
        ensemble_scores.insert(candidate, ensemble);
    }

    let (winner, score) = ensemble_scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(k, &v)| (k.clone(), v))
        .unwrap_or_default();

    let result = VoteResult {
        winner,
        winner_score: score,
        all_scores: ensemble_scores,
        consensus_strength: (vote_result.consensus_strength
            + weighted_result.consensus_strength
            + score)
            / 3.0,
    };

    serde_json::to_string(&result).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_majority_vote() {
        let votes = r#"["A", "B", "A", "A", "B"]"#;
        let result = majority_vote(votes);
        let parsed: VoteResult = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.winner, "A");
    }

    #[test]
    fn test_borda_count() {
        let rankings = r#"[["A", "B", "C"], ["B", "A", "C"], ["A", "C", "B"]]"#;
        let result = borda_count(rankings);
        let parsed: VoteResult = serde_json::from_str(&result).unwrap();
        assert!(!parsed.winner.is_empty());
    }

    #[test]
    fn test_consensus_strength() {
        let responses = r#"["hello world", "hello there", "hi world"]"#;
        let strength = calculate_consensus_strength(responses);
        assert!(strength > 0.0 && strength <= 1.0);
    }
}
