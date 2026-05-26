"""
evaluate_rag.py — Chuẩn đánh giá quốc tế cho Extractive QA (SQuAD-style)
=========================================================================
Tham chiếu:
  - Rajpurkar et al. 2016, "SQuAD: 100,000+ Questions for Machine Comprehension"
    https://arxiv.org/abs/1606.05250
  - Script chính thức: https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
  - ViQuAD (Vietnamese SQuAD) sử dụng chính xác pipeline đánh giá này
    https://vlsp.org.vn/vlsp2021/eval

Các chỉ số:
  ┌─────────────────────────────────────────────────────────────────┐
  │  EM  (Exact Match)  — Tiêu chuẩn cứng, khớp 100% sau normalize  │
  │  F1                 — Token overlap, thước đo mềm chính thức     │
  │  Precision          — Trong dự đoán, bao nhiêu token đúng        │
  │  Recall             — Trong ground truth, bao nhiêu token tìm ra │
  │  ROUGE-L            — Longest Common Subsequence, đánh giá thứ tự│
  └─────────────────────────────────────────────────────────────────┘

Lưu ý quan trọng:
  • EM trên ViQuAD khoảng 65–80% với best model (vi-mrc-large).
    EM 36% trên full corpus là hợp lý cho RAG system không fine-tune.
  • F1 luôn >= EM. Khoảng cách F1 - EM cho biết hệ thống "gần đúng" nhiều không.
  • ROUGE-L phạt việc trả lời quá dài hoặc sai thứ tự từ.
"""

import re
import string
from collections import Counter


# ===========================================================================
# CHUẨN HÓA VĂN BẢN — Theo chuẩn SQuAD official evaluation script
# ===========================================================================

def normalize_text(text: str) -> str:
    """
    Pipeline chuẩn hóa chính thức của SQuAD (Rajpurkar et al. 2016).
    Áp dụng nguyên vẹn cho ViQuAD.

    Bước 1: lowercase
    Bước 2: xóa dấu câu  (string.punctuation)
    Bước 3: xóa mạo từ tiếng Anh a/an/the  (không ảnh hưởng tiếng Việt nhưng giữ
             nguyên để tương thích với script gốc nếu corpus có từ tiếng Anh)
    Bước 4: chuẩn hóa khoảng trắng
    """
    # Bước 1
    text = text.lower()
    # Bước 2 — xóa dấu câu
    text = ''.join(ch for ch in text if ch not in set(string.punctuation))
    # Bước 3 — xóa mạo từ tiếng Anh (word-boundary để không cắt "and" → "nd")
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Bước 4 — chuẩn hóa whitespace
    text = ' '.join(text.split())
    return text


# ===========================================================================
# EXACT MATCH — SQuAD official
# ===========================================================================

def exact_match_score(prediction: str, ground_truths: list[str]) -> float:
    """
    EM = 1.0 nếu prediction (sau normalize) khớp CHÍNH XÁC với BẤT KỲ ground truth nào.
    Đây là chỉ số CỨNG nhất — không cho điểm thành phần.

    Nguồn: evaluate-v1.1.py  L43–L45
    """
    norm_pred = normalize_text(prediction)
    return 1.0 if any(norm_pred == normalize_text(gt) for gt in ground_truths) else 0.0


# ===========================================================================
# TOKEN F1 / PRECISION / RECALL — SQuAD official
# ===========================================================================

def _token_f1_single(prediction: str, ground_truth: str):
    """
    Token-level F1 giữa một prediction và MỘT ground truth.
    Dùng bag-of-words (Counter) — cùng logic với script SQuAD gốc.

    Trả về (f1, precision, recall).
    """
    pred_tokens = normalize_text(prediction).split()
    gt_tokens   = normalize_text(ground_truth).split()

    # Cả hai rỗng → khớp hoàn hảo
    if not pred_tokens and not gt_tokens:
        return 1.0, 1.0, 1.0
    # Một bên rỗng → không khớp
    if not pred_tokens or not gt_tokens:
        return 0.0, 0.0, 0.0

    common    = Counter(pred_tokens) & Counter(gt_tokens)
    num_same  = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gt_tokens)
    f1        = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def token_f1_score(prediction: str, ground_truths: list[str]):
    """
    Lấy điểm F1/Precision/Recall CAO NHẤT trong tất cả ground truth.
    *** Quan trọng ***: precision và recall phải lấy từ CÙNG MỘT ground truth
    (cùng cặp cho f1 cao nhất), không được lấy max riêng lẻ vì sẽ tạo ra
    precision và recall từ các ground truth khác nhau → F1 tính không nhất quán.

    Nguồn: evaluate-v1.1.py  L47–L61
    """
    best_f1, best_p, best_r = 0.0, 0.0, 0.0
    for gt in ground_truths:
        f1, p, r = _token_f1_single(prediction, gt)
        if f1 > best_f1:
            best_f1, best_p, best_r = f1, p, r
    return best_f1, best_p, best_r


# ===========================================================================
# ROUGE-L — Longest Common Subsequence
# ===========================================================================

def _lcs_length(x: list, y: list) -> int:
    """Độ dài LCS bằng dynamic programming O(|x|*|y|)."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Dùng 1D DP để tiết kiệm bộ nhớ
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l_score(prediction: str, ground_truths: list[str]):
    """
    ROUGE-L dựa trên LCS ở mức token.
    Công thức: Lin 2004 "ROUGE: A Package for Automatic Evaluation of Summaries"
      R_lcs = LCS / |reference|
      P_lcs = LCS / |hypothesis|
      F_lcs = (1 + beta²) * R_lcs * P_lcs / (R_lcs + beta² * P_lcs)   với beta=1 → F1
    Lấy điểm cao nhất trong các ground truth.
    """
    pred_tokens = normalize_text(prediction).split()
    best_f = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_text(gt).split()
        if not pred_tokens or not gt_tokens:
            continue
        lcs = _lcs_length(pred_tokens, gt_tokens)
        r = lcs / len(gt_tokens)
        p = lcs / len(pred_tokens)
        if r + p == 0:
            continue
        f = 2 * r * p / (r + p)
        if f > best_f:
            best_f = f
    return best_f


# ===========================================================================
# HÀM ĐÁNH GIÁ CHÍNH
# ===========================================================================

def evaluate_system(pred_file: str, ref_file: str) -> dict | None:
    """
    Đọc file dự đoán và file đáp án chuẩn, tính đầy đủ các chỉ số.

    Format ref_file: mỗi dòng là một hoặc nhiều đáp án đúng, cách nhau bằng ';'
    Ví dụ: "Paris;thành phố Paris"

    Trả về dict chứa tất cả chỉ số (tiện dùng trong code khác),
    đồng thời in bảng kết quả ra màn hình.
    """
    try:
        with open(pred_file, 'r', encoding='utf-8') as fp:
            predictions = [line.rstrip('\n') for line in fp]
        with open(ref_file, 'r', encoding='utf-8') as fr:
            references  = [line.rstrip('\n') for line in fr]
    except Exception as e:
        print(f"[ERROR] Không đọc được file: {e}")
        return None

    # Căn chỉnh độ dài
    if len(predictions) != len(references):
        print(f"[WARN] Số dòng không khớp: pred={len(predictions)}, ref={len(references)}")
        n = min(len(predictions), len(references))
        predictions, references = predictions[:n], references[:n]

    metrics = {
        'em': [], 'f1': [], 'precision': [], 'recall': [], 'rouge_l': []
    }
    skipped = 0

    for pred, ref_line in zip(predictions, references):
        ground_truths = [gt.strip() for gt in ref_line.split(';') if gt.strip()]
        if not ground_truths:
            skipped += 1
            continue

        metrics['em'].append(exact_match_score(pred, ground_truths))

        f1, prec, rec = token_f1_score(pred, ground_truths)
        metrics['f1'].append(f1)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)

        metrics['rouge_l'].append(rouge_l_score(pred, ground_truths))

    n_valid = len(metrics['em'])
    if n_valid == 0:
        print("[ERROR] Không có câu hợp lệ để chấm điểm.")
        return None

    def avg(lst): return sum(lst) / len(lst) * 100

    results = {
        'n_total'   : len(predictions),
        'n_valid'   : n_valid,
        'n_skipped' : skipped,
        'EM'        : avg(metrics['em']),
        'F1'        : avg(metrics['f1']),
        'Precision' : avg(metrics['precision']),
        'Recall'    : avg(metrics['recall']),
        'ROUGE-L'   : avg(metrics['rouge_l']),
    }

    # -----------------------------------------------------------------------
    # In bảng kết quả
    # -----------------------------------------------------------------------
    w = 52
    print()
    print("=" * w)
    print(f"{'KẾT QUẢ ĐÁNH GIÁ — SQuAD-style (ViQuAD)':^{w}}")
    print("=" * w)
    print(f"  Tổng câu          : {results['n_total']:>6}")
    print(f"  Câu hợp lệ        : {results['n_valid']:>6}")
    print(f"  Bỏ qua (ref rỗng) : {results['n_skipped']:>6}")
    print("-" * w)
    print(f"  {'Chỉ số':<22}  {'Giá trị':>8}   {'Diễn giải'}")
    print("-" * w)
    print(f"  {'Exact Match (EM)':<22}  {results['EM']:>7.2f}%   Khớp 100% sau normalize")
    print(f"  {'Token F1':<22}  {results['F1']:>7.2f}%   Token overlap (soft match)")
    print(f"  {'Precision':<22}  {results['Precision']:>7.2f}%   Tỷ lệ token đúng / pred")
    print(f"  {'Recall':<22}  {results['Recall']:>7.2f}%   Tỷ lệ token GT được tìm ra")
    print(f"  {'ROUGE-L':<22}  {results['ROUGE-L']:>7.2f}%   LCS — đánh giá thứ tự từ")
    print("=" * w)
    print()
    print("  Tham chiếu chuẩn (trên ViQuAD test set):")
    print("    Hệ thống tốt (vi-mrc-large fine-tune) : EM ~75%, F1 ~85%")
    print("    RAG system (extractive, không fine-tune): EM ~35–55%, F1 ~55–70%")
    print()
    print("  Lưu ý: EM là chỉ số CỨNG nhất — yêu cầu khớp hoàn toàn.")
    print("         F1 >= EM luôn đúng. Khoảng cách F1−EM cho thấy hệ")
    print("         thống trả lời 'gần đúng' ở mức nào.")
    print("=" * w)

    return results


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    PREDICTION_FILE = "system_outputs/system_output_1.txt"
    REFERENCE_FILE  = "data/test/reference_answers.txt"
    evaluate_system(PREDICTION_FILE, REFERENCE_FILE)