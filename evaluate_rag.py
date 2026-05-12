import string
import re
from collections import Counter

def normalize_text(text):
    """
    Chuẩn hóa văn bản: Chuyển chữ thường, xóa dấu câu, dấu chấm, và khoảng trắng thừa.
    Việc này giúp đánh giá công bằng hơn (Ví dụ: "William Pitt." và "William Pitt" được coi là giống nhau).
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punctuation(text):
        # Giữ lại các ký tự tiếng Việt, xóa dấu câu cơ bản
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def remove_extra_whitespace(text):
        return ' '.join(text.split())

    text = text.lower()
    text = remove_punctuation(text)
    text = remove_articles(text)
    text = remove_extra_whitespace(text)
    return text

def exact_match_score(prediction, ground_truths):
    """Tính điểm Exact Match (EM). Khớp 100% thì được 1, sai thì 0."""
    normalized_pred = normalize_text(prediction)
    for gt in ground_truths:
        if normalized_pred == normalize_text(gt):
            return 1.0
    return 0.0

def f1_score(prediction, ground_truths):
    """Tính điểm F1 (và Precision, Recall) dựa trên sự trùng khớp của các từ (tokens)."""
    prediction_tokens = normalize_text(prediction).split()
    
    best_f1 = 0.0
    best_recall = 0.0
    best_precision = 0.0

    for gt in ground_truths:
        gt_tokens = normalize_text(gt).split()
        
        # Nếu cả 2 đều rỗng thì coi như khớp
        if len(prediction_tokens) == 0 or len(gt_tokens) == 0:
            if prediction_tokens == gt_tokens:
                f1 = 1.0; recall = 1.0; precision = 1.0
            else:
                f1 = 0.0; recall = 0.0; precision = 0.0
        else:
            common = Counter(prediction_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            
            if num_same == 0:
                f1 = 0.0; recall = 0.0; precision = 0.0
            else:
                precision = 1.0 * num_same / len(prediction_tokens)
                recall = 1.0 * num_same / len(gt_tokens)
                f1 = (2 * precision * recall) / (precision + recall)
                
        # Giữ lại điểm cao nhất trong số các đáp án đúng (nếu có nhiều đáp án)
        if f1 > best_f1:
            best_f1 = f1
        if recall > best_recall:
            best_recall = recall
        if precision > best_precision:
            best_precision = precision
            
    return best_f1, best_recall, best_precision

def evaluate_system(pred_file, ref_file):
    """Đọc file dự đoán và file đáp án chuẩn để tiến hành chấm điểm."""
    try:
        with open(pred_file, 'r', encoding='utf-8') as f_pred, \
             open(ref_file, 'r', encoding='utf-8') as f_ref:
            
            predictions = [line.strip() for line in f_pred.readlines()]
            references = [line.strip() for line in f_ref.readlines()]
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return

    # Kiểm tra số lượng dòng
    if len(predictions) != len(references):
        print(f"LỖI: Số lượng dòng không khớp! File dự đoán có {len(predictions)} dòng, file đáp án có {len(references)} dòng.")
        # Cắt cho bằng nhau để code không sập (tuy nhiên kết quả sẽ bị sai lệch một chút)
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]

    total_em = 0.0
    total_f1 = 0.0
    total_recall = 0.0
    total_precision = 0.0
    count = len(predictions)

    print(f"Đang chấm điểm cho {count} câu trả lời...")

    for pred, ref_line in zip(predictions, references):
        # Các đáp án tham chiếu cách nhau bằng dấu chấm phẩy (;) theo yêu cầu bài tập [cite: 82, 207]
        ground_truths = [gt.strip() for gt in ref_line.split(';')]
        
        # Bỏ qua các câu mà không có đáp án tham chiếu nào hợp lệ
        if not ground_truths or (len(ground_truths) == 1 and ground_truths[0] == ""):
            count -= 1
            continue

        total_em += exact_match_score(pred, ground_truths)
        
        f1, rec, prec = f1_score(pred, ground_truths)
        total_f1 += f1
        total_recall += rec
        total_precision += prec

    if count == 0:
        print("Không có câu hỏi hợp lệ nào để chấm điểm.")
        return

    # Tính điểm trung bình (trên thang 100)
    avg_em = (total_em / count) * 100
    avg_f1 = (total_f1 / count) * 100
    avg_recall = (total_recall / count) * 100
    avg_precision = (total_precision / count) * 100

    print("\n" + "="*40)
    print(" KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH RAG")
    print("="*40)
    print(f" Tổng số câu đã chấm : {count}")
    print(f" Exact Match (EM)    : {avg_em:.2f}%")
    print(f" Answer Recall       : {avg_recall:.2f}%")
    print(f" Precision           : {avg_precision:.2f}%")
    print(f" F1-Score            : {avg_f1:.2f}%")
    print("="*40)
    print("*(Ghi chú: Lấy 3 chỉ số EM, Recall và F1-Score để đưa vào Báo cáo)*")

if __name__ == "__main__":
    # Đường dẫn tới file dự đoán (đầu ra của hệ thống)
    PREDICTION_FILE = "system_outputs/system_output_1.txt"
    # Đường dẫn tới file đáp án chuẩn
    REFERENCE_FILE = "data/test/reference_answers.txt"
    
    evaluate_system(PREDICTION_FILE, REFERENCE_FILE)