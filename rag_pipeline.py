import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
DOCS_DIR = "raw_data"
TEST_QUESTIONS_PATH = "data/test/questions.txt"
OUTPUT_DIR = "system_outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "system_output_1.txt")

# ==========================================
# KHỞI TẠO MÔ HÌNH THỦ CÔNG (Bỏ qua pipeline)
# ==========================================
print("Đang tải Embedder Model...")
embedder_model = SentenceTransformer('keepitreal/vietnamese-sbert') 

print("Đang tải QA Reader Model...")
qa_model_name = "timpal0l/mdeberta-v3-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# ==========================================
# 1. ĐỌC VÀ LẬP CHỈ MỤC TÀI LIỆU (FAISS)
# ==========================================
print("Đang nạp ngữ cảnh từ thư mục raw_data...")
documents = []
for filename in os.listdir(DOCS_DIR):
    if filename.endswith(".txt"):
        with open(os.path.join(DOCS_DIR, filename), 'r', encoding='utf-8') as f:
            documents.append(f.read())

print("Đang lập chỉ mục vector (Indexing)...")
doc_embeddings = embedder_model.encode(documents, convert_to_tensor=False)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# ==========================================
# 2. XỬ LÝ TẬP TEST VÀ TẠO ĐÁP ÁN
# ==========================================
print("\nBắt đầu trả lời câu hỏi...")
with open(TEST_QUESTIONS_PATH, 'r', encoding='utf-8') as f:
    questions = [line.strip() for line in f if line.strip()]

os.makedirs(OUTPUT_DIR, exist_ok=True)
results = []

for idx, question in enumerate(questions):
    # --- BƯỚC RETRIEVER ---
    # Tìm đoạn văn bản (context) giống câu hỏi nhất
    q_embed = embedder_model.encode([question], convert_to_tensor=False)
    _, I = index.search(np.array(q_embed), k=1)
    best_context = documents[I[0][0]]
    
    # --- BƯỚC READER (Dùng PyTorch trực tiếp) ---
    try:
        # Tokenize câu hỏi và ngữ cảnh (giới hạn độ dài 512 token để tránh tràn RAM)
        inputs = tokenizer(question, best_context, return_tensors="pt", truncation=True, max_length=512)
        
        # Đưa qua mô hình
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Tìm vị trí bắt đầu và kết thúc của câu trả lời có xác suất cao nhất
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)
        
        # Giải mã token thành chữ
        if start_idx <= end_idx:
            answer_tokens = inputs.input_ids[0, start_idx : end_idx + 1]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        else:
            answer = ""
            
        results.append(answer if answer.strip() else "Không thể tìm thấy câu trả lời.")
        
    except Exception as e:
        results.append("Không thể tìm thấy câu trả lời do lỗi độ dài.")
        
    if (idx + 1) % 10 == 0:
        print(f"Đã xử lý {idx + 1}/{len(questions)} câu hỏi...")

# ==========================================
# 3. XUẤT KẾT QUẢ
# ==========================================
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for ans in results:
        f.write(ans + "\n")

print(f"\nHoàn tất! Hệ thống đã ghi kết quả vào: {OUTPUT_FILE}")