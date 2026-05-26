import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
DOCS_DIR = "raw_data"
TEST_QUESTIONS_PATH = "data/test/questions.txt"
OUTPUT_DIR = "system_outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "system_output_1.txt")

# ==========================================
# KHỞI TẠO MÔ HÌNH
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang sử dụng thiết bị: {device}")

print("Đang tải Embedder Model (vietnamese-bi-encoder)...")
embedder_model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=device)

print("Đang tải Re-ranker Model (cross-encoder)...")
cross_encoder_model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', device=device)

print("Đang tải QA Reader Model (vi-mrc-large)...")
qa_model_name = "nguyenvulebinh/vi-mrc-large"
tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
model.to(device)
model.eval()

# ==========================================
# 1. ĐỌC VÀ CHUNK TÀI LIỆU
# ==========================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

print("Đang nạp và chunk tài liệu từ raw_data...")
all_chunks = []       # Nội dung từng chunk
chunk_sources = []    # Tên file nguồn (để debug)

filenames = sorted([f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")])
for filename in filenames:
    with open(os.path.join(DOCS_DIR, filename), 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if not text:
        continue
    chunks = text_splitter.split_text(text)
    for chunk in chunks:
        all_chunks.append(chunk)
        chunk_sources.append(filename)

print(f"Tổng số chunks: {len(all_chunks)} (từ {len(filenames)} tài liệu)")

# ==========================================
# 2. LẬP CHỈ MỤC VECTOR (FAISS) VÀ BM25
# ==========================================
print("Đang lập chỉ mục BM25...")
tokenized_corpus = [doc.lower().split() for doc in all_chunks]
bm25 = BM25Okapi(tokenized_corpus)

print("Đang encode và lập chỉ mục vector (FAISS)...")
# Encode batch để nhanh hơn
chunk_embeddings = embedder_model.encode(
    all_chunks,
    batch_size=64,
    convert_to_tensor=False,
    show_progress_bar=True,
    normalize_embeddings=True   # cosine similarity
)

# Dùng IndexFlatIP (Inner Product) với vector đã normalize = cosine similarity
dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(np.array(chunk_embeddings, dtype=np.float32))
print(f"FAISS index sẵn sàng: {index.ntotal} vectors, dim={dim}")

# ==========================================
# 3. HÀM READER — EXTRACT ANSWER SPAN
# ==========================================
def extract_answer(question, context, max_answer_len=50):
    """
    Dùng vi-mrc-large để tìm span trả lời trong context.
    Trả về (answer_text, score).
    """
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=512,
        padding=False
    )
    input_ids = inputs.input_ids[0]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0].cpu().float().numpy()
    end_logits   = outputs.end_logits[0].cpu().float().numpy()
    n_tokens = len(start_logits)

    # Tìm span (start <= end, length <= max_answer_len) có tổng score cao nhất
    # Dùng numpy broadcast để tránh vòng lặp Python chậm
    start_arr = start_logits[:n_tokens - 1]
    best_score = float('-inf')
    best_start = 0
    best_end = 0

    for s in range(1, n_tokens):          # bỏ [CLS] token đầu
        for e in range(s, min(s + max_answer_len, n_tokens)):
            score = start_logits[s] + end_logits[e]
            if score > best_score:
                best_score = score
                best_start = s
                best_end = e

    answer_tokens = input_ids[best_start: best_end + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    return answer, best_score

# ==========================================
# 4. XỬ LÝ TẬP TEST
# ==========================================
print("\nĐang đọc câu hỏi test...")
with open(TEST_QUESTIONS_PATH, 'r', encoding='utf-8') as f:
    questions = [line.strip() for line in f if line.strip()]

print(f"Tổng số câu hỏi: {len(questions)}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_RETRIEVE_FAISS = 20
K_RETRIEVE_BM25 = 20
K_RERANK = 3
results = []

for idx, question in enumerate(questions):
    # --- BƯỚC RETRIEVER (HYBRID) ---
    # 1. FAISS Search
    q_embed = embedder_model.encode(
        [question],
        convert_to_tensor=False,
        normalize_embeddings=True
    )
    _, faiss_I = index.search(np.array(q_embed, dtype=np.float32), k=K_RETRIEVE_FAISS)
    faiss_indices = faiss_I[0].tolist()

    # 2. BM25 Search
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:K_RETRIEVE_BM25].tolist()
    
    # 3. Reciprocal Rank Fusion (RRF)
    rrf_k = 60
    rrf_scores = {}
    
    for rank, doc_idx in enumerate(faiss_indices):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1.0 / (rrf_k + rank + 1)
        
    for rank, doc_idx in enumerate(bm25_indices):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1.0 / (rrf_k + rank + 1)
        
    hybrid_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    candidate_indices = hybrid_indices[:20]

    # --- BƯỚC RE-RANKING ---
    cross_inp = [[question, all_chunks[c_idx]] for c_idx in candidate_indices]
    cross_scores = cross_encoder_model.predict(cross_inp)
    
    reranked_pairs = sorted(zip(candidate_indices, cross_scores), key=lambda x: x[1], reverse=True)
    top_k_indices = [c_idx for c_idx, score in reranked_pairs[:K_RERANK]]

    # --- BƯỚC READER ---
    best_answer = ""
    best_score  = float('-inf')

    for chunk_idx in top_k_indices:
        if chunk_idx < 0 or chunk_idx >= len(all_chunks):
            continue
        context = all_chunks[chunk_idx]
        try:
            answer, score = extract_answer(question, context)
            if score > best_score and answer:
                best_score  = score
                best_answer = answer
        except Exception:
            continue

    results.append(best_answer if best_answer.strip() else "Không tìm thấy câu trả lời.")

    if (idx + 1) % 50 == 0:
        print(f"  Đã xử lý {idx + 1}/{len(questions)} câu hỏi...")

# ==========================================
# 5. XUẤT KẾT QUẢ
# ==========================================
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for ans in results:
        f.write(ans + "\n")

print(f"\nHoàn tất! Kết quả đã lưu vào: {OUTPUT_FILE}")
print(f"Tổng số câu trả lời: {len(results)}")