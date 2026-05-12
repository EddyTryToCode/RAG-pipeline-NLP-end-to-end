import pandas as pd
import os

def process_viquad_parquet(train_path, val_path, test_path):
    print("Đang đọc các file Parquet...")
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    # ==========================================
    # 1. TẠO KNOWLEDGE RESOURCE TỪ CẢ 3 TẬP
    # ==========================================
    all_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    unique_contexts = all_df['context'].dropna().unique()
    
    doc_dir = "raw_data"
    os.makedirs(doc_dir, exist_ok=True)
    print(f"Đang trích xuất {len(unique_contexts)} tài liệu ngữ cảnh ra '{doc_dir}'...")
    
    for i, context_text in enumerate(unique_contexts):
        with open(os.path.join(doc_dir, f"doc_{i:05d}.txt"), "w", encoding="utf-8") as f:
            f.write(str(context_text))

    # ==========================================
    # 2. HÀM XUẤT FILE QA CHUẨN CHO VIQUAD 2.0
    # ==========================================
    def export_viquad_format(df, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        
        # BƯỚC QUAN TRỌNG: Lọc bỏ các câu hỏi bẫy (không có đáp án)
        if 'is_impossible' in df.columns:
            df = df[df['is_impossible'] == False]
            
        valid_count = 0
        
        with open(os.path.join(folder_path, "questions.txt"), "w", encoding="utf-8") as f_q, \
             open(os.path.join(folder_path, "reference_answers.txt"), "w", encoding="utf-8") as f_a:
            
            for _, row in df.iterrows():
                question = str(row['question']).replace('\n', ' ').strip()
                ans_data = row['answers']
                
                # Bóc tách text từ trong Dictionary
                if isinstance(ans_data, dict) and 'text' in ans_data and len(ans_data['text']) > 0:
                    texts = ans_data['text']
                    
                    # Nối nhiều đáp án đúng bằng dấu chấm phẩy (;) theo yêu cầu bài tập
                    if isinstance(texts, (list, tuple)):
                        answer_str = ";".join([str(t).replace('\n', ' ').strip() for t in texts])
                    elif hasattr(texts, 'tolist'): # Xử lý nếu là numpy array
                        answer_str = ";".join([str(t).replace('\n', ' ').strip() for t in texts.tolist()])
                    else:
                        answer_str = str(texts).replace('\n', ' ').strip()
                        
                    # Chỉ ghi vào file nếu có đủ cả Hỏi và Đáp
                    if question and answer_str:
                        f_q.write(question + "\n")
                        f_a.write(answer_str + "\n")
                        valid_count += 1
                        
        print(f"  -> Đã xuất thành công {valid_count} cặp Hỏi-Đáp hợp lệ vào {folder_path}/")

    # ==========================================
    # 3. THỰC THI XUẤT FILE
    # ==========================================
    print("\nĐang tạo các file Test/Train...")
    export_viquad_format(df_train, "data/train")
    
    # Mẹo: Tập test của ViQuAD thường bị ẩn đáp án để chống gian lận thi cử. 
    # Nếu chạy tập test mà valid_count = 0, bạn hãy đổi dòng dưới thành df_val nhé!
    export_viquad_format(df_test, "data/test") 

if __name__ == "__main__":
    process_viquad_parquet("train.parquet", "val.parquet", "val.parquet")