import sys
import os
from dotenv import load_dotenv
load_dotenv() # Load environment variables FIRST

import time
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Fix Unicode issues on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from config import DATASET_PROCESSED_DIR
from sequential_adversarial.pipeline import SequentialAdversarialPipeline

def run_evaluation(num_samples=100):
    print("="*50)
    print(f"BATT DAU DANH GIA PIPELINE {num_samples} MAU VIFACTCHECK")
    print("="*50)

    # 2. Đảm bảo cấu hình là dùng NVIDIA API
    os.environ["LLM_PROVIDER"] = "nvidia"
    api_key = os.getenv("NVIDIA_API_KEY")
    
    if not api_key:
         raise ValueError("❌ Lỗi: Không tìm thấy NVIDIA_API_KEY trong file .env hoặc biến môi trường. "
                          "Để 'test thật' bạn bắt buộc phải cung cấp API Key.")

    # 3. Khởi tạo Pipeline (Bắt buộc dùng live API)
    pipeline = SequentialAdversarialPipeline(mock=False)

    # 4. Tải tập dữ liệu
    test_path = DATASET_PROCESSED_DIR / 'test.csv'
    if not test_path.exists():
        print(f"❌ Lỗi: Không tìm thấy file dữ liệu tại {test_path}")
        print("Vui lòng chạy python dataset/download_datasets.py trước!")
        return

    df_test = pd.read_csv(test_path).dropna(subset=['text', 'label'])
    print(f"DONE: Da load tap du lieu day du: {len(df_test)} mau.")

    # 5. Lấy mẫu ngẫu nhiên (cân bằng nhãn nếu có thể)
    fake_count = min(num_samples // 2, sum(df_test['label'] == 'fake'))
    real_count = min(num_samples - fake_count, sum(df_test['label'] == 'real'))
    
    df_fake = df_test[df_test['label'] == 'fake'].sample(n=fake_count, random_state=42)
    df_real = df_test[df_test['label'] == 'real'].sample(n=real_count, random_state=42)
    df_subset = pd.concat([df_fake, df_real]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"📊 Tập mẫu chạy thủ công: {len(df_subset)} samples (Fake: {len(df_fake)}, Real: {len(df_real)})")
    
    y_true = []
    y_pred = []
    results = []

    # 6. Vòng lặp test
    for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Processing Pipeline"):
        text = str(row['text'])
        true_label = str(row['label']).strip().lower()
        
        try:
            result = pipeline.run(text)
            
            # Map conclusion từ AI -> nhãn nhị phân
            if result.verity_report:
                conclusion = result.verity_report.conclusion.lower()
                # Tin giả sẽ mang conclusion là 'false' hoặc 'mixed'
                if conclusion == 'true':
                    pred_label = 'real'
                else:
                    pred_label = 'fake'
                
                confidence = result.verity_report.confidence
                reasoning = result.verity_report.evidence_summary
            else:
                pred_label = 'fake'
                confidence = 0.0
                reasoning = "N/A - Parsing Error"
                
        except Exception as e:
            print(f"\nWARNING: Loi o sample #{idx}: {str(e)}")
            pred_label = 'fake'
            confidence = 0.0
            reasoning = str(e)
            
        y_true.append(true_label)
        y_pred.append(pred_label)
        results.append({
            'Index': idx,
            'True Label': true_label.upper(),
            'Predicted': pred_label.upper(),
            'Confidence': confidence
        })
        
        # In nhanh kết quả từng mẫu
        is_correct = "[OK]" if true_label == pred_label else "[WRONG]"
        print(f"[{idx+1}/{len(df_subset)}] {is_correct} True: {true_label.upper()} | Pred: {pred_label.upper()} | Conf: {confidence:.2f}")

        # Thêm delay để tránh Rate Limit API
        if api_key:
            time.sleep(3) # Cấu hình delay (sec) cho LLM Cloud

    # 7. Tổng kết báo cáo
    print("\n" + "="*50)
    print("📊 BÁO CÁO KẾT QUẢ PHÂN LOẠI")
    print("="*50)
    
    labels = ['fake', 'real']
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    
    print("\n🧮 CONFUSION MATRIX")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=[f"True {l.upper()}" for l in labels], columns=[f"Pred {l.upper()}" for l in labels])
    
    if HAS_TABULATE:
        print(tabulate(df_cm, headers='keys', tablefmt='grid'))
    else:
        print(df_cm)
    
    # 8. Phân tích chi tiết các mẫu đánh giá sai
    print("\n🔎 CÁC MẪU ĐÁNH GIÁ SAI (TOP 3):")
    df_errors = pd.DataFrame([{**r, 'Text': df_subset.loc[r['Index'], 'text'][:100] + '...'} for r in results if r['True Label'] != r['Predicted']])
    
    if not df_errors.empty:
        if HAS_TABULATE:
            print(tabulate(df_errors.head(3), headers='keys', tablefmt='grid'))
        else:
            print(df_errors.head(3))
    else:
        print("🎊 Mô hình đánh giá chính xác 100% trên tập này!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate pipeline on ViFactCheck dataset")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to test")
    args = parser.parse_args()
    
    run_evaluation(num_samples=args.num_samples)
