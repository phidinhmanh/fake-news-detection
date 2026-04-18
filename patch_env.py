"""
patch_env.py — Sửa notebook test_system_colab.ipynb
=====================================================
1. Cell Bước 3: Dùng dotenv load GOOGLE_API_KEY + HF_TOKEN thay vì hardcode
2. Cell Bước 6: Sửa code in kết quả cho khớp với PipelineResult model
3. Cell Bước 7: Thêm HF login trước khi tải PhoBERT
4. Xoá toàn bộ outputs cũ (chứa lỗi/traceback) để notebook sạch khi mở
"""

import json

nb_path = 'notebooks/test_system_colab.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ──────────────────────────────────────────────────────────────────────
# Helper: Clear outputs + reset execution_count
# ──────────────────────────────────────────────────────────────────────
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# ──────────────────────────────────────────────────────────────────────
# Cell: Bước 2 — Thêm python-dotenv, huggingface_hub nếu chưa có
# ──────────────────────────────────────────────────────────────────────
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if '!pip install -q underthesea' in src:
        # Đảm bảo có python-dotenv và huggingface_hub
        if 'python-dotenv' not in src:
            src = src.rstrip()
            src += ' python-dotenv huggingface_hub\n'
        cell['source'] = [
            "!pip install -q underthesea transformers datasets lancedb sentence-transformers google-generativeai langdetect wordcloud plotly streamlit pyngrok python-dotenv huggingface_hub\n"
        ]
        break

# ──────────────────────────────────────────────────────────────────────
# Cell: Bước 3 — Dùng dotenv load cả GOOGLE_API_KEY và HF_TOKEN
# ──────────────────────────────────────────────────────────────────────
NEW_ENV_CELL = [
    "import os\n",
    "import sys\n",
    "from pathlib import Path\n",
    "from dotenv import load_dotenv\n",
    "from huggingface_hub import login\n",
    "\n",
    '# Xác định thư mục gốc project\n',
    'ROOT_DIR = Path("..").resolve() if Path(".").resolve().name == "notebooks" else Path(".").resolve()\n',
    "sys.path.append(str(ROOT_DIR))\n",
    'print(f"Project Root: {ROOT_DIR}")\n',
    "\n",
    "# ── Load biến môi trường từ .env ──────────────────────────────────\n",
    'env_path = ROOT_DIR / ".env"\n',
    "if env_path.exists():\n",
    "    load_dotenv(env_path)\n",
    '    print(f"✅ Đã load .env tại: {env_path}")\n',
    "else:\n",
    '    print(f"⚠️ Không tìm thấy .env tại: {env_path}")\n',
    "\n",
    "# ── Gemini API Key ────────────────────────────────────────────────\n",
    'google_api_key = os.getenv("GOOGLE_API_KEY")\n',
    "if google_api_key:\n",
    '    os.environ["GOOGLE_API_KEY"] = google_api_key\n',
    '    print("✅ GOOGLE_API_KEY đã sẵn sàng.")\n',
    "else:\n",
    '    print("⚠️ THIẾU GOOGLE_API_KEY — Agent LLM sẽ dùng mock mode.")\n',
    "\n",
    "# ── Hugging Face Token ────────────────────────────────────────────\n",
    'hf_token = os.getenv("HF_TOKEN")\n',
    "if hf_token:\n",
    '    os.environ["HF_TOKEN"] = hf_token\n',
    "    login(token=hf_token)\n",
    '    print("✅ HF_TOKEN đã sẵn sàng — đã login Hugging Face Hub.")\n',
    "else:\n",
    '    print("⚠️ THIẾU HF_TOKEN — tải model từ HF Hub có thể bị rate-limit.")\n',
]

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    # Bắt cả trường hợp đã patch lẫn chưa patch
    if 'os.environ["GOOGLE_API_KEY"]' in src or 'load_dotenv' in src:
        cell['source'] = NEW_ENV_CELL
        break

# ──────────────────────────────────────────────────────────────────────
# Cell: Bước 6 — Sửa code in kết quả pipeline cho khớp PipelineResult
# ──────────────────────────────────────────────────────────────────────
NEW_PIPELINE_CELL = [
    "from sequential_adversarial.pipeline import SequentialAdversarialPipeline\n",
    "\n",
    "# Để chạy thật, đổi mock_fallback=False (cần GOOGLE_API_KEY hợp lệ)\n",
    "pipeline = SequentialAdversarialPipeline(mock_fallback=True)\n",
    "\n",
    'test_article = """\n',
    "KHẨN CẤP: Vaccine COVID-19 gây ra hàng nghìn ca tử vong trên toàn thế giới! \n",
    "Theo một nghiên cứu bí mật bị rò rỉ, các hãng dược phẩm đã che giấu sự thật \n",
    "về tác dụng phụ nghiêm trọng của vaccine. Hãy chia sẻ thông tin này ngay!!!\n",
    '"""\n',
    "\n",
    'print("🚀 Đang phân tích bài viết...")\n',
    "result = pipeline.run(test_article)\n",
    "\n",
    'print("\\n" + "=" * 60)\n',
    'print("📊 KẾT QUẢ PHÂN TÍCH")\n',
    'print("=" * 60)\n',
    "\n",
    "# ── Verity Report (Stage 5) ──\n",
    "if result.verity_report:\n",
    '    print(f"Conclusion  : {result.verity_report.conclusion}")\n',
    '    print(f"Confidence  : {result.verity_report.confidence:.0%}")\n',
    '    print(f"Evidence    : {result.verity_report.evidence_summary}")\n',
    '    print(f"Bias        : {result.verity_report.bias_summary}")\n',
    "    if result.verity_report.key_findings:\n",
    '        print("Key findings:")\n',
    "        for kf in result.verity_report.key_findings:\n",
    '            print(f"  • {kf}")\n',
    "else:\n",
    '    print("⚠️ Không sinh được Verity Report.")\n',
    "\n",
    "# ── Claims (Stage 2) ──\n",
    'print(f"\\nSố claims trích xuất: {len(result.claims)}")\n',
    "for i, c in enumerate(result.claims, 1):\n",
    '    print(f"  {i}. [{c.suspicion_score:.0%}] {c.text}")\n',
    "\n",
    "# ── Bias Report (Stage 4) ──\n",
    "if result.bias_report:\n",
    '    print(f"\\nBias framing: {result.bias_report.framing}")\n',
    '    print(f"Distortion  : {result.bias_report.distortion_type or \'None\'}")\n',
]

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if 'SequentialAdversarialPipeline' in src:
        cell['source'] = NEW_PIPELINE_CELL
        break

# ──────────────────────────────────────────────────────────────────────
# Cell: Bước 7 — PhoBERT tokenizer (đã có HF login ở Bước 3, chỉ giữ nguyên)
# ──────────────────────────────────────────────────────────────────────
# Không cần sửa source, chỉ cần clear outputs (đã xử lý ở trên)

# ──────────────────────────────────────────────────────────────────────
# Ghi lại file
# ──────────────────────────────────────────────────────────────────────
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Patched notebook successfully!")
print("   - Step 3: dotenv load GOOGLE_API_KEY + HF_TOKEN")
print("   - Step 6: Fixed pipeline output display to match PipelineResult model")
print("   - Cleared all stale outputs (errors/tracebacks)")
