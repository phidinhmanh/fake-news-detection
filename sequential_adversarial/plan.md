Giai đoạn 1: Input & Trigger (Nguồn tin)
Dữ liệu đầu vào: URL bài báo, đoạn tweet, hoặc file PDF báo cáo tài chính.

Action: Hệ thống bóc tách nội dung thô (Raw text) và chuyển vào Buffer.

Giai đoạn 2: Lead Investigator (Kẻ hoài nghi)
Nhiệm vụ: Trích xuất các "Claims" (Khẳng định) cốt lõi.

Bản chất: Agent này được thiết lập với suspicion_parameter cao. Nó không đọc để hiểu, nó đọc để tìm kẽ hở và các từ ngữ mang tính thao túng cảm xúc (Loaded language).

Giai đoạn 3: Data Analyst (Thư viện số)
Nhiệm vụ: Sử dụng Search Tool (Serper/Google Search) để đối soát.

Action: Tìm ít nhất 3-5 nguồn tin độc lập. Nếu là tin tài chính, nó phải đối chiếu với dữ liệu từ các sàn giao dịch hoặc báo cáo quản trị chính thống.

Giai đoạn 4: Bias Auditor (Kẻ phản biện)
Nhiệm vụ: Kiểm tra "Frame" (Khung định kiến).

Adversarial Logic: Đây là bước quan trọng nhất. Agent này sẽ tranh luận với kết quả của Analyst để tìm ra liệu sự thật có đang bị bóp méo bởi ý đồ chính trị hay kinh tế không.

Giai đoạn 5: Synthesizer (Người đúc kết)
Nhiệm vụ: Tổng hợp báo cáo "Verity Report".

Output: Một cấu trúc gồm: Xác nhận (True/False/Mixed) + Bằng chứng đối ứng + Phân tích định kiến.

Giai đoạn 6: Visual Engine (Minh chứng trực quan)
Công cụ: Mermaid.js & Pillow.

Mục đích: Tự động tạo biểu đồ luồng bằng chứng. Với một nhà đầu tư, việc nhìn thấy "Dòng chảy logic" quan trọng hơn là đọc một đoạn văn dài.

Giai đoạn 7: Persistence (Lưu trữ tri thức)
Action: Lưu kết quả vào cơ sở dữ liệu để phục vụ việc học tập lâu dài (Long-term memory).