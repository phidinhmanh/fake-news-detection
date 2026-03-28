import os
from textwrap import dedent
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Ensure you have your API key set in your environment variables:
# os.environ["GOOGLE_API_KEY"] = "your-gemini-api-key"

# ------------------------------------------------------------------
# 1. LLM Setup
# ------------------------------------------------------------------
# Initialize Gemini 1.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    verbose=True
)

# ------------------------------------------------------------------
# 2. Agent Factory (Dependency Inversion & Open/Closed)
# ------------------------------------------------------------------
class VerityAgentFactory:
    """Interface tạo Agent để dễ dàng thay đổi LLM hoặc Backstory"""
    
    def __init__(self, llm):
        self.llm = llm

    def create_investigator(self) -> Agent:
        return Agent(
            role="Lead Investigator",
            goal="Trích xuất các Claims cốt lõi từ văn bản và phát hiện các từ ngữ thao túng cảm xúc (loaded language).",
            backstory=dedent("""\
                Bạn là một nhà báo điều tra lão luyện với 20 năm kinh nghiệm. 
                Bạn không đọc để tin, bạn đọc để tìm kẽ hở và tính từ mang cảm xúc mạnh.
            """),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def create_analyst(self) -> Agent:
        return Agent(
            role="Data Analyst",
            goal="Đối soát các claims tìm được bằng cách tìm kiếm ít nhất 3 nguồn tin độc lập.",
            backstory=dedent("""\
                Bạn là một thư viện số sống. Bạn có khả năng tìm kiếm, đối chiếu dữ liệu
                và xác minh chéo thông tin cực kỳ chính xác và cẩn thận.
            """),
            llm=self.llm,
            verbose=True,
            # In a real app, you would pass SerperDevTool here: tools=[search_tool]
            allow_delegation=False
        )

    def create_auditor(self) -> Agent:
        return Agent(
            role="Bias Auditor",
            goal="Phản biện kết quả của Data Analyst, tìm kiếm định kiến và lỗ hổng logic.",
            backstory=dedent("""\
                Bạn là một chuyên gia phân tích phản biện. Bạn luôn nghi ngờ mọi luồng tin
                và giỏi phát hiện khi sự thật bị bóp méo bởi ý đồ chính trị hay kinh tế.
            """),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def create_synthesizer(self) -> Agent:
        return Agent(
            role="Synthesizer",
            goal="Tổng hợp Verity Report rõ ràng, khách quan dựa trên toàn bộ quá trình điều tra.",
            backstory=dedent("""\
                Bạn là người đúc kết thông tin. Bạn biến những báo cáo phức tạp, lộn xộn
                thành một Verity Report súc tích, dễ hiểu với kết luận rõ ràng (True/False/Mixed).
            """),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )


# ------------------------------------------------------------------
# 3. Pipeline Construction
# ------------------------------------------------------------------
def run_crewai_pipeline(raw_text: str):
    print("Khởi tạo hệ thống Multi-Agent với CrewAI và Gemini 1.5 Flash...\n")
    factory = VerityAgentFactory(llm=llm)

    investigator = factory.create_investigator()
    analyst = factory.create_analyst()
    auditor = factory.create_auditor()
    synthesizer = factory.create_synthesizer()

    # Task 1: Trích xuất 
    investigate_task = Task(
        description=dedent(f"""\
            Đọc kỹ đoạn văn bản sau:
            "{raw_text}"1
            
            Yêu cầu:
            1. Trích xuất 2-3 khẳng định (claims) quan trọng nhất.
            2. Chỉ ra những từ ngữ mang ý thao túng cảm xúc (loaded language).
        """),
        expected_output="Danh sách các claims chính và phân tích ngôn từ thao túng.",
        agent=investigator
    )

    # Task 2: Đối soát (Phụ thuộc vào Task 1)
    analyze_task = Task(
        description=dedent("""\
            Sử dụng kết quả từ Lead Investigator. 
            Mô phỏng việc kiểm tra chéo các claims này với 3 nguồn tin giả lập độc lập.
            Chỉ ra xem các nguồn tin này ủng hộ hay bác bỏ các claims.
        """),
        expected_output="Báo cáo đối soát từ các nguồn độc lập cho từng claim.",
        agent=analyst
    )

    # Task 3: Phản biện (Adversarial Logic - Phụ thuộc vào Task 2)
    audit_task = Task(
        description=dedent("""\
            Đọc kết quả của Data Analyst. 
            Hãy tìm ra ít nhất 2 điểm mà Analyst có thể đã bị đánh lừa bởi dữ liệu bề nổi 
            hoặc do cách đặt vấn đề bị định kiến (Framing). Tranh luận lại với kết quả đó.
        """),
        expected_output="Báo cáo rủi ro định kiến và phản biện lại Data Analyst.",
        agent=auditor
    )

    # Task 4: Đúc kết (Phụ thuộc vào tất cả)
    synthesize_task = Task(
        description=dedent("""\
            Dựa trên kết quả của Investigator, Analyst và Auditor:
            Viết một "Verity Report". 
            Bao gồm:
            - Kết luận (True / False / Mixed)
            - Bằng chứng đối ứng
            - Phân tích định kiến
        """),
        expected_output="Verity Report hoàn chỉnh và trình bày đẹp bằng Markdown.",
        agent=synthesizer
    )

    # Thiết lập Crew
    verity_crew = Crew(
        agents=[investigator, analyst, auditor, synthesizer],
        tasks=[investigate_task, analyze_task, audit_task, synthesize_task],
        process=Process.sequential, # Chạy tuần tự theo quy trình
        verbose=True
    )

    # Thực thi
    print("Bắt đầu thực thi Pipeline...\n")
    result = verity_crew.kickoff()
    
    return result


if __name__ == "__main__":
    # Input Data giả lập
    sample_news = (
        "CÔNG TY XYZ BÁO CÁO LỢI NHUẬN TĂNG TRƯỞNG LỊCH SỬ KHIẾN GIỚI ĐẦU TƯ CHOÁNG VÁNG! "
        "Với tài lãnh đạo tuyệt đỉnh của CEO, quý 3 năm nay công ty XYZ đã tăng doanh thu 500% "
        "bất chấp thị trường ảm đạm. Phố Wall đang điên cuồng săn lùng cổ phiếu này. "
        "Bất kỳ ai không mua ngay bây giờ sẽ hối hận suốt đời!"
    )

    if not os.getenv("GOOGLE_API_KEY"):
        print("CẢNH BÁO: Chưa cấu hình GOOGLE_API_KEY. Vui lòng set biến môi trường này trước khi chạy thật.")
        print("VD: export GOOGLE_API_KEY='your_api_key'\n")
    else:
        final_report = run_crewai_pipeline(sample_news)
        print("====================================")
        print("KẾT QUẢ CUỐI CÙNG (VERITY REPORT):")
        print("====================================")
        print(final_report)
