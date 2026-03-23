"""
schemas.py — API Contract v1
==============================
Hợp đồng giữa 3 người. Sau khi commit cuối tuần 2, KHÔNG ai thay đổi
mà không thông báo cả team.

Commit ngày 1 bởi Người A.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Input từ UI gửi lên API.

    Attributes:
        text: Nội dung bài viết cần kiểm tra (tối đa 2048 ký tự).
        lang: Ngôn ngữ — mặc định ``'vi'`` nếu không truyền.
    """

    text: str = Field(
        ...,
        max_length=2048,
        description="Nội dung bài viết cần kiểm tra",
    )
    lang: Literal["vi", "en"] = Field(
        default="vi",
        description="Ngôn ngữ của bài viết",
    )


class PredictResponse(BaseModel):
    """Model B trả về cho UI.

    Attributes:
        label: Kết quả phân loại cuối cùng.
        confidence: Độ tin cậy của prediction (0.0–1.0).
        domain: Domain tự động nhận diện từ nội dung.
        shap_tokens: Danh sách (token, weight) để highlight trên UI.
        source_score: Điểm credibility của nguồn (``None`` nếu không có metadata).
    """

    label: Literal["fake", "real"] = Field(
        ...,
        description="Kết quả phân loại cuối cùng",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Độ tin cậy của prediction",
    )
    domain: Literal["politics", "health", "finance", "social"] = Field(
        ...,
        description="Domain tự động nhận diện từ nội dung",
    )
    shap_tokens: list[tuple[str, float]] = Field(
        default_factory=list,
        description="Danh sách (token, weight) để highlight trên UI",
    )
    source_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Điểm credibility của nguồn",
    )
