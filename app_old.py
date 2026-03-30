import os
import re
import requests
import pandas as pd
import gradio as gr

from dotenv import load_dotenv
from google import genai
from bs4 import BeautifulSoup as bs

load_dotenv("./.gemini_env")


# ---------------------------
# 텍스트 정리 함수
# ---------------------------
def text_clean(text):
    temp = re.sub(r"</?[^>]+>", "", str(text))
    temp = re.sub(r"[^가-힣a-zA-Z0-9]", " ", temp)
    temp = re.sub(r"\s+", " ", temp).strip()
    return temp


# ---------------------------
# 네이버 뉴스 검색 함수
# ---------------------------
def search_news(keyword):
    url = "https://openapi.naver.com/v1/search/news"
    payload = {
        "query": keyword,
        "display": 10,   # 우선 10개만
        "start": 1,
        "sort": "date"
    }
    headers = {
        "X-Naver-Client-Id": os.getenv("Client_Id"),
        "X-Naver-Client-Secret": os.getenv("Client_Secret")
    }

    r = requests.get(url, params=payload, headers=headers, timeout=10)
    r.raise_for_status()

    data = r.json()
    items = data.get("items", [])

    result = {}
    for item in items:
        for key, value in item.items():
            if key in ("title", "description"):
                value = text_clean(value)
            result.setdefault(key, []).append(value)

    if not result:
        return pd.DataFrame(columns=["title", "originallink"])

    df = pd.DataFrame(result)

    needed_cols = [col for col in ["title", "originallink"] if col in df.columns]
    return df[needed_cols].head(10)


# ---------------------------
# 뉴스 본문 추출 함수
# ---------------------------
def content_extract(news_df):
    full_text = ""
    collected_links = []

    if news_df is None or news_df.empty:
        return full_text, collected_links

    for link in news_df["originallink"]:
        try:
            r = requests.get(link, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
        except Exception:
            continue

        soup = bs(r.text, "lxml")

        # id나 class에 content가 포함된 태그 전부 탐색
        paragraphs = soup.select('[id*="content"], [class*="content"]')

        page_text = []
        for tag in paragraphs:
            txt = text_clean(tag.get_text(" ", strip=True))
            if len(txt) > 20:
                page_text.append(txt)

        if page_text:
            full_text += " ".join(page_text) + "\n"
            collected_links.append(link)

    return full_text.strip(), collected_links


# ---------------------------
# Gemini 요약 함수
# ---------------------------
def summary_gemini(full_text):
    if not full_text.strip():
        return "본문을 추출하지 못했습니다. 다른 키워드로 다시 시도해 주세요."

    prompt = f"""
다음 뉴스 기사들을 주제별로 분류해서 한국어로 정리해줘.

요구사항:
1. 주제별로 묶어라.
2. 각 주제 요약은 500자 이내로 작성해라.
3. 마지막에는 '핀테크 분야 영향'과 '경제 전반 영향'을 구분해서 자세히 분석해라.
4. 전체 답변은 보기 쉽게 제목과 항목을 나눠서 작성해라.

기사 원문:
{full_text}
"""

    api_key = os.getenv("google_api_key")
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return response.text


# ---------------------------
# 챗봇 응답 함수
# ---------------------------
def chatbot_response(message, history):
    keyword = message.strip()

    if not keyword:
        return "검색할 키워드를 입력해 주세요."

    try:
        news_df = search_news(keyword)

        if news_df.empty:
            return f"'{keyword}'에 대한 뉴스 검색 결과가 없습니다."

        full_text, used_links = content_extract(news_df)

        if not full_text:
            return (
                f"'{keyword}' 뉴스는 찾았지만 본문 추출에 실패했어요.\n\n"
                "가능한 원인:\n"
                "- 언론사 페이지 구조가 달라서 본문 선택이 안 됨\n"
                "- 접속 차단 또는 동적 렌더링 페이지\n"
            )

        summary = summary_gemini(full_text)

        news_list_text = "\n".join(
            [f"{i+1}. {title}" for i, title in enumerate(news_df["title"].tolist())]
        )

        link_text = "\n".join([f"- {link}" for link in used_links[:10]])

        final_answer = f"""검색 키워드: {keyword}

[검색된 뉴스 제목]
{news_list_text}

[요약 및 분석]
{summary}

[본문 추출에 사용된 링크]
{link_text}
"""
        return final_answer

    except Exception as e:
        return f"오류가 발생했습니다: {type(e).__name__}: {e}"


# ---------------------------
# 예시 입력
# ---------------------------
examples = [
    ["삼성전자"],
    ["카카오페이"],
    ["비트코인"],
    ["금리"],
    ["핀테크"]
]


# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks(title="뉴스 요약 챗봇") as demo:
    gr.Markdown(
        """
# 뉴스 요약 챗봇
네이버 뉴스 검색 결과를 모아서 기사 본문을 추출한 뒤,
Gemini로 주제별 요약과 핀테크/경제 영향 분석을 제공합니다.
"""
    )

    chatbot = gr.Chatbot(height=500, type="messages")
    msg = gr.Textbox(
        label="검색 키워드 입력",
        placeholder="예: 삼성전자, 비트코인, 금리, 핀테크"
    )
    clear_btn = gr.Button("대화 초기화")

    gr.Examples(
        examples=examples,
        inputs=msg
    )

    def user_submit(user_message, history):
        history = history or []
        history.append({"role": "user", "content": user_message})
        return "", history

    def bot_submit(history):
        user_message = history[-1]["content"]
        bot_message = chatbot_response(user_message, history)
        history.append({"role": "assistant", "content": bot_message})
        return history

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_submit, chatbot, chatbot
    )

    clear_btn.click(lambda: [], None, chatbot, queue=False)

if __name__=="__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)