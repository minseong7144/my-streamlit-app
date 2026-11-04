
import os, base64, json, requests
import streamlit as st
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText

# OpenAI 클라이언트 생성
client = OpenAI()

# 페이지 설정
st.set_page_config(page_title="음성 AI 서비스", page_icon="🤖", layout="wide")
st.title("음성 인식 AI 서비스")
st.write("음성으로 질문하시면 답변해 드립니다.")

# 역할 지정
sys_role = """
당신은 '김 대리'의 개인 비서 '에이블'입니다.
김대리가 요청한 업무를 최선을 다해 처리합니다.

사용자가 회의 내용 요약을 요청하면
첨부된 회의 자료를 기반으로 답변하세요.

당신의 역할은 다음과 같습니다.
1. 미팅 내용을 핵심 위주로 요약합니다.
2. 주요 논의 사항과 결정 사항을 정리합니다.
3. 김 대리에게 배정된 업무를 상세하게 알려줍니다.
4. 담당 업무의 마감 기한이 있다면 함께 알려줍니다.
5. 미팅 내용에 없는 정보는 임의로 추가하지 않습니다.
6. 음성으로 전달하기 쉽도록 자연스러운 문장으로 답변합니다.
7. 사용자가 이메일 발송을 명시적으로 요청한 경우에만
   send_email 도구를 사용합니다.
8. 이메일에는 회의 요약, 주요 결정 사항,
   김 대리의 담당 업무와 마감 기한을 포함합니다.
9. 이메일 발송 성공 여부는 도구 실행 결과로 판단합니다.
10. 회의 자료가 없다면 회의 내용을 임의로 작성하지 않습니다.
"""


# 이메일 발송 함수
def send_email(to, subject, body):

    # 환경변수에서 이메일 인증 정보 가져오기
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as s:
            s.starttls()
            s.login(sender, password)
            s.sendmail(sender, [to], msg.as_string())

        return {"status": "success"}

    except Exception as e:
        return {"status": "failed", "error": str(e)}


# Function Calling 도구 정의
tools = [{
    "type": "function",
    "name": "send_email",
    "description": (
        "사용자가 명시적으로 이메일 발송을 요청하면 "
        "회의 요약 이메일을 준비합니다."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "회의 요약 이메일 제목"
            },
            "body": {
                "type": "string",
                "description": "회의 요약 및 담당 업무를 포함한 본문"
            }
        },
        "required": ["subject", "body"],
        "additionalProperties": False
    },
    "strict": True
}]


# 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 지시 영역
with st.form("task_form"):
    uploaded_file = st.file_uploader("텍스트 파일 업로드", type=["txt"])
    audio_value = st.audio_input("무엇을 도와드릴까요?")
    submit = st.form_submit_button("실행")
    recipient_email = st.text_input(
        "이메일 수신자 주소",
        placeholder="example@gmail.com"
    )
# 버튼을 눌렀을 때만 실행
if submit:
    st.session_state.pending_email = None
    # 업로드 된 파일이 있으면 디코딩
    text = ""
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")

    # 녹음이 없으면 녹음 요청
    if audio_value is None:
        st.warning("음성을 녹음해주세요.")
        st.stop()

    # 시스템 메세지
    system_content = sys_role
    if text:
        system_content += f"\n\n다음 내용을 참고하세요:\n{text}"
    input_messages = [{"role": "system", "content": system_content}]

    # STT: 음성 → 텍스트
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", audio_value, "audio/wav"),
        language="ko"
        )

    st.subheader("음성 인식 결과")
    st.write(transcript.text)
    # 사용자 메시지
    input_messages.append({"role": "user", "content": transcript.text})

    # LLM 호출 및 응답 듣기
    with st.spinner("답변을 준비하고 있습니다..."):
        # LLM 호출
        response = client.responses.create(
            model="gpt-4o-mini",
            input=input_messages,
            tools=tools
        )
        
# 도구 호출 확인
tool_calls = [
    item for item in response.output
    if item.type == "function_call"
    and item.name == "send_email"
]

# 이메일 발송 요청이 없는 경우
if not tool_calls:

    answer = response.output_text

# 이메일 발송 요청이 있는 경우
else:

    if not text:
        answer = (
            "미팅 파일이 첨부되지 않았습니다. "
            "회의록 파일을 먼저 업로드해주세요."
        )

    else:
        # Function Calling 결과 추출
        item = tool_calls[0]
        args = json.loads(item.arguments)

        # 김 대리 이메일로 자동 발송
        result = send_email(
            to=os.secrets["USER_EMAIL"],
            subject=args["subject"],
            body=args["body"]
        )

        # 이메일 발송 결과를 LLM에 전달
        tool_outputs = [{
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps(
                result, ensure_ascii=False
            )
        }]

        final_response = client.responses.create(
            model="gpt-4o-mini",
            previous_response_id=response.id,
            input=tool_outputs
        )

        answer = final_response.output_text

        # 실제 발송 결과 표시
        if result["status"] == "success":
            st.success("이메일 전송 완료!")
        else:
            st.error("이메일 전송에 실패했습니다.")

# 응답 출력
st.subheader("AI 답변")
st.write(answer)
