
import os, base64, json, requests
import streamlit as st
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText
import calendar as pycalendar
from datetime import date
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
11. 이메일 본문은 일반 텍스트 형식으로 작성합니다.
    **, # 등 Markdown 서식 기호를 사용하지 않습니다.
12. 이전 대화 내용을 참고하여 후속 질문에 답변합니다.
13. 이메일 발송은 이전 대화와 관계없이
    현재 사용자가 명시적으로 요청했을 때만 수행합니다.
14. 첨부된 회의록에 김 대리의 마감 업무가 있다면
    업무마다 save_deadline 도구를 호출합니다.
15. 회의록에 정확한 마감일이 있는 경우에만
    YYYY-MM-DD 형식으로 저장합니다.
16. "다음 주까지"처럼 정확한 날짜가 불분명하면
    due_date는 null로 전달하고,
    due_text에 원문 표현을 보존합니다.
17. 마감일과 담당 업무를 임의로 만들어내지 않습니다.   
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


# 일정 등록 Function Calling 도구
tools.append({
    "type": "function",
    "name": "save_deadline",
    "description": (
        "첨부된 회의록에서 김 대리에게 배정된 "
        "마감 업무를 발견하면 업무별로 저장합니다. "
        "정확한 날짜를 알 수 없어도 원문 마감 표현을 저장합니다."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "김 대리에게 배정된 업무"
            },
            "due_date": {
                "type": ["string", "null"],
                "description": (
                    "정확히 확인되는 마감일 YYYY-MM-DD. "
                    "날짜가 불확실하면 null"
                )
            },
            "due_text": {
                "type": "string",
                "description": "회의록에 나온 마감일 표현 원문"
            }
        },
        "required": ["task", "due_date", "due_text"],
        "additionalProperties": False
    },
    "strict": True
})


# 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

MAX_TURNS = 10

# 회의 업무 일정 저장
if "deadlines" not in st.session_state:
    st.session_state.deadlines = []

# 지시 영역
with st.form("task_form"):
    uploaded_file = st.file_uploader("텍스트 파일 업로드", type=["txt"])
    audio_value = st.audio_input("무엇을 도와드릴까요?")
    submit = st.form_submit_button("실행")
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
    
    # 이전 대화 기록 추가
    input_messages.extend(st.session_state.messages)

    # 현재 사용자 음성 명령 추가
    input_messages.append({
        "role": "user",
        "content": transcript.text
    })

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
        ]

        tool_outputs = []

        # 도구를 호출하지 않은 경우
        if not tool_calls:
            answer = response.output_text

        # 하나 이상의 도구를 호출한 경우
        else:
            for item in tool_calls:

                args = json.loads(item.arguments)

                # 1. 이메일 발송
                if item.name == "send_email":

                    if not text:
                        result = {
                            "status": "failed",
                            "error": "회의록 파일이 없습니다."
                        }

                    else:
                        result = send_email(
                            to=os.getenv("GMAIL_ADDRESS"),
                            subject=args["subject"],
                            body=args["body"]
                        )

                    if result["status"] == "success":
                        st.success("이메일 전송 완료!")
                    else:
                        st.error("이메일 전송에 실패했습니다.")

                # 2. 회의 업무 일정 저장
                elif item.name == "save_deadline":

                    from datetime import date

                    task = args["task"]
                    due_date = args["due_date"]
                    due_text = args["due_text"]

                    # 날짜 형식 확인
                    if due_date is not None:
                        try:
                            date.fromisoformat(due_date)
                        except (ValueError, TypeError):
                            due_date = None

                    new_task = {
                        "task": task,
                        "due_date": due_date,
                        "due_text": due_text
                    }

                    # 동일한 일정 중복 저장 방지
                    if new_task not in st.session_state.deadlines:
                        st.session_state.deadlines.append(new_task)

                    result = {
                        "status": "success",
                        "message": "업무 일정 저장 완료"
                    }

                else:
                    result = {
                        "status": "failed",
                        "error": "알 수 없는 도구"
                    }

                # 각 도구의 실행 결과 수집
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(
                        result, ensure_ascii=False
                    )
                })

            # 도구 실행 결과를 LLM에 전달
            final_response = client.responses.create(
                model="gpt-4o-mini",
                previous_response_id=response.id,
                input=tool_outputs
            )

            answer = final_response.output_text

        # 사용자 질문 저장
        st.session_state.messages.append({
            "role": "user",
            "content": transcript.text
        })

        # AI 최종 응답 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        # 최근 10개 대화 턴만 유지
        st.session_state.messages = st.session_state.messages[-MAX_TURNS * 2:]

        # 응답 출력
        st.subheader("AI 답변")
        st.write(answer)

        # TTS: 텍스트 → 음성 변환
        if answer and answer.strip():

            with st.spinner("음성을 생성하고 있습니다..."):
                speech = client.audio.speech.create(
                    model="tts-1",
                    voice="nova",
                    input=answer
                )

            # 음성 재생
            st.subheader("AI 음성 답변")
            st.audio(speech.content, format="audio/mp3")


# 사이드바에 대화 히스토리 표시
with st.sidebar:

    # 업무 캘린더
    st.divider()
    st.header("📅 업무 캘린더")
    
    
# 기본 Streamlit 월간 달력

    if "calendar_year" not in st.session_state:
        st.session_state.calendar_year = date.today().year

    if "calendar_month" not in st.session_state:
        st.session_state.calendar_month = date.today().month

    if "selected_deadline_date" not in st.session_state:
        st.session_state.selected_deadline_date = date.today().isoformat()

    year = st.session_state.calendar_year
    month = st.session_state.calendar_month

    # 월 이동 버튼
    prev_col, title_col, next_col = st.columns([1, 3, 1])

    with prev_col:
        if st.button("◀", key="prev_month"):
            if month == 1:
                st.session_state.calendar_year -= 1
                st.session_state.calendar_month = 12
            else:
                st.session_state.calendar_month -= 1
            st.rerun()

    with title_col:
        st.markdown(f"**{year}년 {month}월**")

    with next_col:
        if st.button("▶", key="next_month"):
            if month == 12:
                st.session_state.calendar_year += 1
                st.session_state.calendar_month = 1
            else:
                st.session_state.calendar_month += 1
            st.rerun()

    # 마감일 집합
    deadline_dates = {
        item["due_date"]
        for item in st.session_state.deadlines
        if item["due_date"] is not None
    }

    # 월요일부터 시작하는 달력을 일요일 시작으로 설정
    cal = pycalendar.Calendar(firstweekday=6)

    # 요일 표시
    cols = st.columns(7)
    for col, day_name in zip(
        cols, ["일", "월", "화", "수", "목", "금", "토"]
    ):
        col.write(day_name)

    # 월간 날짜 버튼
    for week in cal.monthdayscalendar(year, month):
        cols = st.columns(7)

        for col, day in zip(cols, week):
            if day == 0:
                continue

            date_str = f"{year}-{month:02d}-{day:02d}"

            label = (
                f"🔴{day}" if date_str in deadline_dates
                else str(day)
            )

            if col.button(
                label,
                key=f"cal_{date_str}",
                use_container_width=True
            ):
                st.session_state.selected_deadline_date = date_str

    selected_date = st.session_state.selected_deadline_date



    # 선택한 날짜의 업무 찾기
    selected_tasks = [
    item for item in st.session_state.deadlines
    if item["due_date"] == selected_date
]

    st.subheader("선택한 날짜의 업무")

    if selected_tasks:
        for item in selected_tasks:
            st.write("📌", item["task"])
    else:
        st.caption("이 날짜에 등록된 업무가 없습니다.")

    # 마감일이 확정된 전체 일정
    st.subheader("전체 마감 일정")

    dated_tasks = [
        item for item in st.session_state.deadlines
        if item["due_date"] is not None
    ]

    for item in sorted(
        dated_tasks,
        key=lambda x: x["due_date"]
    ):
        st.write(
            f"📅 {item['due_date']} — {item['task']}"
        )

    # 날짜가 불명확한 업무
    undated_tasks = [
        item for item in st.session_state.deadlines
        if item["due_date"] is None
    ]

    if undated_tasks:
        st.subheader("날짜 확인 필요")

        for item in undated_tasks:
            st.write(
                f"📌 {item['task']} ({item['due_text']})"
            )
    # 2. 구분선
    st.divider()

    # 3. 대화 히스토리
    st.header("대화 히스토리")

    turn_count = len(st.session_state.messages) // 2
    st.caption(f"최근 대화 {turn_count}/10개")

    if not st.session_state.messages:
        st.write("저장된 대화가 없습니다.")

    else:
        for i in range(0, len(st.session_state.messages), 2):

            user_msg = st.session_state.messages[i]
            ai_msg = st.session_state.messages[i + 1]

            with st.expander(f"대화 {i // 2 + 1}"):
                st.markdown("**사용자**")
                st.write(user_msg["content"])

                st.markdown("**에이블**")
                st.write(ai_msg["content"])
