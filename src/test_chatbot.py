import os
import re
import difflib
from flask import Flask, request, jsonify
from openai import OpenAI

# CORS 설정
from flask_cors import CORS

# [한국어 특화 라이브러리]
from pykospacing import Spacing
from hanspell import spell_checker

# 로그 설정
import logging
import time 


# ----------------------------------------
# 0. 로깅 설정
# ----------------------------------------
logging.basicConfig(
    level=logging.INFO,  # 필요하면 DEBUG로 변경
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def _short(text: str, maxlen: int = 80) -> str:
    """로그용으로 텍스트 앞부분만 잘라서 표시"""
    if not text:
        return ""
    return (text[:maxlen] + "…") if len(text) > maxlen else text

# ----------------------------------------
# 1. 설정 및 초기화
# ----------------------------------------
# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # 모든 도메인 허용 (개발용)

# [중요] 띄어쓰기 모델 미리 로드 (서버 시작 시 한 번만 실행 - 속도 최적화)
logging.info("Loading Spacing Model... (잠시만 기다려주세요)")
spacing = Spacing()
logging.info("Spacing Model Loaded!")


# ----------------------------------------
# 2. 텍스트 처리 및 교정 함수들
# ----------------------------------------

def preprocess_user_input(text: str) -> str:
    """
    0단계: 전처리 - 더듬는 말, 불필요한 반복 제거 (Regex)
    """
    if not text: return text

    logging.debug(f"[preprocess] original: {_short(text)}")
    
    # 더듬는 말 제거
    filler_patterns = [r'\b음+\b', r'\b어+\b', r'\b그+게\b', r'\b막\b']
    for pat in filler_patterns:
        text = re.sub(pat, ' ', text).strip()
    
    # 반복 문자 축소 (ㅋㅋㅋ -> ㅋㅋ)
    text = re.sub(r'(ㅋ)\1{2,}', r'\1\1', text)
    text = re.sub(r'(\.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()

    logging.debug(f"[preprocess] processed: {_short(text)}")
    return text


def apply_korean_algorithms(text: str) -> str:
    """
    1단계: 라이브러리 기반 기계적 교정 (띄어쓰기 교정)
        - 맞춤법 검사 부분은 삭제
    """
    logging.debug(f"[ko_lib] input: {_short(text)}")
    try:
        # 1. 띄어쓰기 (Deep Learning)
        text_spaced = spacing(text)
        logging.debug(f"[ko_lib] spaced: {_short(text_spaced)}")
        
        # 2. 맞춤법 (Naver API wrapper) - 실패 시 띄어쓰기 결과만 반환
        # try:
        #    spelled_sent = spell_checker.check(text_spaced)
        #    checked = spelled_sent.checked
        #    logging.debug(f"[ko_lib] spelled: {_short(checked)}")
        #    return checked
        #except Exception as e:
        #    logging.warning(f"[ko_lib] spell_checker error: {e}")
        #    return text_spaced  # 맞춤법 검사 실패 시 띄어쓰기만 적용
        return text_spaced
    except Exception as e:
        logging.error(f"[ko_lib] Korean Lib Error: {e}")
        return text


def generate_diff_feedback(original: str, corrected: str) -> str:
    """
    교정 피드백 생성 (HTML 태그로 차이점 강조)
    """
    matcher = difflib.SequenceMatcher(None, original, corrected)
    html_output = []
    
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            html_output.append(original[a0:a1])
        elif opcode == 'insert':  # 추가된 부분 (초록)
            html_output.append(
                f"<span style='color:#4caf50; background:#e8f5e9; font-weight:bold;'>{corrected[b0:b1]}</span>"
            )
        elif opcode == 'delete':  # 삭제된 부분 (빨강 취소선)
            html_output.append(
                f"<span style='color:#f44336; text-decoration:line-through; opacity:0.7;'>{original[a0:a1]}</span>"
            )
        elif opcode == 'replace':  # 바뀐 부분
            html_output.append(
                f"<span style='color:#f44336; text-decoration:line-through; opacity:0.7;'>{original[a0:a1]}</span>"
            )
            html_output.append(
                f"<span style='color:#4caf50; background:#e8f5e9; font-weight:bold;'>{corrected[b0:b1]}</span>"
            )
            
    return "".join(html_output)


def get_corrected_text_with_context(user_input: str, history: list, user_level: str = "intermediate"):
    """
    2단계: 문맥(History)을 고려한 LLM 최종 교정
    """
    logging.info(f"[correct_with_context] start (level={user_level}, history_len={len(history)})")
    logging.debug(f"[correct_with_context] raw_input: {_short(user_input)}")

    # 1. 라이브러리로 기본 오류 수정
    base_corrected = apply_korean_algorithms(user_input)
    logging.debug(f"[correct_with_context] base_corrected: {_short(base_corrected)}")
    
    # 2. 대화 기록을 문자열로 변환 (최근 6개만)
    context_str = ""
    if history:
        for msg in history[-6:]:
            role_name = "학생" if msg['role'] == 'user' else "선생님"
            context_str += f"- {role_name}: {msg['content']}\n"
    else:
        context_str = "(대화 시작)"

    # 3. LLM 프롬프트
    system_prompt = (
    "당신은 외국인을 위한 한국어 문장 교정 전담 AI입니다. "
    "당신의 목표는 문장을 더 예쁘게 바꾸는 것이 아니라, "
    "명백한 오류만 최소한으로 고치는 것입니다.\n\n"
    "### 교정 원칙 ###\n"
    "1. 문장이 이미 문법적으로 자연스럽고 의미 전달에 문제가 없다면, "
    "입력된 문장을 **한 글자도 바꾸지 말고 그대로 반환**합니다.\n"
    "2. 오타, 잘못된 조사, 잘못된 활용, 띄어쓰기 오류 등 "
    "객관적인 오류만 수정합니다. 스타일을 더 공손하게/자연스럽게 만들기 위한 "
    "불필요한 변경은 하지 않습니다.\n"
    "3. 사용자의 의미·정보·뉘앙스를 절대 바꾸지 않습니다. "
    "질문의 형태, 높임말/반말, 말투(의문형/명령형 등)를 유지합니다.\n"
    "   - 예: '당신은 무엇입니까?'는 문법적으로 문제가 없으므로 "
    "그대로 '당신은 무엇입니까?'라고 반환해야 합니다. "
    "이 문장을 '무엇을 물어보시겠어요?'처럼 바꾸지 마십시오.\n"
    "4. 문장을 더 길게 설명하거나, 의미를 추가하거나, 다른 표현으로 의역하지 마십시오. "
    "원래 문장의 구조와 길이를 최대한 유지합니다.\n"
    "5. 수정이 필요한 경우에도, 바뀐 글자 수를 최소로 유지하도록 노력합니다. "
    "한 문장을 여러 문장으로 나누거나, 여러 문장을 하나로 합치는 등의 큰 구조 변경은 "
    "정말 필요할 때만 사용합니다.\n"
    "6. 사용자의 한국어 실력 수준(user_level)은 **표현 난이도 조절**에만 사용하고, "
    "문장의 의미와 말투는 바꾸지 않습니다.\n"
    "7. 고유명사, 숫자, 전문 용어, 의도적인 반복/강조 등은 문제가 없는 한 그대로 둡니다.\n"
    "8. 부연 설명, 분석, 이유 설명 등을 출력하지 말고, "
    "교정된 문장 텍스트만 한 번 출력합니다.\n"
)

    
    user_prompt = (
        f"### 대화 흐름 ###\n{context_str}\n\n"
        f"### 현재 문장 (기초 교정됨) ###\n{base_corrected}\n\n"
        "### 교정 결과 ###"
    )

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        dt = time.time() - t0

        final_correction = response.choices[0].message.content.strip()
        logging.info(f"[correct_with_context] LLM success (elapsed={dt:.2f}s)")
        logging.debug(f"[correct_with_context] final_correction: {_short(final_correction)}")

        # 토큰 사용량 있으면 같이 로그
        usage = getattr(response, "usage", None)
        if usage:
            logging.info(
                f"[correct_with_context] token usage input={usage.prompt_tokens}, "
                f"output={usage.completion_tokens}, total={usage.total_tokens}"
            )
    except Exception as e:
        print(f"LLM Error: {e}")
        final_correction = base_corrected # 에러 시 기계적 교정본 사용

    # 4. Diff 생성
    diff_html = generate_diff_feedback(user_input, final_correction)
    
    return final_correction, diff_html


# ----------------------------------------
# 3. RAG (검색) 및 응답 생성
# ----------------------------------------

# 별도 DB로 대체 필요
KNOWLEDGE_BASE = {
    # ...
    "학습 단계 정보": "학습 단계 정보는 홈페이지 메뉴의 [자기 학습 정보 보기]메뉴로 들어가면 확인할 수 있습니다.",
    "다음 학습 단계": "다음 단계 학습은 현재 단계를 완료해야만 넘어갈 수 있습니다."
    # ...
}

def retrieve_context(query: str) -> str:
    logging.debug(f"[RAG] retrieve_context query: {_short(query)}")
    context = []
    hit_keys = []
    for k, v in KNOWLEDGE_BASE.items():
        if k in query:
            context.append(v)
            hit_keys.append(k)

    if context:
        logging.info(f"[RAG] hit keys: {hit_keys}")
    else:
        logging.info("[RAG] no related knowledge found")

    return "\n".join(context) if context else "관련 지식 없음"

def get_chatbot_response_with_rag(corrected_text: str, history: list) -> tuple:
    """
    # 1. RAG로 관련 문서 찾고
    # 2.  user_input : 이번 턴(교정된) 사용자 발화
        chat_history : 이전 턴 대화 내역 (role/user, assistant 구조)
    """
    logging.info("[chatbot] generating response with RAG")
    logging.debug(f"[chatbot] corrected_text: {_short(corrected_text)}")

    retrieved_context = retrieve_context(corrected_text)
    logging.debug(f"[chatbot] retrieved_context: {_short(retrieved_context)}")

    system_msg = (
        "당신의 이름은 무궁화입니다.\n"
        "당신은 친절한 한국어 선생님입니다. 학생의 말에 대해 자연스럽게 대답해 주세요.\n"
        "필요하다면 아래 지식을 참고해서 설명이나 답변을 해주세요.\n"
        f"참고 지식: {retrieved_context}"
    )

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": corrected_text}
            ],
            temperature=0.1
        )
        dt = time.time() - t0
        logging.info(f"[chatbot] response LLM success (elapsed={dt:.2f}s)")

        usage = getattr(response, "usage", None)
        if usage:
            logging.info(
                f"[chatbot] token usage input={usage.prompt_tokens}, "
                f"output={usage.completion_tokens}, total={usage.total_tokens}"
            )

        return response.choices[0].message.content.strip(), retrieved_context
    except Exception as e:
        logging.error(f"[chatbot] LLM Error: {e}")
        return f"오류 발생: {e}", ""


# ----------------------------------------
# 4. API 라우트 (서버 통신)
# ----------------------------------------

@app.route('/chat', methods=['POST'])
def chat():
    t_start = time.time()

    data = request.get_json()
    user_input = data.get('message')
    chat_history = data.get('history', [])  # [핵심] 대화 기록 받기
    user_level = data.get('level', 'intermediate')

    if not user_input:
        logging.warning("[/chat] empty message received")
        return jsonify({"error": "No message"}), 400
    
    logging.info(
        f"[/chat] request "
        f"level={user_level}, history_len={len(chat_history)}, input_len={len(user_input)}"
    )
    logging.debug(f"[/chat] raw_input: {_short(user_input)}")

    # 1. 전처리
    t0 = time.time()
    preprocessed = preprocess_user_input(user_input)
    t1 = time.time()
    logging.info(f"[/chat] preprocess done (elapsed={t1 - t0:.3f}s)")

    # 2. 교정 (라이브러리 + LLM + 문맥반영)
    corrected, diff_html = get_corrected_text_with_context(preprocessed, chat_history, user_level)
    t2 = time.time()
    logging.info(f"[/chat] correction done (elapsed={t2 - t1:.3f}s)")
    logging.debug(f"[/chat] corrected_input: {_short(corrected)}")

    # 3. 챗봇 응답 생성
    bot_response, rag_info = get_chatbot_response_with_rag(corrected, chat_history)
    t3 = time.time()
    logging.info(f"[/chat] response generation done (elapsed={t3 - t2:.3f}s)")
    logging.debug(f"[/chat] bot_response: {_short(bot_response)}")

    total = t3 - t_start
    logging.info(f"[/chat] total elapsed={total:.3f}s")

    return jsonify({
        "original_input": user_input,
        "corrected_input": corrected,
        "diff_html": diff_html,        # 화면에 보여줄 교정 피드백
        "chatbot_response": bot_response,
        "retrieved_context": rag_info
    })


@app.route('/', methods=['GET'])
def index():
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>한국어 AI 튜터</title>
    <style>
        body { margin:0; padding:0; background:#121212; color:#eee; font-family:'Malgun Gothic', sans-serif; display:flex; justify-content:center; align-items:center; height:100vh; }
        .container { width:1000px; height:650px; display:grid; grid-template-columns: 1fr 350px; gap:20px; }
        
        /* 왼쪽 채팅창 */
        .chat-panel { background:#1e1e1e; border-radius:15px; display:flex; flex-direction:column; padding:20px; }
        #chatbox { flex:1; overflow-y:auto; margin-bottom:15px; padding-right:10px; }
        .msg { padding:10px 15px; border-radius:10px; margin-bottom:10px; max-width:80%; line-height:1.4; }
        .user { background:#ffd700; color:#000; align-self:flex-end; margin-left:auto; }
        .bot { background:#333; color:#fff; align-self:flex-start; }
        
        .input-area { display:flex; gap:10px; }
        input { flex:1; padding:12px; border-radius:8px; border:none; background:#333; color:white; outline:none; }
        button { padding:12px 20px; background:#ffd700; border:none; border-radius:8px; font-weight:bold; cursor:pointer; }
        button:hover { background:#e6c200; }

        /* 오른쪽 분석창 */
        .info-panel { background:#1e1e1e; border-radius:15px; padding:20px; display:flex; flex-direction:column; }
        .info-title { font-size:18px; font-weight:bold; color:#ffd700; margin-bottom:15px; text-align:center; }
        #debugBox { flex:1; overflow-y:auto; font-size:14px; color:#ccc; }
        .debug-item { background:#2a2a2a; padding:10px; margin-bottom:10px; border-radius:8px; }
        .debug-label { font-weight:bold; color:#fff; margin-bottom:5px; display:block; }
    </style>
</head>
<body>
    <div class="container">
        <div class="chat-panel">
            <div id="chatbox">
                <div class="msg bot">안녕하세요! 무엇을 도와드릴까요?</div>
            </div>
            <div class="input-area">
                <input type="text" id="message" placeholder="한국어로 대화를 시작해보세요..." />
                <button id="submitBtn">전송</button>
            </div>
        </div>

        <div class="info-panel">
            <div class="info-title">실시간 AI 분석</div>
            <div id="debugBox">
                <div class="debug-item">대화를 시작하면 이곳에 교정 내용이 표시됩니다.</div>
            </div>
        </div>
    </div>

<script>
    const chatbox = document.getElementById('chatbox');
    const debugBox = document.getElementById('debugBox');
    const msgInput = document.getElementById('message');
    const submitBtn = document.getElementById('submitBtn');

    // [핵심] 대화 문맥(History) 저장용 배열
    let chatHistory = [];

    function addMsg(text, type) {
        const div = document.createElement('div');
        div.classList.add('msg', type);
        div.textContent = text;
        chatbox.appendChild(div);
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    function addDebug(label, htmlContent) {
        const div = document.createElement('div');
        div.classList.add('debug-item');
        div.innerHTML = `<span class="debug-label">${label}</span>${htmlContent}`;
        debugBox.appendChild(div);
        debugBox.scrollTop = debugBox.scrollHeight;
    }

    submitBtn.onclick = function() {
        const text = msgInput.value.trim();
        if(!text) return;

        // 1. 사용자 메시지 표시
        addMsg(text, 'user');
        msgInput.value = '';

        // 2. 서버 전송 (메시지 + 히스토리)
        fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                message: text,
                history: chatHistory,  // 문맥 전송
                level: "intermediate"
            })
        })
        .then(res => res.json())
        .then(data => {
            // 3. 분석 결과 표시 (초기화 후 표시)
            debugBox.innerHTML = "";
            addDebug("📝 원래 문장", data.original_input);
            addDebug("📝 교정 피드백", data.diff_html);
            addDebug("🧠 RAG 지식", data.retrieved_context);

            // 4. 챗봇 응답 표시
            addMsg(data.chatbot_response, 'bot');

            // 5. 히스토리 업데이트 (교정된 문장 + 챗봇 응답)
            chatHistory.push({ "role": "user", "content": data.corrected_input });
            chatHistory.push({ "role": "assistant", "content": data.chatbot_response });
        })
        .catch(err => {
            console.error(err);
            addMsg("오류가 발생했습니다.", 'bot');
        });
    };

    // 엔터키 입력 지원
    msgInput.addEventListener("keypress", (e) => {
        if(e.key === "Enter") submitBtn.click();
    });
</script>
</body>
</html>
    """

if __name__ == '__main__':
    print("=== 한국어 교육 AI 챗봇 서버 시작 (http://127.0.0.1:5000) ===")
    app.run(debug=True)