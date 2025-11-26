import os
import re
import difflib
from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS

# [한국어 특화 라이브러리]
from pykospacing import Spacing
from hanspell import spell_checker

# ----------------------------------------
# 1. 설정 및 초기화
# ----------------------------------------
# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # 모든 도메인 허용 (개발용)

# [중요] 띄어쓰기 모델 미리 로드 (서버 시작 시 한 번만 실행 - 속도 최적화)
print("Loading Spacing Model... (잠시만 기다려주세요)")
spacing = Spacing()
print("Model Loaded!")


# ----------------------------------------
# 2. 텍스트 처리 및 교정 함수들
# ----------------------------------------

def preprocess_user_input(text: str) -> str:
    """
    0단계: 전처리 - 더듬는 말, 불필요한 반복 제거 (Regex)
    """
    if not text: return text
    
    # 더듬는 말 제거
    filler_patterns = [r'\b음+\b', r'\b어+\b', r'\b그+게\b', r'\b막\b']
    for pat in filler_patterns:
        text = re.sub(pat, ' ', text).strip()
    
    # 반복 문자 축소 (ㅋㅋㅋ -> ㅋㅋ)
    text = re.sub(r'(ㅋ)\1{2,}', r'\1\1', text)
    text = re.sub(r'(\.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def apply_korean_algorithms(text: str) -> str:
    """
    1단계: 라이브러리 기반 기계적 교정 (띄어쓰기 -> 맞춤법)
    """
    try:
        # 1. 띄어쓰기 (Deep Learning)
        text_spaced = spacing(text)
        
        # 2. 맞춤법 (Naver API wrapper) - 실패 시 띄어쓰기 결과만 반환
        try:
            spelled_sent = spell_checker.check(text_spaced)
            return spelled_sent.checked
        except Exception:
            return text_spaced # 맞춤법 검사 실패 시 띄어쓰기만 적용
            
    except Exception as e:
        print(f"Korean Lib Error: {e}")
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
        elif opcode == 'insert': # 추가된 부분 (초록)
            html_output.append(f"<span style='color:#4caf50; background:#e8f5e9; font-weight:bold;'>{corrected[b0:b1]}</span>")
        elif opcode == 'delete': # 삭제된 부분 (빨강 취소선)
            html_output.append(f"<span style='color:#f44336; text-decoration:line-through; opacity:0.7;'>{original[a0:a1]}</span>")
        elif opcode == 'replace': # 바뀐 부분
            html_output.append(f"<span style='color:#f44336; text-decoration:line-through; opacity:0.7;'>{original[a0:a1]}</span>")
            html_output.append(f"<span style='color:#4caf50; background:#e8f5e9; font-weight:bold;'>{corrected[b0:b1]}</span>")
            
    return "".join(html_output)


def get_corrected_text_with_context(user_input: str, history: list, user_level: str = "intermediate"):
    """
    2단계: 문맥(History)을 고려한 LLM 최종 교정
    """
    # 1. 라이브러리로 기본 오류 수정
    base_corrected = apply_korean_algorithms(user_input)
    
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
        "당신은 외국인을 위한 한국어 교육 전문가 AI입니다. "
        "아래 [대화 흐름]을 참고하여, 학생이 방금 말한 [현재 문장]을 자연스럽게 교정해 주세요.\n\n"
        "### 교정 원칙 ###\n"
        "1. 문법, 오타, 띄어쓰기를 완벽하게 수정하세요.\n"
        "2. [중요] 이전 대화의 문맥(존댓말/반말 여부, 상황)에 맞는 말투로 수정하세요.\n"
        "3. 사용자의 원래 의미는 유지하되, 더 한국인스러운 자연스러운 표현을 쓰세요.\n"
        "4. 부연 설명 없이 **교정된 문장 텍스트만** 출력하세요."
    )
    
    user_prompt = (
        f"### 대화 흐름 ###\n{context_str}\n\n"
        f"### 현재 문장 (기초 교정됨) ###\n{base_corrected}\n\n"
        "### 교정 결과 ###"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        final_correction = response.choices[0].message.content.strip()
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
    context = []
    for k, v in KNOWLEDGE_BASE.items():
        if k in query:
            context.append(v)
    return "\n".join(context) if context else "관련 지식 없음"

def get_chatbot_response_with_rag(corrected_text: str):
    retrieved_context = retrieve_context(corrected_text)
    
    system_msg = (
        "당신은 친절한 한국어 선생님입니다. 학생의 말에 대해 자연스럽게 대답해 주세요. "
        "필요하다면 아래 문법/문화 지식을 참고해서 설명이나 답변을 해주세요.\n"
        f"참고 지식: {retrieved_context}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": corrected_text}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip(), retrieved_context
    except Exception as e:
        return f"오류 발생: {e}", ""


# ----------------------------------------
# 4. API 라우트 (서버 통신)
# ----------------------------------------

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message')
    chat_history = data.get('history', [])  # [핵심] 대화 기록 받기
    user_level = data.get('level', 'intermediate')

    if not user_input:
        return jsonify({"error": "No message"}), 400

    # 1. 전처리
    preprocessed = preprocess_user_input(user_input)

    # 2. 교정 (라이브러리 + LLM + 문맥반영)
    corrected, diff_html = get_corrected_text_with_context(preprocessed, chat_history, user_level)

    # 3. 챗봇 응답 생성
    bot_response, rag_info = get_chatbot_response_with_rag(corrected)

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