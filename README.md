# Gemini Discord Bot v2

## 기능

- @봇 멘션으로 대화
- 빈 멘션은 AI 호출 없이 랜덤 응답
- 사용자별 쿨타임
- 쿨타임 중 반복 도배 시 임시 차단
- 사용자별 최근 대화 기억
- N턴마다 장기 기억 요약
- 입력 사전 검열
- AI가 새 검열 규칙을 발견하면 `( MOD )` 형식으로 반환
- AI가 반환한 새 규칙을 moderation.json에 자동 저장
- 저장된 규칙은 다음 요청부터 AI 호출 전에 사전 검열
- 이미 저장된 규칙이 AI 출력에 나오면 출력 후검열
- 여러 Gemini API 키를 순환 사용

## 검열 규칙

moderation.json:

{
  "rules": [
    {
      "word": "예시단어",
      "message": "이 표현은 사용할 수 없어."
    }
  ]
}

사용자가 해당 단어를 입력하면 AI까지 가지 않고:

( MOD )
( 예시단어 )
( 이 표현은 사용할 수 없어. )

형태로 바로 응답한다.

AI가 아직 등록되지 않은 검열 표현을 발견하면:

( MOD )
( 예시단어 )
( 이 표현은 사용할 수 없어. )

형식으로 반환하도록 지시하고, 실제 사용자 입력에 그 단어가 있는 경우에만 moderation.json에 자동 등록한다.

## 설치

pip install -r requirements.txt

## 실행

1. `.env.example`을 `.env`로 복사
2. Discord 토큰과 Gemini API 키 입력
3. `personality.txt` 수정
4. `python DCBot.py`

## 키 로테이션

`GEMINI_API_KEYS=KEY1,KEY2,KEY3`처럼 입력하면 요청마다 순환 선택한다.

여러 키는 정상적으로 관리하는 프로젝트/키의 분산 호출 및 장애 대응에 사용하는 것을 전제로 한다.
같은 프로젝트의 여러 키가 별도 쿼터를 보장하는 것은 아니다.

## 참고

현재 Gemini API는 `google-genai` Python SDK의 `generate_content`를 사용할 수 있다.
2026년 9월 현재 Google은 신규 AI Studio 키를 인증 키로 만들고 있으며, 구형 표준 키 정책도 변경되고 있으므로 현재 유효한 키를 사용하는 것이 좋다.
