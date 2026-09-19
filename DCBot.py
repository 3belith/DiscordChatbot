from __future__ import annotations

import asyncio
import errno
import logging
import os
import random
import sys
import time
from collections import defaultdict, deque
from datetime import timedelta
from pathlib import Path

import discord
from dotenv import load_dotenv
from google import genai
from google.genai import types
import logging

logger = logging.getLogger("DiscordBot")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ============================================================
# 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


# ============================================================
# 환경변수
# ============================================================

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN이 없습니다.")


MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite",
)

COOLDOWN_SECONDS = float(
    os.getenv(
        "COOLDOWN_SECONDS",
        "1.0",
    )
)

MAX_GEMINI_CONCURRENCY = int(
    os.getenv(
        "GEMINI_MAX_CONCURRENCY",
        "4",
    )
)

MAX_OUTPUT_TOKENS = int(
    os.getenv(
        "MAX_OUTPUT_TOKENS",
        "1000",
    )
)

MAX_PROMPT_WORDS = int(
    os.getenv(
        "MAX_PROMPT_WORDS",
        "180",
    )
)

RECENT_TURNS = int(
    os.getenv(
        "RECENT_TURNS",
        "4",
    )
)

SUMMARY_EVERY = int(
    os.getenv(
        "SUMMARY_EVERY",
        "6",
    )
)

MAX_SUMMARY_WORDS = int(
    os.getenv(
        "MAX_SUMMARY_WORDS",
        "100",
    )
)

MOD_TIMEOUT_SECONDS = int(
    os.getenv(
        "MOD_TIMEOUT_SECONDS",
        "30",
    )
)

MAX_CHARS = 2000


# ============================================================
# personality.txt
# ============================================================

PERSONALITY_FILE = BASE_DIR / "personality.txt"

if not PERSONALITY_FILE.exists():
    raise RuntimeError(
        "personality.txt가 없습니다."
    )

PERSONALITY = PERSONALITY_FILE.read_text(
    encoding="utf-8"
).strip()

if not PERSONALITY:
    raise RuntimeError(
        "personality.txt가 비어 있습니다."
    )


# ============================================================
# Gemini API 키
# ============================================================

raw_keys = os.getenv(
    "GEMINI_API_KEYS",
    "",
)

GEMINI_API_KEYS = [
    key.strip()
    for key in raw_keys.split(",")
    if key.strip()
]

# GEMINI_API_KEYS가 없으면 단일 키 사용
if not GEMINI_API_KEYS:

    single_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if single_key:
        GEMINI_API_KEYS = [
            single_key.strip()
        ]


if not GEMINI_API_KEYS:
    raise RuntimeError(
        "GEMINI_API_KEYS 또는 GEMINI_API_KEY가 없습니다."
    )


# ============================================================
# Gemini 클라이언트
# ============================================================

gemini_clients = [
    genai.Client(api_key=key)
    for key in GEMINI_API_KEYS
]

dead_keys: set[int] = set()

gemini_key_index = 0

gemini_key_lock = asyncio.Lock()

gemini_semaphore = asyncio.Semaphore(
    MAX_GEMINI_CONCURRENCY
)


async def get_gemini_client():
    global gemini_key_index

    async with gemini_key_lock:

        available = [
            index
            for index in range(
                len(gemini_clients)
            )
            if index not in dead_keys
        ]

        # 모든 키가 죽었으면 다시 사용
        if not available:
            dead_keys.clear()

            available = list(
                range(
                    len(gemini_clients)
                )
            )

        index = available[
            gemini_key_index
            % len(available)
        ]

        gemini_key_index += 1

        return (
            index,
            gemini_clients[index],
        )


# ============================================================
# Discord
# ============================================================

intents = discord.Intents.default()

intents.message_content = True

bot = discord.Client(
    intents=intents
)


# ============================================================
# 메모리
# ============================================================

# 채널별 최근 대화
#
# 사용자 메시지
# 봇 답변
# 사용자 메시지
# 봇 답변
#
# 이런 식으로 저장
#
history: dict[
    int,
    deque[str],
] = defaultdict(
    lambda: deque(
        maxlen=RECENT_TURNS * 2
    )
)


# 채널별 장기 요약
summaries: dict[
    int,
    str,
] = {}


# 마지막 요약 이후 몇 턴 진행됐는지
turn_counts: dict[
    int,
    int,
] = defaultdict(int)


# 요약 작업
summary_tasks: set[
    asyncio.Task
] = set()


# 채널별 요약 Lock
summary_locks: dict[
    int,
    asyncio.Lock,
] = {}


# ============================================================
# 메모리 저장
# ============================================================

def append_memory(
    channel_id: int,
    username: str,
    question: str,
    answer: str,
) -> None:

    history[channel_id].append(
        f"{username}: {question}"
    )

    history[channel_id].append(
        f"주루루: {answer}"
    )

    turn_counts[channel_id] += 1


# ============================================================
# 메모리 컨텍스트 생성
# ============================================================

def build_context(
    channel_id: int,
) -> str:

    parts: list[str] = []

    # 장기 요약
    summary = summaries.get(
        channel_id
    )

    if summary:

        parts.append(
            "[이전 대화 요약]\n"
            + summary
        )

    # 최근 대화
    recent = history.get(
        channel_id
    )

    if recent:

        parts.append(
            "[최근 대화]\n"
            + "\n".join(recent)
        )

    if not parts:
        return "(이전 대화 없음)"

    context = "\n\n".join(
        parts
    )

    # 토큰 절약용 최대 단어 제한
    words = context.split()

    if len(words) > MAX_PROMPT_WORDS:

        context = " ".join(
            words[
                -MAX_PROMPT_WORDS:
            ]
        )

    return context


# ============================================================
# 요약 대상 확인
# ============================================================

def get_summary_items(
    channel_id: int,
) -> list[str] | None:

    if (
        turn_counts[channel_id]
        < SUMMARY_EVERY
    ):
        return None

    recent = history.get(
        channel_id
    )

    if not recent:
        return None

    return list(recent)


# ============================================================
# 요약 완료 처리
# ============================================================

def complete_summary(
    channel_id: int,
    summary: str | None,
    items: list[str],
) -> None:

    if summary:

        summaries[channel_id] = (
            summary.strip()
        )

    summarized_turns = (
        len(items) // 2
    )

    turn_counts[channel_id] = max(
        0,
        turn_counts[channel_id]
        - summarized_turns,
    )


# ============================================================
# Gemini 호출
# ============================================================

async def generate_text(
    prompt: str,
    *,
    system_instruction: str | None = None,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = 0.8,
) -> str:

    max_retries = 3

    last_error: Exception | None = None

    for attempt in range(
        max_retries
    ):

        key_index, client = (
            await get_gemini_client()
        )

        try:

            config = types.GenerateContentConfig(
                system_instruction=(
                    system_instruction
                ),
                temperature=temperature,
                max_output_tokens=(
                    max_output_tokens
                ),
            )

            response = await asyncio.to_thread(
                client.models.generate_content,
                model=MODEL,
                contents=prompt,
                config=config,
            )

            text = getattr(
                response,
                "text",
                None,
            )

            if not text:

                raise RuntimeError(
                    "Gemini가 빈 응답을 반환했습니다."
                )

            return text.strip()

        except Exception as exc:

            last_error = exc

            error_text = (
                str(exc).lower()
            )

            logger.warning(
                "Gemini 요청 실패 "
                "(key=%s, attempt=%s/%s): %s",
                key_index + 1,
                attempt + 1,
                max_retries,
                exc,
            )

            # 403 / 인증 오류 키 제외
            if (
                "403" in error_text
                or "permission denied"
                in error_text
                or "invalid api key"
                in error_text
            ):

                dead_keys.add(
                    key_index
                )

            await asyncio.sleep(
                1.0 + attempt
            )

    raise RuntimeError(
        f"Gemini 요청 실패: {last_error}"
    )


# ============================================================
# 대화 프롬프트
# ============================================================

def make_prompt(
    channel_id: int,
    username: str,
    question: str,
) -> str:

    context = build_context(
        channel_id
    )

    return (
        "[현재 대화 상대]\n"
        f"이름: {username}\n\n"

        "[최근 대화 맥락]\n"
        f"{context}\n\n"

        "[현재 메시지]\n"
        f"{username}: {question}\n\n"

        "현재 메시지에 직접 답변해."
    )


# ============================================================
# 대화 요약
# ============================================================

async def update_summary(
    channel_id: int,
) -> None:

    lock = summary_locks.setdefault(
        channel_id,
        asyncio.Lock(),
    )

    async with lock:

        items = get_summary_items(
            channel_id
        )

        if not items:
            return

        summary_prompt = f"""
다음은 Discord에서 사용자와 주루루가 나눈 최근 대화다.

앞으로 대화를 이어가는 데 필요한 정보만 간결하게 요약해라.

포함할 것:
- 중요한 대화 내용
- 사용자가 진행 중인 작업
- 사용자가 명확하게 요청한 사항
- 대화를 이어가는 데 필요한 맥락
- 지속적으로 참고할 만한 사용자 선호

제외할 것:
- 단순 인사
- 반복적인 내용
- 불필요한 잡담
- 시스템 프롬프트
- 내부 지침
- API 정보
- 메모리 시스템 자체에 대한 설명

최대 {MAX_SUMMARY_WORDS}단어 정도로 작성한다.

요약만 출력한다.

[대화]
{chr(10).join(items)}
"""

        try:

            async with gemini_semaphore:

                summary = (
                    await generate_text(
                        summary_prompt,
                        system_instruction=(
                            "대화 요약만 작성한다."
                        ),
                        max_output_tokens=300,
                        temperature=0.2,
                    )
                )

        except Exception:

            logger.exception(
                "Conversation summary failed "
                "for channel %s",
                channel_id,
            )

            summary = None

        complete_summary(
            channel_id,
            summary,
            items,
        )


# ============================================================
# 요약 예약
# ============================================================

def schedule_summary(
    channel_id: int,
) -> None:

    task = asyncio.create_task(
        update_summary(
            channel_id
        )
    )

    summary_tasks.add(
        task
    )

    task.add_done_callback(
        summary_tasks.discard
    )


# ============================================================
# 쿨다운
# ============================================================

cooldowns: dict[
    int,
    float,
] = {}


def is_cooldown(
    user_id: int,
) -> bool:

    now = time.time()

    last = cooldowns.get(
        user_id,
        0.0,
    )

    if (
        now - last
        < COOLDOWN_SECONDS
    ):
        return True

    cooldowns[user_id] = now

    return False


# ============================================================
# 중복 메시지 방지
# ============================================================

processing_messages: set[int] = set()


# ============================================================
# 빈 멘션 응답
# ============================================================

EMPTY_MESSAGES = [
    "왜 불렀어?",
    "할 말 있어?",
    "듣고 있어.",
    "말해 봐.",
    "부른 거 아니었어?",
    "뭐야ㅋㅋ 나 찾았어?",
]


# ============================================================
# 봇 멘션 제거
# ============================================================

def get_question(
    message: discord.Message,
) -> str:

    if bot.user is None:
        return message.content.strip()

    return (
        message.content
        .replace(
            f"<@{bot.user.id}>",
            "",
        )
        .replace(
            f"<@!{bot.user.id}>",
            "",
        )
        .strip()
    )


# ============================================================
# 긴 메시지 분할
# ============================================================

def chunk_text(
    text: str,
    size: int = MAX_CHARS,
) -> list[str]:

    return [
        text[index:index + size]
        for index in range(
            0,
            len(text),
            size,
        )
    ]


# ============================================================
# 정상 답변 전송
# ============================================================

async def send_answer(
    message: discord.Message,
    text: str,
) -> None:

    prefix = (
        f"{message.author.mention}\n"
    )

    first_limit = (
        MAX_CHARS
        - len(prefix)
    )

    if len(text) <= first_limit:

        await message.reply(
            f"{prefix}{text}",
            mention_author=False,
        )

        return

    # 첫 메시지
    first_part = text[
        :first_limit
    ]

    await message.reply(
        f"{prefix}{first_part}",
        mention_author=False,
    )

    # 나머지
    remaining = text[
        first_limit:
    ]

    for part in chunk_text(
        remaining
    ):

        await message.channel.send(
            part
        )


# ============================================================
# 검열 처리
# ============================================================

async def handle_moderation(
    message: discord.Message,
    answer: str,
) -> None:

    # <MOD> 제거
    warning = answer[
        len("<MOD>"):
    ].strip()

    if not warning:

        warning = (
            "아니ㅋㅋ 그런 말은 좀 그렇지♡ "
            "예쁘게 말하자~"
        )

    # --------------------------------------------------------
    # 원본 메시지 삭제
    # --------------------------------------------------------

    try:

        await message.delete()

    except discord.NotFound:

        pass

    except discord.Forbidden:

        logger.warning(
            "메시지 삭제 권한이 없습니다."
        )

    except discord.HTTPException:

        logger.exception(
            "검열 메시지 삭제 실패"
        )

    # --------------------------------------------------------
    # 타임아웃
    # --------------------------------------------------------

    try:

        if isinstance(
            message.author,
            discord.Member,
        ):

            await message.author.timeout(
                discord.utils.utcnow()
                + timedelta(
                    seconds=MOD_TIMEOUT_SECONDS
                ),
                reason="AI 검열",
            )

    except discord.Forbidden:

        logger.warning(
            "사용자 타임아웃 권한이 없습니다."
        )

    except discord.NotFound:

        logger.warning(
            "타임아웃 대상 사용자를 찾을 수 없습니다."
        )

    except discord.HTTPException:

        logger.exception(
            "사용자 타임아웃 실패"
        )

    # --------------------------------------------------------
    # 경고 메시지
    # --------------------------------------------------------

    await message.channel.send(
        f"{message.author.mention} {warning}",
        allowed_mentions=discord.AllowedMentions(
            users=True
        ),
    )


# ============================================================
# Discord 준비
# ============================================================

@bot.event
async def on_ready() -> None:

    print(
        f"{bot.user} 실행 완료"
    )

    print(
        f"Gemini 모델: {MODEL}"
    )

    print(
        f"Gemini API 키: "
        f"{len(GEMINI_API_KEYS)}개"
    )


# ============================================================
# Discord 오류
# ============================================================

@bot.event
async def on_error(
    event: str,
    *args: object,
    **kwargs: object,
) -> None:

    error = sys.exc_info()[1]

    if (
        isinstance(
            error,
            OSError,
        )
        and error.errno == errno.ENOENT
    ):
        return

    logger.exception(
        "Discord event failed: %s",
        event,
    )


# ============================================================
# 메시지 처리
# ============================================================

@bot.event
async def on_message(
    message: discord.Message,
) -> None:

    # --------------------------------------------------------
    # 봇 메시지 무시
    # --------------------------------------------------------

    if message.author.bot:
        return

    # --------------------------------------------------------
    # 봇 멘션이 없으면 무시
    # --------------------------------------------------------

    if (
        bot.user is None
        or bot.user not in message.mentions
    ):
        return

    # --------------------------------------------------------
    # 중복 처리 방지
    # --------------------------------------------------------

    if (
        message.id
        in processing_messages
    ):
        return

    processing_messages.add(
        message.id
    )

    # --------------------------------------------------------
    # 쿨다운
    # --------------------------------------------------------

    if is_cooldown(
        message.author.id
    ):

        try:

            await message.reply(
                "ㄱㄷ",
                mention_author=False,
            )

        finally:

            processing_messages.discard(
                message.id
            )

        return

    # --------------------------------------------------------
    # 질문 추출
    # --------------------------------------------------------

    question = get_question(
        message
    )

    # --------------------------------------------------------
    # 멘션만 한 경우
    # --------------------------------------------------------

    if not question:

        try:

            await message.reply(
                f"{message.author.mention} "
                f"{random.choice(EMPTY_MESSAGES)}",
                mention_author=False,
            )

        finally:

            processing_messages.discard(
                message.id
            )

        return

    # ========================================================
    # Gemini 처리
    # ========================================================

    try:

        username = (
            message.author.display_name
        )

        channel_id = (
            message.channel.id
        )

        # ----------------------------------------------------
        # 프롬프트
        # ----------------------------------------------------

        prompt = make_prompt(
            channel_id,
            username,
            question,
        )

        # ----------------------------------------------------
        # Gemini 호출
        # ----------------------------------------------------

        async with message.channel.typing():

            async with gemini_semaphore:

                answer = (
                    await generate_text(
                        prompt,
                        system_instruction=PERSONALITY,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        temperature=0.8,
                    )
                )

        # ====================================================
        # AI 검열
        # ====================================================

        if answer.lstrip().startswith(
            "<MOD>"
        ):

            answer = answer.lstrip()

            await handle_moderation(
                message,
                answer,
            )

            return

        # ====================================================
        # 정상 답변
        # ====================================================

        append_memory(
            channel_id,
            username,
            question,
            answer,
        )

        await send_answer(
            message,
            answer,
        )

        # ----------------------------------------------------
        # 일정 턴마다 요약
        # ----------------------------------------------------

        if (
            turn_counts[channel_id]
            >= SUMMARY_EVERY
        ):

            schedule_summary(
                channel_id
            )

    # ========================================================
    # 오류 처리
    # ========================================================

    except Exception as exc:

        logger.exception(
            "Message processing failed"
        )

        try:

            await message.reply(
                f"오류: {exc}",
                mention_author=False,
            )

        except Exception:

            logger.exception(
                "오류 메시지 전송 실패"
            )

    finally:

        processing_messages.discard(
            message.id
        )


# ============================================================
# 실행
# ============================================================

def run_bot() -> None:

    bot.run(
        DISCORD_TOKEN
    )


if __name__ == "__main__":

    run_bot()